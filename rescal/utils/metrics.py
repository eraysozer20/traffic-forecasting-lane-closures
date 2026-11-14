# utils/metrics.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from utils.features import re_normalization


# =========================
# Basic Metrics
# =========================
def metric(pred: np.ndarray, label: np.ndarray):
    """
    pred, label: arrays broadcastable to same shape, typically (S, P, N)
    Returns: (mae, rmse, mape, b) where b is per-node RMSE vector from the original code.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)

        mae = np.nan_to_num(mae)
        mae = np.mean(mae)

        rmse = np.nan_to_num(rmse)
        result = np.mean(rmse, axis=0)          # (P, N) if input is (S,P,N)
        b = np.sqrt(np.mean(result, axis=0))    # (N,)

        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape)
        mape = np.mean(mape)

    return mae, rmse, mape, b


# =========================
# Seen / Unseen Helpers
# =========================
def build_seen_unseen_sets(df_con: pd.DataFrame, test_start: int) -> Tuple[Set[int], Set[int]]:
    """
    df_con: DataFrame (time x N) where 1 means closure on that sensor at that timestamp
    test_start: integer row index where test split begins

    seen = sensors with closure in train+val
    unseen = sensors with closure in test but NOT in train+val
    """
    trainval_has = (df_con.iloc[:test_start] == 1).any(axis=0).to_numpy()
    test_has = (df_con.iloc[test_start:] == 1).any(axis=0).to_numpy()

    seen_mask = trainval_has
    unseen_mask = np.logical_and(test_has, np.logical_not(trainval_has))

    seen = set(np.where(seen_mask)[0].tolist())
    unseen = set(np.where(unseen_mask)[0].tolist())
    return seen, unseen


def build_neighbors_from_distance(df_dis: pd.DataFrame, radius_miles: float) -> List[np.ndarray]:
    """
    Returns neighbors list where:
      neighbors[s] = np.array of sensor indices within radius (excluding itself)
    """
    dis_mat = df_dis.to_numpy(dtype=float)
    N = dis_mat.shape[0]
    neighbors: List[np.ndarray] = []
    for s in range(N):
        idx = np.where((dis_mat[s] <= radius_miles) & (np.arange(N) != s))[0]
        neighbors.append(idx)
    return neighbors


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


# =========================
# Collect preds (ResCAL residual loader)
# =========================
@torch.no_grad()
def collect_preds_on_E_resid(
    rescal,
    loader_E_resid,
    support,
    mean: float,
    std: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    loader yields: X, U, R, BASE, Y
    BASE and Y are normalized (same scale as baseline output).
    Returns denormalized arrays:
      pred_base, pred_final, label each shaped (S, P, N).
    """
    preds_base, preds_final, labels = [], [], []
    rescal.eval()

    for X, U, R, BASE, Y in loader_E_resid:
        X = X.to(args.device)
        U = U.to(args.device)
        BASE = BASE.to(args.device)
        Y = Y.to(args.device)

        R_hat, _ = rescal(X, U, support)  # (B,P,N)
        Y_hat = BASE + R_hat              # (B,P,N) normalized

        preds_base.append(re_normalization(BASE.detach().cpu().numpy(), mean, std))
        preds_final.append(re_normalization(Y_hat.detach().cpu().numpy(), mean, std))
        labels.append(re_normalization(Y.detach().cpu().numpy(), mean, std))

    return (
        np.concatenate(preds_base, axis=0),
        np.concatenate(preds_final, axis=0),
        np.concatenate(labels, axis=0),
    )


# =========================
# Event-based closure metrics (arrays)
# =========================
def event_closure_metrics_from_arrays(
    *,
    preds: np.ndarray,     # (S,P,N) denorm
    labels: np.ndarray,    # (S,P,N) denorm
    df_con: pd.DataFrame,  # (time,N) closure 0/1
    df_dis: pd.DataFrame,  # (N,N) miles
    test_start: int,       # row idx where test begins (in df_con)
    radius_miles: float,
    y_t_list,              # list length S, each length P
    P: int,
) -> Dict:
    """
    For every predicted horizon timestamp ts (from y_t_list),
    check which sensors are closed at ts in df_con.
    For each such closure-event sensor s:
      - compute MAE/RMSE over all sensors
      - compute MAE/RMSE over local neighbors within radius_miles

    Separately aggregate events for:
      - seen sensors (had closures in train+val)
      - unseen sensors (closures only appear in test)
    """
    df_con = _ensure_datetime_index(df_con)

    seen_sensors, unseen_sensors = build_seen_unseen_sets(df_con, test_start)
    neighbors = build_neighbors_from_distance(df_dis, radius_miles)

    assert len(y_t_list) == preds.shape[0], (
        f"y_t_list ({len(y_t_list)}) != num samples ({preds.shape[0]})"
    )
    assert preds.shape == labels.shape, "preds and labels must have same shape (S,P,N)"
    assert preds.shape[1] == P, f"preds second dim must be P={P}, got {preds.shape[1]}"

    abs_err = np.abs(preds - labels)  # (S,P,N)

    acc = {
        "seen_all_mae": [], "seen_all_mse": [],
        "seen_loc_mae": [], "seen_loc_mse": [],
        "unseen_all_mae": [], "unseen_all_mse": [],
        "unseen_loc_mae": [], "unseen_loc_mse": [],
    }

    def add_event(prefix: str, err_vec: np.ndarray, neigh_idx: Optional[np.ndarray]):
        # all-sensors
        acc[f"{prefix}_all_mae"].append(float(np.mean(err_vec)))
        acc[f"{prefix}_all_mse"].append(float(np.mean(np.square(err_vec))))

        # local neighborhood
        if neigh_idx is not None and len(neigh_idx) > 0:
            acc[f"{prefix}_loc_mae"].append(float(np.mean(err_vec[neigh_idx])))
            acc[f"{prefix}_loc_mse"].append(float(np.mean(np.square(err_vec[neigh_idx]))))

    for i in range(preds.shape[0]):
        y_times = y_t_list[i]
        for p in range(P):
            ts = pd.to_datetime(y_times[p])
            if ts not in df_con.index:
                continue

            closed_vec = (df_con.loc[ts].to_numpy() == 1)
            closed_sensors = np.where(closed_vec)[0]
            if closed_sensors.size == 0:
                continue

            # this event contributes for EACH closed sensor s
            for s in closed_sensors:
                if s in unseen_sensors:
                    add_event("unseen", abs_err[i, p, :], neighbors[s])
                elif s in seen_sensors:
                    add_event("seen", abs_err[i, p, :], neighbors[s])

    def summarize(prefix: str) -> Dict[str, float]:
        all_mae = acc[f"{prefix}_all_mae"]; all_mse = acc[f"{prefix}_all_mse"]
        loc_mae = acc[f"{prefix}_loc_mae"]; loc_mse = acc[f"{prefix}_loc_mse"]

        return {
            "events_all": len(all_mae),
            "MAE_all": float(np.mean(all_mae)) if all_mae else float("nan"),
            "RMSE_all": float(math.sqrt(np.mean(all_mse))) if all_mse else float("nan"),
            "events_local": len(loc_mae),
            "MAE_local": float(np.mean(loc_mae)) if loc_mae else float("nan"),
            "RMSE_local": float(math.sqrt(np.mean(loc_mse))) if loc_mse else float("nan"),
        }

    return {
        "radius_miles": radius_miles,
        "seen_sensors_count": len(seen_sensors),
        "unseen_sensors_count": len(unseen_sensors),
        "seen": summarize("seen"),
        "unseen": summarize("unseen"),
    }
