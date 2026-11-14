# utils/local_rescal_prep.py
import os
import os.path as osp
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils.data_loading import loadData_hyp2
from utils.residuals import (
    find_heldout_closure_events,
    expand_and_merge_events,
    build_sample_mask_from_expanded_idx_intervals,
)


# =========================================================
# Weekly baseline / template
# =========================================================

def compute_weekly_mu_train_nonclosure_simple(
    speed_df_A: pd.DataFrame,
    closure_df_A: pd.DataFrame,
    interval_minutes: int = 5,
) -> np.ndarray:
    """
    Build weekly baseline mu using A/train only and non-closure rows only.

    speed_df_A   : shape (T_A, N), raw speed
    closure_df_A : shape (T_A, N), center closure labels (0/1 or >0 center)
    returns:
        mu : np.ndarray of shape (N, 7, S)
             where S = number of slots per day
    """
    speed_df_A = speed_df_A.copy()
    closure_df_A = closure_df_A.copy()

    speed_df_A.index = pd.to_datetime(speed_df_A.index)
    closure_df_A.index = pd.to_datetime(closure_df_A.index)

    assert len(speed_df_A) == len(closure_df_A), "speed_df_A and closure_df_A length mismatch"
    assert list(speed_df_A.columns) == list(closure_df_A.columns), "column mismatch"

    T_A, N = speed_df_A.shape
    S = (24 * 60) // interval_minutes

    dt = speed_df_A.index
    weekday = dt.dayofweek.to_numpy(dtype=np.int64)
    slot = (dt.hour * (60 // interval_minutes) + (dt.minute // interval_minutes)).to_numpy(dtype=np.int64)

    X = speed_df_A.to_numpy(dtype=np.float32)
    C = closure_df_A.to_numpy(dtype=np.float32)

    sums = np.zeros((N, 7, S), dtype=np.float64)
    counts = np.zeros((N, 7, S), dtype=np.int32)

    for t in range(T_A):
        w = int(weekday[t])
        s = int(slot[t])

        vals = X[t]                      # (N,)
        nonclosure = (C[t] <= 0)        # only non-closure rows
        valid = (~np.isnan(vals)) & nonclosure
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        sums[idx, w, s] += vals[idx]
        counts[idx, w, s] += 1

    mu = np.zeros((N, 7, S), dtype=np.float32)
    nz = counts > 0
    mu[nz] = (sums[nz] / counts[nz]).astype(np.float32)

    # per-sensor fallback
    sensor_sum = np.zeros(N, dtype=np.float64)
    sensor_cnt = np.zeros(N, dtype=np.int64)
    for t in range(T_A):
        vals = X[t]
        valid = (~np.isnan(vals)) & (C[t] <= 0)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        sensor_sum[idx] += vals[idx]
        sensor_cnt[idx] += 1

    sensor_mean = np.divide(sensor_sum, np.maximum(sensor_cnt, 1), dtype=np.float64)
    global_mean = float(sensor_sum.sum() / max(sensor_cnt.sum(), 1))

    miss_i, miss_w, miss_s = np.where(~nz)
    for i, w, s in zip(miss_i, miss_w, miss_s):
        if sensor_cnt[i] > 0:
            mu[i, w, s] = float(sensor_mean[i])
        else:
            mu[i, w, s] = global_mean

    return mu


def weekly_mu_for_timestamps(
    mu: np.ndarray,
    timestamps: np.ndarray,
    node_indices: List[int],
    interval_minutes: int = 5,
) -> torch.Tensor:
    """
    Build weekly baseline values for the given timestamps and selected nodes.

    mu         : (N, 7, S)
    timestamps : length T array of datetimes
    node_indices: local node ids

    returns:
      baseline_local : torch.Tensor of shape (N_local, T)
    """
    dt = pd.to_datetime(pd.Index(timestamps))
    weekday = dt.dayofweek.to_numpy(dtype=np.int64)
    slot = (dt.hour * (60 // interval_minutes) + (dt.minute // interval_minutes)).to_numpy(dtype=np.int64)

    N_local = len(node_indices)
    T = len(timestamps)
    out = np.zeros((N_local, T), dtype=np.float32)

    for j, n in enumerate(node_indices):
        out[j, :] = mu[n, weekday, slot]

    return torch.from_numpy(out)


# =========================================================
# Local graph helpers
# =========================================================

def get_local_nodes_from_distance(
    center_node: int,
    dist_df: pd.DataFrame,
    k_neighbors: int,
) -> List[int]:
    """
    Returns local nodes in fixed order:
      [center, nearest_1, nearest_2, ..., nearest_k]

    dist_df must be square with same node order as dataset columns.
    """
    dist_row = dist_df.iloc[center_node].to_numpy(dtype=np.float32)  # (N,)
    idx = np.arange(len(dist_row))

    # exclude center
    mask = idx != center_node
    nbrs = idx[mask]

    # nearest by ascending distance
    nbrs = nbrs[np.argsort(dist_row[mask])]

    chosen = nbrs[:k_neighbors].tolist()
    return [int(center_node)] + [int(x) for x in chosen]


def slice_local_adjacency(
    A_global: np.ndarray,
    local_nodes: List[int],
) -> torch.Tensor:
    """
    Slice local adjacency from global adjacency.
    returns: (N_local, N_local)
    """
    A_local = A_global[np.ix_(local_nodes, local_nodes)].astype(np.float32)
    return torch.from_numpy(A_local)


# =========================================================
# Event-center assignment
# =========================================================

def expand_events_without_merging(
    events: List[Dict],
    total_len: int,
    pre_steps: int,
    post_steps: int,
) -> List[Dict]:
    """
    Expand each event but DO NOT merge.
    We need individual events to assign a closure center to each sample.
    """
    out = []
    for ev in events:
        s = max(0, ev["start_idx"] - pre_steps)
        e = min(total_len - 1, ev["end_idx"] + post_steps)
        out.append({
            "sensor_idx": int(ev["sensor_idx"]),
            "start_idx": int(s),
            "end_idx": int(e),
        })
    return out


def choose_center_for_sample(
    sample_times_x,
    sample_times_y,
    split_index: pd.Index,
    expanded_events: List[Dict],
) -> int | None:
    """
    Choose which held-out closure center this sample belongs to.

    Rule:
      among expanded events overlapping the sample window,
      choose the event with maximum overlap length.
    """
    sample_times = pd.to_datetime(list(sample_times_x) + list(sample_times_y))
    sample_start = sample_times.min()
    sample_end = sample_times.max()

    split_times = pd.to_datetime(split_index)

    best_sensor = None
    best_overlap = -1

    for ev in expanded_events:
        ev_start = split_times[ev["start_idx"]]
        ev_end = split_times[ev["end_idx"]]

        latest_start = max(sample_start, ev_start)
        earliest_end = min(sample_end, ev_end)

        if latest_start <= earliest_end:
            overlap_steps = int((earliest_end - latest_start) / pd.Timedelta(minutes=5)) + 1
            if overlap_steps > best_overlap:
                best_overlap = overlap_steps
                best_sensor = int(ev["sensor_idx"])

    return best_sensor


# =========================================================
# Local ResCAL dataset builder
# =========================================================

def build_local_rescal_dataset_for_fold(
    args,
    fold: Dict,
    k_neighbors: int = 5,
    interval_minutes: int = 5,
):
    """
    Build local closure-centered ResCAL dataset for one fold.

    Uses:
      - saved full-network unseen residual pack
      - full A cached x/y timestamps
      - weekly train baseline
      - local graph slicing

    Output tensors:
      X_local    : (S, Cx, N_local, T)
      U_local    : (S, P,  N_local, T)
      R_local    : (S, P,  N_local)
      BASE_local : (S, P,  N_local)
      Y_local    : (S, P,  N_local)
      A_local    : (S, N_local, N_local)
    """
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    save_dir = osp.join("save", args.dataset_choice, args.model_variant, f"fold_{fold_id}")
    resid_path = osp.join(save_dir, "residuals_unseen_fold_A.pt")
    meta_path = osp.join(save_dir, "residuals_unseen_fold_A_meta.json")

    if not osp.exists(resid_path):
        raise FileNotFoundError(f"Missing residual pack: {resid_path}")

    pack = torch.load(resid_path, map_location="cpu", weights_only=False)
    X = pack["X"]        # (S, Cx, N, T)  currently speed/raw + closure + time
    U = pack["U"]        # (S, P, N, T)
    R = pack["R"]        # (S, P, N)
    BASE = pack["BASE"]  # (S, P, N)
    Y = pack["Y"]        # (S, P, N)

    # full A split and timestamps
    A_train_full, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    t_A = timestamp_list[0]

    # reload raw A cached x/y timestamps to align selected samples
    cache_A = osp.join("data", "processed", args.dataset_choice, args.model_variant, "hyp2_A.pt")
    A_pack_full = torch.load(cache_A, map_location="cpu", weights_only=False)
    x_time_A_all = A_pack_full[2]
    y_time_A_all = A_pack_full[3]

    # raw speed / closure / distance for A only
    speed_df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    dis_df = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    # align columns/order with node_ids from loadData_hyp2
    speed_df = speed_df.loc[:, node_ids]
    con_df = con_df.loc[:, node_ids]

    speed_df_A = speed_df.iloc[:len(t_A)].copy()
    con_df_A = con_df.iloc[:len(t_A)].copy()
    speed_df_A.index = pd.to_datetime(speed_df_A.index)
    con_df_A.index = pd.to_datetime(con_df_A.index)

    # weekly baseline (train non-closure only)
    mu = compute_weekly_mu_train_nonclosure_simple(
        speed_df_A=speed_df_A,
        closure_df_A=con_df_A,
        interval_minutes=interval_minutes,
    )

    # rebuild the SAME selected sample order used when residuals were built
    raw_events = find_heldout_closure_events(
        con_df_A,
        heldout_sensors=heldout,
        center_threshold=0.999
    )

    merged = expand_and_merge_events(
        raw_events,
        total_len=len(con_df_A),
        pre_steps=args.P,
        post_steps=args.T,
    )

    keep_mask = build_sample_mask_from_expanded_idx_intervals(
        x_time_A_all,
        y_time_A_all,
        split_index=con_df_A.index,
        merged_intervals=merged,
    )
    selected_indices = [i for i, keep in enumerate(keep_mask) if keep]

    assert len(selected_indices) == X.shape[0], (
        f"Selected sample count mismatch: selected_indices={len(selected_indices)} vs residual pack={X.shape[0]}"
    )

    x_time_sel = [x_time_A_all[i] for i in selected_indices]
    y_time_sel = [y_time_A_all[i] for i in selected_indices]

    # expanded non-merged events for center assignment
    expanded_events = expand_events_without_merging(
        raw_events,
        total_len=len(con_df_A),
        pre_steps=args.P,
        post_steps=args.T,
    )

    X_local_list = []
    U_local_list = []
    R_local_list = []
    BASE_local_list = []
    Y_local_list = []
    A_local_list = []
    centers_list = []
    local_nodes_list = []

    N_local = k_neighbors + 1

    for s in range(X.shape[0]):
        center = choose_center_for_sample(
            x_time_sel[s],
            y_time_sel[s],
            split_index=con_df_A.index,
            expanded_events=expanded_events,
        )

        if center is None:
            # skip rare ambiguous/no-overlap sample
            continue

        local_nodes = get_local_nodes_from_distance(
            center_node=center,
            dist_df=dis_df,
            k_neighbors=k_neighbors,
        )

        # --- weekly-normalized speed ---
        x_times = pd.to_datetime(x_time_sel[s])  # true input timestamps, length T
        weekly_base_local = weekly_mu_for_timestamps(
            mu=mu,
            timestamps=x_times,
            node_indices=local_nodes,
            interval_minutes=interval_minutes,
        )  # (N_local, T)

        raw_speed_local = X[s, 0, local_nodes, :]           # (N_local, T)
        speed_dev_local = raw_speed_local - weekly_base_local

        # --- other channels from saved residual X ---
        if X.shape[1] == 4:
            closure_local = X[s, 1, local_nodes, :]
            tod_local = X[s, 2, local_nodes, :]
            dow_local = X[s, 3, local_nodes, :]
            X_local = torch.stack(
                [speed_dev_local, closure_local, tod_local, dow_local],
                dim=0
            )  # (4, N_local, T)
        elif X.shape[1] == 3:
            tod_local = X[s, 1, local_nodes, :]
            dow_local = X[s, 2, local_nodes, :]
            X_local = torch.stack(
                [speed_dev_local, tod_local, dow_local],
                dim=0
            )
        else:
            raise ValueError(f"Unexpected X channel count: {X.shape[1]}")

        U_local = U[s, :, local_nodes, :]          # (P, N_local, T)
        R_local = R[s, :, local_nodes]             # (P, N_local)
        BASE_local = BASE[s, :, local_nodes]       # (P, N_local)
        Y_local = Y[s, :, local_nodes]             # (P, N_local)
        A_local = slice_local_adjacency(A_adj, local_nodes)  # (N_local, N_local)

        X_local_list.append(X_local)
        U_local_list.append(U_local)
        R_local_list.append(R_local)
        BASE_local_list.append(BASE_local)
        Y_local_list.append(Y_local)
        A_local_list.append(A_local)
        centers_list.append(int(center))
        local_nodes_list.append(local_nodes)

    if len(X_local_list) == 0:
        raise RuntimeError(f"No local samples built for fold {fold_id}")

    X_local = torch.stack(X_local_list)        # (S, Cx, N_local, T)
    U_local = torch.stack(U_local_list)        # (S, P, N_local, T)
    R_local = torch.stack(R_local_list)        # (S, P, N_local)
    BASE_local = torch.stack(BASE_local_list)  # (S, P, N_local)
    Y_local = torch.stack(Y_local_list)        # (S, P, N_local)
    A_local = torch.stack(A_local_list)        # (S, N_local, N_local)

    out = {
        "X_local": X_local,
        "U_local": U_local,
        "R_local": R_local,
        "BASE_local": BASE_local,
        "Y_local": Y_local,
        "A_local": A_local,
        "centers": centers_list,
        "local_nodes": local_nodes_list,
        "k_neighbors": int(k_neighbors),
        "fold_id": int(fold_id),
        "heldout_sensors": heldout,
    }

    save_path = osp.join(save_dir, f"local_rescal_k{k_neighbors}.pt")
    torch.save(out, save_path)

    meta = {
        "fold_id": int(fold_id),
        "heldout_sensors": heldout,
        "k_neighbors": int(k_neighbors),
        "num_samples": int(X_local.shape[0]),
        "N_local": int(X_local.shape[2]),
        "save_path": save_path,
    }

    meta_save = osp.join(save_dir, f"local_rescal_k{k_neighbors}_meta.json")
    with open(meta_save, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return save_path, meta