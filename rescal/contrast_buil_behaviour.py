import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import torch

import util

# IMPORTANT: same config fields as your main
DATASET_CFG = {
    "Tyson": dict(
        adj="Tyson/adj.csv",
        data="Tyson/new_speed_5 - order.csv",
        con_data="Tyson/new_construction_5 - order.csv",
        dis_data="Tyson/adj_1.csv",
        num_nodes=131,
        lambda_=3.0,
    ),
    "PEMSD8": dict(
        adj="PEMS08/adj.csv",
        data="PEMS08/speed_cut.csv",
        con_data="PEMS08/closure_cut.csv",
        dis_data="PEMS08/adj_1_miles.csv",
        num_nodes=212,
        lambda_=18.0,
    ),
    "SIM_GRID": dict(
        adj="SIM/GRID/adj.csv",
        data="SIM/GRID/sensor_readings.csv",
        con_data="SIM/GRID/closure_labels.csv",
        dis_data="SIM/GRID/adj_1_miles.csv",
        num_nodes=120,
        lambda_=1.1,
    ),
}

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/")
    p.add_argument("--dataset_choice", type=str, default="Tyson", choices=list(DATASET_CFG.keys()))
    p.add_argument("--model_variant", type=str, default="wz", choices=["wz", "no"])  # util cache path

    # util.loadData_hyp2 expects these, plus adj/data/con_data/dis_data/num_nodes/lambda_ via cfg
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--P", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)

    # Weekly baseline config
    p.add_argument("--interval_minutes", type=int, default=5)
    p.add_argument("--use_diff", action="store_true")
    p.add_argument("--robust_scale", action="store_true")
    p.add_argument("--eps", type=float, default=1e-6)

    # Gap handling
    p.add_argument("--allow_round_to_grid", action="store_true")
    p.add_argument("--max_round_minutes", type=int, default=2)

    # I/O
    p.add_argument("--prep_dir", type=str, default="data/processed_contrastive")
    p.add_argument("--save_dir", type=str, default="data/processed_contrastive")
    return p.parse_args()

def apply_dataset_cfg(args):
    cfg = DATASET_CFG[args.dataset_choice]
    for k, v in cfg.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args

def load_speed_matrix(csv_path: str, node_ids: pd.Index):
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    missing = [c for c in node_ids if c not in df.columns]
    if missing:
        raise ValueError(f"data file missing {len(missing)} node columns. First few: {missing[:8]}")
    return df.loc[:, node_ids].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)

def build_regular_grid(original_dt: pd.DatetimeIndex, interval_minutes: int):
    tmin = original_dt.min()
    tmax = original_dt.max()
    freq = f"{interval_minutes}min"
    return pd.date_range(start=tmin, end=tmax, freq=freq)

def map_to_grid_indices(original_dt: pd.DatetimeIndex,
                        grid: pd.DatetimeIndex,
                        interval_minutes: int,
                        allow_round: bool,
                        max_round_minutes: int):
    if not allow_round:
        grid_map = {ts: i for i, ts in enumerate(grid)}
        return np.array([grid_map.get(ts, -1) for ts in original_dt], dtype=int)

    grid0 = grid[0].value
    step_ns = pd.Timedelta(minutes=interval_minutes).value
    max_round_ns = pd.Timedelta(minutes=max_round_minutes).value

    mapping = np.full(len(original_dt), -1, dtype=int)
    for k, ts in enumerate(original_dt):
        if pd.isna(ts):
            continue
        delta = ts.value - grid0
        idx = int(np.round(delta / step_ns))
        if idx < 0 or idx >= len(grid):
            continue
        nearest = grid[idx]
        if abs(ts.value - nearest.value) <= max_round_ns:
            mapping[k] = idx
    return mapping

def map_matrix_to_grid(X_raw: np.ndarray, mapping_idx: np.ndarray, grid_len: int):
    T_raw, N = X_raw.shape
    X_grid = np.full((grid_len, N), np.nan, dtype=np.float32)
    present = np.zeros((grid_len, N), dtype=bool)
    for t in range(T_raw):
        g = mapping_idx[t]
        if g < 0:
            continue
        row = X_raw[t]
        X_grid[g, :] = row
        present[g, :] = ~np.isnan(row)
    return X_grid, present

def map_event_mask_to_grid(event_mask_raw: np.ndarray, mapping_idx: np.ndarray, grid_len: int):
    T_raw, N = event_mask_raw.shape
    out = np.zeros((grid_len, N), dtype=bool)
    for t in range(T_raw):
        g = mapping_idx[t]
        if g < 0:
            continue
        out[g, :] |= event_mask_raw[t, :]
    return out

def build_train_mask_on_grid(mapping_idx: np.ndarray, train_raw_len: int, grid_len: int):
    train_mask = np.zeros(grid_len, dtype=bool)
    idxs = mapping_idx[:train_raw_len]
    idxs = idxs[idxs >= 0]
    train_mask[idxs] = True
    return train_mask

def compute_weekly_mu_train_nonclosure(
    X_grid: np.ndarray,
    event_mask_grid: np.ndarray,
    grid: pd.DatetimeIndex,
    train_mask_grid: np.ndarray,
    interval_minutes: int,
):
    Tg, N = X_grid.shape
    S = (24 * 60) // interval_minutes

    weekday_grid = grid.weekday.to_numpy(dtype=np.int64)
    slot_grid = (grid.hour * (60 // interval_minutes) + (grid.minute // interval_minutes)).to_numpy(dtype=np.int64)

    sums = np.zeros((N, 7, S), dtype=np.float64)
    counts = np.zeros((N, 7, S), dtype=np.int32)

    train_rows = np.where(train_mask_grid)[0]
    for t in train_rows:
        w = int(weekday_grid[t])
        s = int(slot_grid[t])
        vals = X_grid[t]
        ok = (~event_mask_grid[t]) & (~np.isnan(vals))
        if not np.any(ok):
            continue
        idx = np.where(ok)[0]
        sums[idx, w, s] += vals[idx]
        counts[idx, w, s] += 1

    mu = np.zeros((N, 7, S), dtype=np.float64)
    nz = counts > 0
    mu[nz] = sums[nz] / counts[nz]

    # sensor mean (train non-closure)
    sensor_sum = np.zeros(N, dtype=np.float64)
    sensor_cnt = np.zeros(N, dtype=np.int64)
    for t in train_rows:
        vals = X_grid[t]
        ok = (~event_mask_grid[t]) & (~np.isnan(vals))
        idx = np.where(ok)[0]
        if idx.size == 0:
            continue
        sensor_sum[idx] += vals[idx]
        sensor_cnt[idx] += 1
    sensor_mean = (sensor_sum / np.maximum(sensor_cnt, 1)).astype(np.float64)

    global_mean = float(sensor_sum.sum() / sensor_cnt.sum()) if sensor_cnt.sum() > 0 else 0.0

    # fill missing
    miss_i, miss_w, miss_s = np.where(~nz)
    for i, w, s in zip(miss_i, miss_w, miss_s):
        mu[i, w, s] = sensor_mean[i] if sensor_cnt[i] > 0 else global_mean

    return mu.astype(np.float32), weekday_grid, slot_grid, counts, sensor_mean.astype(np.float32), global_mean

def apply_mu_to_grid(X_grid: np.ndarray, mu: np.ndarray, weekday_grid: np.ndarray, slot_grid: np.ndarray):
    Tg, N = X_grid.shape
    R = np.full_like(X_grid, np.nan, dtype=np.float32)
    for t in range(Tg):
        w = int(weekday_grid[t])
        s = int(slot_grid[t])
        baseline = mu[:, w, s]
        vals = X_grid[t]
        m = ~np.isnan(vals)
        if np.any(m):
            R[t, m] = vals[m] - baseline[m]
    return R

def mad_train_nonclosure(R_grid: np.ndarray, event_mask_grid: np.ndarray, train_mask_grid: np.ndarray, eps: float):
    Tg, N = R_grid.shape
    mad = np.zeros(N, dtype=np.float32)
    train_rows = np.where(train_mask_grid)[0]
    for i in range(N):
        vals = R_grid[train_rows, i]
        ok = (~event_mask_grid[train_rows, i]) & (~np.isnan(vals))
        v = vals[ok]
        if v.size == 0:
            mad[i] = 1.0
            continue
        med = np.median(v)
        mad_i = np.median(np.abs(v - med))
        mad[i] = float(max(mad_i, eps))
    return mad

def main():
    args = apply_dataset_cfg(build_args())

    print("CONFIG:", args.dataset_choice, "variant:", args.model_variant)
    print("adj:", args.adj)
    print("data:", args.data)
    print("con :", args.con_data)
    print("dis :", args.dis_data)
    print("interval_minutes:", args.interval_minutes,
          "use_diff:", args.use_diff,
          "robust_scale:", args.robust_scale,
          "allow_round_to_grid:", args.allow_round_to_grid)

    # Authoritative split + node ordering + timestamps
    A_train, B_val, C_test, C_time, A, mean, std, node_ids, timestamp_list = util.loadData_hyp2(args)

    ts_A, ts_B, ts_C = timestamp_list
    ts_raw = np.concatenate([ts_A, ts_B, ts_C], axis=0)
    T_raw = len(ts_raw)
    train_raw_len = len(ts_A)
    N = len(node_ids)

    # Load raw speed/flow
    X_raw = load_speed_matrix(osp.join(args.root, args.data), node_ids)
    if X_raw.shape[0] != T_raw:
        print(f"[WARN] data rows {X_raw.shape[0]} != ts_raw {T_raw}; using min length.")
        L = min(X_raw.shape[0], T_raw)
        X_raw = X_raw[:L]
        ts_raw = ts_raw[:L]
        T_raw = L
        train_raw_len = min(train_raw_len, T_raw)

    # Load event mask produced by contrastive_prep_data.py
    event_mask_path = osp.join(args.prep_dir, args.dataset_choice, "event_mask.pt")
    if not osp.exists(event_mask_path):
        raise FileNotFoundError(f"Missing {event_mask_path}. Run contrastive_prep_data.py first.")
    event_mask_raw = torch.load(event_mask_path).cpu().numpy().astype(bool)

    # Align time length if needed
    if event_mask_raw.shape[0] != T_raw or event_mask_raw.shape[1] != N:
        print(f"[WARN] event_mask_raw {event_mask_raw.shape} vs expected ({T_raw},{N}); using min.")
        L = min(event_mask_raw.shape[0], T_raw)
        M = min(event_mask_raw.shape[1], N)
        event_mask_raw = event_mask_raw[:L, :M]
        X_raw = X_raw[:L, :M]
        ts_raw = ts_raw[:L]
        T_raw, N = X_raw.shape
        train_raw_len = min(train_raw_len, T_raw)
        node_ids = node_ids[:M]

    # Parse timestamps
    original_dt = pd.to_datetime(pd.Series(ts_raw), infer_datetime_format=True)
    if original_dt.isna().any():
        print(f"[WARN] {int(original_dt.isna().sum())} timestamps failed to parse; those rows will be dropped.")
    original_dt_idx = pd.DatetimeIndex(original_dt)

    # Build grid on parseable timestamps
    grid = build_regular_grid(pd.DatetimeIndex(original_dt.dropna()), args.interval_minutes)
    mapping_idx = map_to_grid_indices(
        original_dt_idx, grid,
        interval_minutes=args.interval_minutes,
        allow_round=args.allow_round_to_grid,
        max_round_minutes=args.max_round_minutes,
    )
    kept = int(np.sum(mapping_idx >= 0))
    dropped = int(np.sum(mapping_idx < 0))
    print("T_raw:", T_raw, "T_grid:", len(grid), f"mapping kept={kept} dropped={dropped}")

    # Map to grid
    X_grid, present_grid = map_matrix_to_grid(X_raw, mapping_idx, grid_len=len(grid))
    event_mask_grid = map_event_mask_to_grid(event_mask_raw, mapping_idx, grid_len=len(grid))
    train_mask_grid = build_train_mask_on_grid(mapping_idx, train_raw_len=train_raw_len, grid_len=len(grid))
    print("train_mask_grid rows:", int(train_mask_grid.sum()))

    # Weekly mu from TRAIN non-closure
    mu, weekday_grid, slot_grid, counts, sensor_mean, global_mean = compute_weekly_mu_train_nonclosure(
        X_grid, event_mask_grid, grid, train_mask_grid, args.interval_minutes
    )
    print("mu shape:", mu.shape, "global_mean:", global_mean)

    # Residual on grid
    R_grid = apply_mu_to_grid(X_grid, mu, weekday_grid, slot_grid)

    # Robust scale using train non-closure residuals
    mad = None
    if args.robust_scale:
        mad = mad_train_nonclosure(R_grid, event_mask_grid, train_mask_grid, eps=args.eps)
        R_grid = (R_grid / mad[None, :]).astype(np.float32)

    # Diff residual (on grid)
    channels = [R_grid]
    if args.use_diff:
        DR = np.full_like(R_grid, np.nan, dtype=np.float32)
        DR[1:, :] = R_grid[1:, :] - R_grid[:-1, :]
        channels.append(DR)

    B = np.stack(channels, axis=-1).astype(np.float32)  # [Tg,N,C]

    finite_r = B[..., 0][~np.isnan(B[..., 0])]
    if finite_r.size > 0:
        print("Residual finite min/max/mean:", float(finite_r.min()), float(finite_r.max()), float(finite_r.mean()))
    else:
        print("[WARN] No finite residual values (check mapping / data).")

    # Save
    save_dir = osp.join(args.save_dir, args.dataset_choice)
    os.makedirs(save_dir, exist_ok=True)

    behavior_pt = osp.join(save_dir, "behavior.pt")
    meta_pt = osp.join(save_dir, "behavior_meta.pt")
    mu_pt = osp.join(save_dir, "baseline_mu.pt")
    grid_event_mask_pt = osp.join(save_dir, "grid_event_mask.pt")

    torch.save(torch.from_numpy(B), behavior_pt)
    torch.save(torch.from_numpy(event_mask_grid), grid_event_mask_pt)
    torch.save(torch.from_numpy(mu), mu_pt)

    torch.save({
        "dataset_choice": args.dataset_choice,
        "interval_minutes": int(args.interval_minutes),
        "use_diff": bool(args.use_diff),
        "robust_scale": bool(args.robust_scale),
        "allow_round_to_grid": bool(args.allow_round_to_grid),
        "max_round_minutes": int(args.max_round_minutes),
        "node_ids": list(map(str, node_ids)),
        "grid_timestamps": list(map(str, grid.to_pydatetime())),
        "raw_timestamps": list(map(str, ts_raw)),
        "mapping_idx": mapping_idx.astype(np.int32),
        "train_raw_len": int(train_raw_len),
        "train_mask_grid_sum": int(train_mask_grid.sum()),
        "mu_shape": tuple(mu.shape),
        "global_mean": float(global_mean),
        "sensor_mean": sensor_mean,
        "counts_nonzero_cells": int(np.sum(counts > 0)),
        "counts_total_cells": int(counts.size),
        "mad": None if mad is None else mad,
    }, meta_pt)

    print("\nSaved:")
    print(" ", behavior_pt)
    print(" ", grid_event_mask_pt)
    print(" ", mu_pt)
    print(" ", meta_pt)

if __name__ == "__main__":
    main()