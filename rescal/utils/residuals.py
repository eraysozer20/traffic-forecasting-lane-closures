# utils/residuals.py
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
import pandas as pd


def generate_time_features(anchor_timestamps, num_nodes, T):
    """
    anchor_timestamps: length-S list/array of timestamps, typically first y timestamp per sample
    returns: (S, T, N, 2)  -> [tod, dow]
    """
    dt = pd.to_datetime(anchor_timestamps)
    minutes = dt.hour * 60 + dt.minute
    tod = (minutes / 1440.0).to_numpy()
    dow = (dt.dayofweek / 6.0).to_numpy()

    num_samples = len(anchor_timestamps)
    time_feature_seq = np.zeros((num_samples, T, 2), dtype=np.float32)

    step_size = 5.0 / 1440.0
    for i in range(T):
        lag = (T - 1) - i
        time_feature_seq[:, i, 0] = (tod - (lag * step_size)) % 1.0
        time_feature_seq[:, i, 1] = dow

    time_feature_seq = torch.from_numpy(time_feature_seq).unsqueeze(2).expand(-1, -1, num_nodes, -1)
    return time_feature_seq


def find_heldout_closure_events(df_con_split, heldout_sensors, center_threshold=0.999):
    """
    Find contiguous closure-center events for held-out sensors.

    Important:
    We detect events only where the held-out sensor itself is truly the closure center.
    Since your closure channel is:
      - 1.0 at closure center
      - (0,1) nearby
    we use >= center_threshold to detect real center activity.

    Returns:
      events = list of dicts with:
        sensor_idx, start_idx, end_idx, start_time, end_time
    """
    con_arr = df_con_split.to_numpy()
    times = pd.to_datetime(df_con_split.index)

    events = []

    for s in heldout_sensors:
        active = con_arr[:, s] >= center_threshold
        start = None

        for i, flag in enumerate(active):
            if flag and start is None:
                start = i
            elif (not flag) and start is not None:
                end = i - 1
                events.append({
                    "sensor_idx": int(s),
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "start_time": times[start],
                    "end_time": times[end],
                })
                start = None

        if start is not None:
            end = len(active) - 1
            events.append({
                "sensor_idx": int(s),
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time": times[start],
                "end_time": times[end],
            })

    return events


def expand_and_merge_events(events, total_len, pre_steps, post_steps):
    """
    Expand each event:
      start -> start - pre_steps
      end   -> end   + post_steps

    Then merge overlaps per fold.

    Returns:
      merged = list of dicts with expanded index/time ranges
    """
    if len(events) == 0:
        return []

    # expand
    expanded = []
    for ev in events:
        s = max(0, ev["start_idx"] - pre_steps)
        e = min(total_len - 1, ev["end_idx"] + post_steps)
        expanded.append({
            "start_idx": s,
            "end_idx": e,
            "start_time": ev["start_time"],
            "end_time": ev["end_time"],
            "sensor_idx": ev["sensor_idx"],
        })

    # sort by start
    expanded = sorted(expanded, key=lambda d: d["start_idx"])

    # merge overlapping intervals
    merged = []
    cur = dict(expanded[0])
    cur["sensor_ids"] = [cur["sensor_idx"]]

    for ev in expanded[1:]:
        if ev["start_idx"] <= cur["end_idx"] + 1:
            cur["end_idx"] = max(cur["end_idx"], ev["end_idx"])
            cur["sensor_ids"].append(ev["sensor_idx"])
        else:
            merged.append(cur)
            cur = dict(ev)
            cur["sensor_ids"] = [cur["sensor_idx"]]

    merged.append(cur)
    return merged


def build_sample_mask_from_intervals(x_time_list, y_time_list, merged_intervals):
    """
    x_time_list, y_time_list come from cached pack:
      - x_time_list[i] shape (T,)
      - y_time_list[i] shape (P,)

    Keep a sample if its combined time window overlaps ANY expanded interval.
    """
    if len(merged_intervals) == 0:
        return np.zeros(len(x_time_list), dtype=bool)

    mask = []

    for xt, yt in zip(x_time_list, y_time_list):
        sample_times = pd.to_datetime(list(xt) + list(yt))
        sample_start = sample_times.min()
        sample_end = sample_times.max()

        keep = False
        for inter in merged_intervals:
            inter_start = pd.to_datetime(inter["start_time"])
            inter_end = pd.to_datetime(inter["end_time"])

            # NOTE:
            # inter["start_time"] and inter["end_time"] above are the original event times.
            # We need expanded times from the split index, so we compute them later outside.
            if (sample_start <= inter_end) and (sample_end >= inter_start):
                keep = True
                break

        mask.append(keep)

    return np.array(mask, dtype=bool)


def build_sample_mask_from_expanded_idx_intervals(x_time_list, y_time_list, split_index, merged_intervals):
    """
    Better version: use expanded idx intervals against the split timestamp index.
    """
    split_times = pd.to_datetime(split_index)

    mask = []
    for xt, yt in zip(x_time_list, y_time_list):
        sample_times = pd.to_datetime(list(xt) + list(yt))
        sample_start = sample_times.min()
        sample_end = sample_times.max()

        keep = False
        for inter in merged_intervals:
            inter_start = split_times[inter["start_idx"]]
            inter_end = split_times[inter["end_idx"]]
            if (sample_start <= inter_end) and (sample_end >= inter_start):
                keep = True
                break

        mask.append(keep)

    return np.array(mask, dtype=bool)


def subset_pack(pack, keep_mask):
    """
    Filter all 4 parts of cached pack:
      pack = (x_set, y_set, x_time_set, y_time_set)
    """
    out = []
    for part in pack:
        if len(part) != len(keep_mask):
            raise ValueError(f"Mask length {len(keep_mask)} != part length {len(part)}")
        out.append([p for p, keep in zip(part, keep_mask) if keep])
    return tuple(out)


@torch.no_grad()
def build_residual_dataset(
    base_model,
    source_loader,
    mean,
    std,
    args,
    anchor_timestamps=None,
    cache_path=None,
    use_existing_closure_channel=True,
):
    """
    Builds ResCAL dataset from a source_loader.

    Output tensors:
      X    : (S, Cx, N, T)
      U    : (S, P,  N, T)
      R    : (S, P,  N)
      BASE : (S, P,  N)
      Y    : (S, P,  N)

    If use_existing_closure_channel=True:
      X channels become:
        [speed, closure_context, tod, dow]   => Cx = 4
    else:
      X channels become:
        [speed, tod, dow]                    => Cx = 3
    """
    if cache_path is not None and osp.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        need = ["X", "U", "R", "BASE", "Y"]
        if all(k in pack for k in need):
            return TensorDataset(pack["X"], pack["U"], pack["R"], pack["BASE"], pack["Y"])
        print("[BuildResid] Cache exists but missing keys -> rebuilding:", cache_path)

    base_model.eval()
    X_list, R_list, BASE_list, Y_list = [], [], [], []

    for x, y in source_loader:
        x = x.to(args.device)
        y = y.to(args.device)

        y_norm = y[:, :, :, 0]         # speed only
        base_out = base_model(x)       # (B,P,N)
        r_norm = y_norm - base_out     # (B,P,N)

        X_list.append(x.detach().cpu())        # (B,T,N,Forig)
        R_list.append(r_norm.detach().cpu())
        BASE_list.append(base_out.detach().cpu())
        Y_list.append(y_norm.detach().cpu())

    X_raw = torch.cat(X_list, dim=0)           # (S,T,N,Forig)
    R = torch.cat(R_list, dim=0).contiguous()  # (S,P,N)
    BASE = torch.cat(BASE_list, dim=0).contiguous()
    Y = torch.cat(Y_list, dim=0).contiguous()

    # Rebuild X for ResCAL
    speed = X_raw[:, :, :, 0:1]                # (S,T,N,1)

    parts = [speed]

    if use_existing_closure_channel:
        closure_ch = X_raw[:, :, :, 1:2]       # existing baseline closure feature
        parts.append(closure_ch)

    if getattr(args, "rescal_use_time_features", 1) and anchor_timestamps is not None:
        valid_timestamps = anchor_timestamps[:X_raw.shape[0]]
        time_feats = generate_time_features(valid_timestamps, args.num_nodes, args.T)
        parts.append(time_feats)

    X = torch.cat(parts, dim=-1)               # (S,T,N,Cx)
    X = X.permute(0, 3, 2, 1).contiguous()     # (S,Cx,N,T)

    # Build U (same Method A logic)
    num_samples = X.shape[0]
    Tin, P, N = args.T, args.P, args.num_nodes
    pad_len = Tin + P
    R_padded = torch.cat([torch.zeros(pad_len, P, N), R], dim=0)

    U_list = []
    for i in range(num_samples):
        current_sample_idx_padded = i + pad_len
        sequence_u = []
        for t in range(Tin):
            global_t_idx = current_sample_idx_padded - (Tin - 1) + t
            step_u = []
            for k in range(P):
                val = R_padded[global_t_idx - (k + 1), k, :]
                step_u.append(val)
            seq_step = torch.stack(step_u)       # (P,N)
            sequence_u.append(seq_step)
        U_sample = torch.stack(sequence_u)       # (Tin,P,N)
        U_sample = U_sample.permute(0, 2, 1)     # (Tin,N,P)
        U_list.append(U_sample)

    U = torch.stack(U_list)                      # (S,Tin,N,P)
    U = U.permute(0, 3, 2, 1).contiguous()       # (S,P,N,Tin)

    K = getattr(args, "rescal_residual_history", None)

    if K is None:
        K = args.P

    K = int(K)

    if K < 1 or K > args.P:
        raise ValueError(f"rescal_residual_history must be between 1 and P={args.P}, got {K}")

    U = U[:, :K, :, :]

    if cache_path is not None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "X": X,
                "U": U,
                "R": R,
                "BASE": BASE,
                "Y": Y,
                "anchor_timestamps": anchor_timestamps,
            },
            cache_path,
        )

    return TensorDataset(X, U, R, BASE, Y)