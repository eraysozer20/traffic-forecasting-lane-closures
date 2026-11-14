import os
import os.path as osp
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from data.data_utils import (
    extract_closure_events,
    get_event_time_indices,
    filter_samples_by_event_mask,
    generate_time_features_from_indices,
)


def build_u_method_a(
    R: torch.Tensor,
    T: int,
    P: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Build residual-history tensor U using the old Method A logic.

    Args:
        R: residual targets of shape (S, P, N)
        T: input length
        P: forecast horizon
        num_nodes: number of nodes

    Returns:
        U: tensor of shape (S, P, N, T)
    """
    num_samples = R.shape[0]
    pad_len = T + P

    R_padded = torch.cat(
        [torch.zeros(pad_len, P, num_nodes, dtype=R.dtype), R],
        dim=0
    )  # (pad_len + S, P, N)

    U_list = []
    for i in range(num_samples):
        current_sample_idx_padded = i + pad_len
        sequence_u = []

        for t in range(T):
            global_t_idx = current_sample_idx_padded - (T - 1) + t
            step_u = []

            for k in range(P):
                val = R_padded[global_t_idx - (k + 1), k, :]   # (N,)
                step_u.append(val)

            seq_step = torch.stack(step_u, dim=0)              # (P,N)
            sequence_u.append(seq_step)

        U_sample = torch.stack(sequence_u, dim=0)              # (T,P,N)
        U_sample = U_sample.permute(1, 2, 0).contiguous()      # (P,N,T)
        U_list.append(U_sample)

    U = torch.stack(U_list, dim=0)                             # (S,P,N,T)
    return U


@torch.no_grad()
def build_residual_dataset_fullgraph(
    base_model,
    source_loader,
    speed_df: pd.DataFrame,
    weekly_avg_df: pd.DataFrame,
    closure_df: pd.DataFrame,
    sample_start_indices: Sequence[int],
    args,
    cache_path: Optional[str] = None,
    use_existing_closure_channel: bool = True,
):
    """
    Build full-graph ResCAL dataset using closure-related samples only.

    Baseline:
        uses source_loader as-is

    ResCAL input X:
        [speed_dev, closure_context, tod, dow] if use_existing_closure_channel=True
        [speed_dev, tod, dow]                  otherwise

    Output:
        TensorDataset(X, U, R, BASE, Y)

    Shapes:
        X    : (S, Cx, N, T)
        U    : (S, P,  N, T)
        R    : (S, P,  N)
        BASE : (S, P,  N)
        Y    : (S, P,  N)
    """

    if cache_path is not None and osp.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        required = ["X", "U", "R", "BASE", "Y", "kept_sample_start_indices"]
        if all(k in pack for k in required):
            return TensorDataset(
                pack["X"], pack["U"], pack["R"], pack["BASE"], pack["Y"]
            )
        print(f"[ResCAL] Cache exists but missing keys, rebuilding: {cache_path}")

    assert len(speed_df) == len(weekly_avg_df) == len(closure_df), (
        "speed_df, weekly_avg_df, closure_df must have the same number of rows"
    )
    assert speed_df.shape[1] == closure_df.shape[1] == args.num_nodes, (
        "Node count mismatch between dataframes and args.num_nodes"
    )
    assert len(sample_start_indices) == len(source_loader.dataset), (
        "sample_start_indices must align with source_loader.dataset"
    )

    total_time = len(speed_df)
    num_nodes = args.num_nodes
    T = args.T
    P = args.P

    # ---------------------------------------------------------
    # 1. Run baseline over all samples from source_loader
    # ---------------------------------------------------------
    base_model.eval()

    X_raw_list = []
    R_list = []
    BASE_list = []
    Y_list = []

    for x, y in source_loader:
        x = x.to(args.device)
        y = y.to(args.device)

        y_speed = y[:, :, :, 0]   # (B,P,N)
        base_out = base_model(x)  # (B,P,N)
        r = y_speed - base_out    # (B,P,N)

        X_raw_list.append(x.detach().cpu())      # (B,T,N,F)
        R_list.append(r.detach().cpu())
        BASE_list.append(base_out.detach().cpu())
        Y_list.append(y_speed.detach().cpu())

    X_raw_all = torch.cat(X_raw_list, dim=0).contiguous()   # (S,T,N,F)
    R_all = torch.cat(R_list, dim=0).contiguous()           # (S,P,N)
    BASE_all = torch.cat(BASE_list, dim=0).contiguous()     # (S,P,N)
    Y_all = torch.cat(Y_list, dim=0).contiguous()           # (S,P,N)

    # ---------------------------------------------------------
    # 2. Keep only closure-related samples
    # ---------------------------------------------------------
    events = extract_closure_events(closure_df)
    event_mask = get_event_time_indices(
        events=events,
        T=T,
        P=P,
        max_len=total_time,
    )

    kept_sample_start_indices = filter_samples_by_event_mask(
        sample_indices=np.asarray(sample_start_indices),
        event_mask=event_mask,
        T=T,
        P=P,
    )

    kept_sample_start_indices = np.asarray(kept_sample_start_indices, dtype=np.int64)

    if kept_sample_start_indices.size == 0:
        raise ValueError("No closure-related samples found for ResCAL dataset.")

    # Map kept dataframe indices back to dataset positions
    start_idx_to_pos = {
        int(start_idx): pos for pos, start_idx in enumerate(sample_start_indices)
    }
    keep_positions = [start_idx_to_pos[int(start_idx)] for start_idx in kept_sample_start_indices]

    X_raw_kept = X_raw_all[keep_positions]
    R = R_all[keep_positions]
    BASE = BASE_all[keep_positions]
    Y = Y_all[keep_positions]

    # ---------------------------------------------------------
    # 3. Rebuild ResCAL X with weekly-average removed speed
    # ---------------------------------------------------------
    speed_dev_df = speed_df - weekly_avg_df

    speed_dev_samples = []
    closure_samples = []

    for start_idx in kept_sample_start_indices:
        x_slice = slice(int(start_idx), int(start_idx) + T)

        speed_window = speed_dev_df.iloc[x_slice].to_numpy(dtype=np.float32)   # (T,N)
        speed_dev_samples.append(speed_window[..., None])                       # (T,N,1)

        if use_existing_closure_channel:
            # If your baseline closure/context channel is derived from closure_df,
            # this keeps the ResCAL input aligned with that source.
            closure_window = closure_df.iloc[x_slice].to_numpy(dtype=np.float32)  # (T,N)
            closure_samples.append(closure_window[..., None])                      # (T,N,1)

    speed_dev_tensor = torch.tensor(
        np.stack(speed_dev_samples, axis=0), dtype=torch.float32
    )  # (S,T,N,1)

    parts = [speed_dev_tensor]

    if use_existing_closure_channel:
        closure_tensor = torch.tensor(
            np.stack(closure_samples, axis=0), dtype=torch.float32
        )  # (S,T,N,1)
        parts.append(closure_tensor)

    time_feats = generate_time_features_from_indices(
        df_index=speed_df.index,
        sample_start_indices=kept_sample_start_indices,
        num_nodes=num_nodes,
        T=T,
    )  # (S,T,N,2)
    parts.append(time_feats)

    X = torch.cat(parts, dim=-1)           # (S,T,N,Cx)
    X = X.permute(0, 3, 2, 1).contiguous() # (S,Cx,N,T)

    # ---------------------------------------------------------
    # 4. Build U from filtered residual sequence
    # ---------------------------------------------------------
    U = build_u_method_a(
        R=R,
        T=T,
        P=P,
        num_nodes=num_nodes,
    )  # (S,P,N,T)

    # ---------------------------------------------------------
    # 5. Optional cache
    # ---------------------------------------------------------
    if cache_path is not None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "X": X,
                "U": U,
                "R": R,
                "BASE": BASE,
                "Y": Y,
                "kept_sample_start_indices": kept_sample_start_indices,
            },
            cache_path,
        )

    return TensorDataset(X, U, R, BASE, Y)