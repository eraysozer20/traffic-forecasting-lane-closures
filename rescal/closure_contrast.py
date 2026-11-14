# find_closure_roots.py
import os
import os.path as osp
import numpy as np
import pandas as pd

# Reuse your existing helper functions
from utils.cli import build_args
from utils.configs import apply_dataset_cfg
from utils.data_loading import loadData_hyp2


def _reconstruct_construction_from_split(ds, split_times, T, P):
    """
    Reconstruct per-timestamp construction matrix (time x nodes) from a split dataset.
    ds samples contain x:(T,N,3), y:(P,N,3) where channel-1 is construction.
    """
    if len(ds) == 0:
        return pd.DataFrame(index=pd.to_datetime(split_times))

    x0, _ = ds[0]
    num_nodes = int(x0.shape[1])

    # data_generate may truncate to full batches; only keep covered timeline.
    max_covered = len(ds) + T + P - 1
    L = min(len(split_times), max_covered)

    con = np.full((L, num_nodes), np.nan, dtype=np.float32)

    for i in range(len(ds)):
        x, y = ds[i]
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        x_con = x[:, :, 1]  # (T, N)
        y_con = y[:, :, 1]  # (P, N)

        for t in range(T):
            g = i + t
            if g < L and np.isnan(con[g, 0]):
                con[g, :] = x_con[t, :]

        for p in range(P):
            g = i + T + p
            if g < L and np.isnan(con[g, 0]):
                con[g, :] = y_con[p, :]

    valid = ~np.isnan(con).any(axis=1)
    con = con[valid]
    idx = pd.to_datetime(np.array(split_times[:L])[valid])

    return pd.DataFrame(con, index=idx)


def build_construction_df_from_hyp2(args):
    """
    Use loadData_hyp2 outputs (A/B/C datasets with speed, construction, time)
    and reconstruct one full sequential construction dataframe.
    """
    A_ds, B_ds, C_ds, _, A_adj, _, _, _, timestamp_list = loadData_hyp2(args)
    num_nodes = int(A_adj.shape[0])
    node_cols = list(range(num_nodes))

    df_A = _reconstruct_construction_from_split(A_ds, timestamp_list[0], args.T, args.P)
    df_B = _reconstruct_construction_from_split(B_ds, timestamp_list[1], args.T, args.P)
    df_C = _reconstruct_construction_from_split(C_ds, timestamp_list[2], args.T, args.P)

    for d in (df_A, df_B, df_C):
        d.columns = node_cols

    return pd.concat([df_A, df_B, df_C], axis=0)

def find_closure_roots(df_con, min_duration=1):
    """
    Scans the dataframe for consecutive sequences of 1.0 (Roots).
    
    Args:
        df_con: DataFrame (Time x Nodes) or (Nodes x Time). 
                Assuming (Time x Nodes) based on standard traffic datasets.
        min_duration: Filter out events shorter than these many timesteps.
    
    Returns:
        pd.DataFrame: A catalog of all closure roots.
    """
    events = []
    
    # Ensure we are working with (Time, Nodes)
    # If the CSV is (Nodes, Time), we transpose. 
    # Usually PeMS data is (Time, Nodes). 
    # We check if shape[0] > shape[1] to guess, or rely on index.
    # Assuming standard (Time, Nodes) for now.
    
    num_timesteps, num_nodes = df_con.shape
    nodes = df_con.columns
    times = df_con.index
    
    print(f"Scanning {num_nodes} sensors over {num_timesteps} timesteps for closure roots (val == 1)...")

    for col_idx, node_id in enumerate(nodes):
        # Get the series for this sensor
        series = df_con.iloc[:, col_idx].values
        
        # Boolean mask for roots
        # We look for exactly 1 (or close to 1 if float precision is an issue)
        is_root = np.isclose(series, 1.0)
        
        if not np.any(is_root):
            continue
            
        # Find consecutive runs
        # Pad with False to handle edge cases at start/end of array
        padded = np.concatenate(([False], is_root, [False]))
        diffs = np.diff(padded.astype(int))
        
        # Starts are 1 (False -> True), Ends are -1 (True -> False)
        start_indices = np.where(diffs == 1)[0]
        end_indices = np.where(diffs == -1)[0]
        
        for start_step, end_step in zip(start_indices, end_indices):
            duration = end_step - start_step
            
            if duration >= min_duration:
                events.append({
                    'node_id': node_id,         # The sensor name/ID
                    'node_idx': col_idx,        # The integer index (0 to N)
                    'start_idx': start_step,    # Integer start time
                    'end_idx': end_step,        # Integer end time
                    'start_time': times[start_step], # Actual timestamp
                    'end_time': times[end_step - 1], # Actual timestamp
                    'duration': duration
                })

    df_events = pd.DataFrame(events)
    
    # Sort by start time
    if not df_events.empty:
        df_events = df_events.sort_values(by='start_idx').reset_index(drop=True)
        
    return df_events

def main():
    # 1. Setup Arguments & Config using your helpers
    args = build_args()
    apply_dataset_cfg(args)
    
    print(f"[INFO] Dataset: {args.dataset_choice}")
    print("[INFO] Building closure signal from loadData_hyp2() splits A/B/C")

    # 2. Build construction dataframe from A/B/C datasets (channel-1)
    df_con = build_construction_df_from_hyp2(args)
    
    # 3. Find Roots
    roots_df = find_closure_roots(df_con, min_duration=12) # e.g. min 12 steps (1 hour if 5min steps)
    
    # 4. Display Results
    print(f"\n[RESULT] Found {len(roots_df)} closure root events.")
    
    if not roots_df.empty:
        print("\n--- First 10 Events ---")
        print(roots_df[['node_id', 'start_time', 'duration']].head(10))
        
        print("\n--- Events by Node ---")
        print(roots_df['node_id'].value_counts().head())
        
        # Optional: Save to CSV for the next step
        save_name = f"closure_roots_{args.dataset_choice}.csv"
        roots_df.to_csv(save_name, index=False)
        print(f"\n[INFO] Saved catalog to {save_name}")
    else:
        print("[WARN] No values == 1.0 found. Check if data is already normalized or uses different flag values.")

if __name__ == "__main__":
    main()