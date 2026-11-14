#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. CONFIGURATION
# ==========================================

DISTRICT = "03"   # default

_env_d = os.getenv("PEMS_DISTRICT_ID")
if _env_d:
    DISTRICT = f"{int(_env_d):02d}"

file_path = r'data\despt_inputs\pems' + DISTRICT + r'\closure_labels.h5'
key_name = 'closure_labels'

# START SEARCH FROM 2004 (Skipping the sparse early years)
SEARCH_START_DATE = "2004-01-01"

WINDOW_SIZE = pd.DateOffset(years=0, months=6)
STRIDE = pd.DateOffset(days=7)
RECENCY_BIAS = 3.0

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# Plot popup window control
SHOW_PLOT = False   # default off


# 2. HELPER FUNCTIONS
# ==========================================

def get_active_nodes(sub_df):
    if sub_df.empty:
        return set()
    has_closure = (sub_df > 0.5).any(axis=0)
    return set(has_closure[has_closure].index)


def load_closure_df(file_path, key_name):
    print("Loading data...")
    try:
        df = pd.read_hdf(file_path, key=key_name)
    except KeyError:
        with pd.HDFStore(file_path, mode='r') as store:
            df = store.get(store.keys()[0])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def plot_winner_window(df, winner, window_size, train_ratio, val_ratio):
    win_start = winner['Window_Start']
    win_end = win_start + window_size

    print(f"\nVisualizing Winner: {win_start.date()}")

    w_df = df[win_start:win_end]
    split_1_idx = int(len(w_df) * train_ratio)
    split_2_idx = int(len(w_df) * (train_ratio + val_ratio))

    if len(w_df) == 0 or split_1_idx >= len(w_df) or split_2_idx >= len(w_df):
        print("[WARN] Winner window is too small to plot properly.")
        return

    date_train_end = w_df.index[split_1_idx]
    date_val_end = w_df.index[split_2_idx]

    train_df = w_df.iloc[:split_1_idx]
    train_nodes = get_active_nodes(train_df)

    intervals = []
    active_cols = w_df.columns[(w_df > 0.5).any()]

    for node in active_cols:
        is_unseen = node not in train_nodes
        series = w_df[node]
        binary = (series > 0.5).astype(int)
        diffs = binary.diff()

        starts = diffs[diffs == 1].index.tolist()
        ends = diffs[diffs == -1].index.tolist()

        if len(binary) > 0 and binary.iloc[0] == 1:
            starts.insert(0, w_df.index[0])
        if len(binary) > 0 and binary.iloc[-1] == 1:
            ends.append(w_df.index[-1])

        for s, e in zip(starts, ends):
            intervals.append({
                'Node': str(node),
                'Start': s,
                'End': e,
                'Type': 'Unseen' if is_unseen else 'Seen'
            })

    if not intervals:
        print("[WARN] No intervals found to plot.")
        return

    plot_df = pd.DataFrame(intervals)

    fig, ax = plt.subplots(figsize=(14, 10))
    unique_nodes = sorted(plot_df['Node'].unique())
    node_map = {node: i for i, node in enumerate(unique_nodes)}

    seen_df = plot_df[plot_df['Type'] == 'Seen']
    if not seen_df.empty:
        ax.hlines(
            y=seen_df['Node'].map(node_map),
            xmin=seen_df['Start'],
            xmax=seen_df['End'],
            linewidth=3,
            color='#95a5a6',
            alpha=0.5,
            label='Seen in Train'
        )

    unseen_df = plot_df[plot_df['Type'] == 'Unseen']
    if not unseen_df.empty:
        ax.hlines(
            y=unseen_df['Node'].map(node_map),
            xmin=unseen_df['Start'],
            xmax=unseen_df['End'],
            linewidth=4,
            color='#c0392b',
            alpha=1.0,
            label='UNSEEN (New)'
        )

    ax.axvline(x=date_train_end, color='black', linestyle='--')
    ax.axvline(x=date_val_end, color='black', linestyle='--')

    ax.axvspan(win_start, date_train_end, color='green', alpha=0.05)
    ax.text(
        win_start + pd.Timedelta(days=10),
        len(unique_nodes),
        f" TRAIN (n={int(winner['Train_Count'])})",
        color='green',
        fontweight='bold',
        fontsize=12
    )

    ax.axvspan(date_train_end, date_val_end, color='orange', alpha=0.1)
    ax.text(
        date_train_end + pd.Timedelta(days=10),
        len(unique_nodes),
        f" VAL (Unseen={int(winner['Val_Unseen'])})",
        color='#d35400',
        fontweight='bold',
        fontsize=12
    )

    ax.axvspan(date_val_end, win_end, color='blue', alpha=0.1)
    ax.text(
        date_val_end + pd.Timedelta(days=10),
        len(unique_nodes),
        f" TEST (Unseen={int(winner['Test_Unseen'])})",
        color='blue',
        fontweight='bold',
        fontsize=12
    )

    ax.set_title(f"Selected Window: {win_start.date()}", fontsize=16)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def get_best_date(show_plot=False):
    # 3. LOAD DATA
    # ==========================================
    df = load_closure_df(file_path, key_name)

    # 4. SLIDING WINDOW ANALYSIS
    # ==========================================
    results = []
    dataset_start = df.index.min()
    dataset_end = df.index.max()
    total_duration = (dataset_end - dataset_start).days

    # enforce the 2004 start constraint
    search_start_dt = pd.Timestamp(SEARCH_START_DATE)
    current_window_start = max(dataset_start, search_start_dt)

    print(f"\nDistrict: {DISTRICT}")
    print(f"Input file: {file_path}")
    print(f"Scanning windows starting from {current_window_start.date()}...")
    print("Constraints: Train>=10 nodes | Early Activity check | Val/Test Unseen > 0")

    iteration_count = 0

    while current_window_start + WINDOW_SIZE <= dataset_end:
        current_window_end = current_window_start + WINDOW_SIZE
        window_df = df[current_window_start:current_window_end]

        if not window_df.empty:
            n_rows = len(window_df)
            split_1 = int(n_rows * TRAIN_RATIO)
            split_2 = int(n_rows * (TRAIN_RATIO + VAL_RATIO))

            train_df = window_df.iloc[:split_1]
            val_df = window_df.iloc[split_1:split_2]
            test_df = window_df.iloc[split_2:]

            # Early activity check: first half of training must not be empty
            half_train_idx = int(len(train_df) / 2)
            early_train_activity = (train_df.iloc[:half_train_idx] > 0.5).sum().sum()

            if early_train_activity > 0:
                train_nodes = get_active_nodes(train_df)
                val_nodes = get_active_nodes(val_df)
                test_nodes = get_active_nodes(test_df)

                train_count = len(train_nodes)
                val_unseen_count = len(val_nodes - train_nodes)
                test_unseen_count = len(test_nodes - train_nodes)

                if (train_count >= 10) and (val_unseen_count > 0) and (test_unseen_count > 0):
                    progress = (current_window_start - dataset_start).days / total_duration
                    recency_factor = 1.0 + (progress * RECENCY_BIAS)

                    results.append({
                        'Window_Start': current_window_start,
                        'Train_Count': train_count,
                        'Val_Unseen': val_unseen_count,
                        'Test_Unseen': test_unseen_count,
                        'Recency_Factor': round(recency_factor, 2)
                    })

        current_window_start += STRIDE
        iteration_count += 1

        if iteration_count % 50 == 0:
            print(f"Processed: {current_window_start.date()}", end='\r')

    res_df = pd.DataFrame(results)

    # 5. RANKING & SELECTION
    # ==========================================
    if res_df.empty:
        print("\nNo windows found. Check if your data has enough unseen nodes post-2004.")
        return None

    res_df['Score'] = (res_df['Val_Unseen'] + res_df['Test_Unseen']) * res_df['Recency_Factor']
    best_windows = res_df.sort_values(by='Score', ascending=False)

    print("\n\n--- TOP 5 CANDIDATES (Post-2004 Start) ---")
    print(
        best_windows[
            ['Window_Start', 'Train_Count', 'Val_Unseen', 'Test_Unseen', 'Recency_Factor', 'Score']
        ].head(5)
    )

    winner = best_windows.iloc[0]
    win_start = winner['Window_Start']
    best_date_str = win_start.strftime("%Y-%m-%d")

    print(f"\nBest start date: {best_date_str}")

    if show_plot:
        plot_winner_window(df, winner, WINDOW_SIZE, TRAIN_RATIO, VAL_RATIO)

    return best_date_str


# 6. MAIN
# ==========================================
if __name__ == "__main__":
    best_date = get_best_date(show_plot=SHOW_PLOT)
    print(f"\nReturned best date string: {best_date}")