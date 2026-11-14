#!/usr/bin/env python3
"""
Compare speed (flow) during *specific* closure events vs. baseline in nearby weeks.

You must specify the exact closure events (ID + start + end) in CLOSURE_EVENTS below.
For each such event and each sensor (all columns in the speed file):

  - collect speeds during [Start Date, End Date]
  - collect baseline speeds at the same (day_of_week, time_of_day)
    within a +/- HALF_BASELINE_WINDOW_DAYS window around the closure start,
    excluding times when that sensor is under any of these specified closures.
  - compute mean speeds and relative change.

Outputs a CSV with per-(closure_event, sensor_id) statistics.
"""

from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


# ======================================================
# CONFIG
# ======================================================

ROOT = Path(__file__).resolve().parent

# Wide speed matrix: timestamp + one column per station ID
SPEED_CSV = ROOT / "data" / "pems" / "pems08" / "pems08_all_common_flow.csv"

# Lane closure matches CSV
CLOSURE_CSV = ROOT / "data" / "lane_closures" / "pems08" / "pems08_lane_closure_matches.csv"

# Output result
OUT_CSV = ROOT / "data" / "analysis" / "pems08_closure_speed_comparison_selected.csv"

# Column names
TIME_COL = "date"             # in SPEED_CSV
CLOSURE_START_COL = "Start Date"
CLOSURE_END_COL = "End Date"
CLOSURE_ID_COL = "ID"
CLOSURE_MATCHED_COL = "matched_station_ids"
CLOSURE_STATUS_COL = "Status"
CLOSURE_DISTRICT_COL = "District"

# >>> FILL THIS WITH THE EVENTS YOU WANT (copy from tail-end script output)
CLOSURE_EVENTS = [
    # Example (replace with your real lines):
    {"id": "C10BD", "start": "2024-02-09 22:01:00", "end": "2024-02-12 05:01:00"},
    {"id": "T10BD", "start": "2023-11-06 00:01:00", "end": "2023-11-06 02:59:00"},
    # {"id": "T10BD", "start": "YYYY-MM-DD HH:MM:SS", "end": "YYYY-MM-DD HH:MM:SS"},
]

# Limit baseline to ~1 year window around closure start:
HALF_BASELINE_WINDOW_DAYS = 183

# Minimum number of data points required
MIN_POINTS_CLOSURE = 3
MIN_POINTS_BASELINE = 20


# ======================================================
# LOAD DATA
# ======================================================

def load_speed_data(path: Path, time_col: str) -> pd.DataFrame:
    print(f"[LOAD] speed data: {path}")
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise KeyError(f"Expected time column '{time_col}' in speed data")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    idx = df.index
    df["dow"] = idx.dayofweek
    df["tod_min"] = idx.hour * 60 + idx.minute

    print(f"  rows: {len(df)}, time range: {idx.min()} → {idx.max()}")
    return df


def load_selected_closures(path: Path) -> pd.DataFrame:
    print(f"[LOAD] closures: {path}")
    df = pd.read_csv(path, low_memory=False)

    if not CLOSURE_EVENTS:
        raise ValueError("CLOSURE_EVENTS is empty. Please fill it with (id, start, end).")

    # Filter to district 8
    if CLOSURE_DISTRICT_COL in df.columns:
        df = df[df[CLOSURE_DISTRICT_COL] == 8].copy()

    # Parse dates
    for col in [CLOSURE_START_COL, CLOSURE_END_COL]:
        if col not in df.columns:
            raise KeyError(f"Expected '{col}' column in closure CSV")
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=[CLOSURE_START_COL, CLOSURE_END_COL])

    # Only approved closures
    if CLOSURE_STATUS_COL in df.columns:
        df = df[df[CLOSURE_STATUS_COL] == "Approved"].copy()

    if CLOSURE_ID_COL not in df.columns:
        raise KeyError(f"Expected '{CLOSURE_ID_COL}' column in closure CSV")
    if CLOSURE_MATCHED_COL not in df.columns:
        raise KeyError(f"Expected '{CLOSURE_MATCHED_COL}' column in closure CSV")

    # Build mask by matching ID + start (and optionally end)
    mask = pd.Series(False, index=df.index)

    events_resolved = []
    for ev in CLOSURE_EVENTS:
        cid = ev["id"]
        s_raw = ev["start"]
        e_raw = ev.get("end", None)

        start_ts = pd.to_datetime(s_raw)
        if e_raw is not None:
            end_ts = pd.to_datetime(e_raw)
        else:
            end_ts = None

        # Match on ID + exact Start Date; if end is given, also match End Date
        m = (df[CLOSURE_ID_COL] == cid) & (df[CLOSURE_START_COL] == start_ts)
        if end_ts is not None:
            m = m & (df[CLOSURE_END_COL] == end_ts)

        matched_rows = df[m]
        if matched_rows.empty:
            print(f"[WARN] No rows found for ID={cid}, start={start_ts}, end={end_ts}")
        else:
            print(f"[INFO] Found {len(matched_rows)} row(s) for ID={cid}, start={start_ts}, end={end_ts}")
            mask |= m
            # Keep track for reference
            for _, r in matched_rows.iterrows():
                events_resolved.append(
                    (r[CLOSURE_ID_COL], r[CLOSURE_START_COL], r[CLOSURE_END_COL])
                )

    df_sel = df[mask].copy()
    print(f"  Selected closure rows: {len(df_sel)}")
    if not events_resolved:
        print("[ERROR] No closure rows matched CLOSURE_EVENTS.")
    else:
        print("[SELECTED EVENTS]")
        for cid, s, e in sorted(events_resolved, key=lambda x: (x[0], x[1])):
            print(f"  ID={cid}, start={s}, end={e}")

    return df_sel


# ======================================================
# CLOSURE MASK
# ======================================================

def build_closure_mask(speed_df: pd.DataFrame, closures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a boolean DataFrame `is_closed` with same index as speed_df and
    columns = sensor columns, True where sensor is under any of the selected closures.
    """
    helper_cols = {"dow", "tod_min"}
    sensor_cols = [c for c in speed_df.columns if c not in helper_cols]

    print(f"[MASK] building closure mask for {len(sensor_cols)} sensors "
          f"and {len(closures_df)} closure rows")

    # fixed here
    is_closed = pd.DataFrame(False, index=speed_df.index, columns=sensor_cols)

    t_min, t_max = speed_df.index.min(), speed_df.index.max()

    for _, row in closures_df.iterrows():
        start = row[CLOSURE_START_COL]
        end = row[CLOSURE_END_COL]
        if pd.isna(start) or pd.isna(end):
            continue

        # Clip to speed data range
        if end < t_min or start > t_max:
            continue
        start_clip = max(start, t_min)
        end_clip = min(end, t_max)

        station_str = row[CLOSURE_MATCHED_COL]
        if pd.isna(station_str):
            continue
        stations = [s.strip() for s in str(station_str).split(";") if s.strip()]

        mask_time = (speed_df.index >= start_clip) & (speed_df.index <= end_clip)
        if not mask_time.any():
            continue

        for s in stations:
            if s in is_closed.columns:
                is_closed.loc[mask_time, s] = True

    return is_closed



# ======================================================
# MAIN ANALYSIS
# ======================================================

def compute_closure_vs_baseline(
    speed_df: pd.DataFrame,
    closures_df: pd.DataFrame,
    is_closed: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    sensor_cols = [c for c in speed_df.columns if c not in {"dow", "tod_min"}]

    for _, row in closures_df.iterrows():
        closure_id = row[CLOSURE_ID_COL]
        start = row[CLOSURE_START_COL]
        end = row[CLOSURE_END_COL]

        print(f"[CLOSURE] ID={closure_id}, start={start}, end={end}")

        mask_time_closure = (speed_df.index >= start) & (speed_df.index <= end)
        if not mask_time_closure.any():
            print("  -> no speed data in this window, skipping")
            continue

        sub_keys = speed_df.loc[mask_time_closure, ["dow", "tod_min"]].drop_duplicates()
        closure_keys = list(zip(sub_keys["dow"], sub_keys["tod_min"]))
        if not closure_keys:
            print("  -> no (dow, tod_min) keys found, skipping")
            continue

        baseline_start = start - timedelta(days=HALF_BASELINE_WINDOW_DAYS)
        baseline_end = start + timedelta(days=HALF_BASELINE_WINDOW_DAYS)

        for s in sensor_cols:
            closure_speeds = speed_df.loc[mask_time_closure, s].dropna()
            if len(closure_speeds) < MIN_POINTS_CLOSURE:
                continue

            baseline_mask = pd.Series(False, index=speed_df.index)

            for dow, tod in closure_keys:
                m = (
                    (speed_df["dow"] == dow)
                    & (speed_df["tod_min"] == tod)
                    & (speed_df.index >= baseline_start)
                    & (speed_df.index <= baseline_end)
                )
                baseline_mask |= m

            baseline_mask &= (~is_closed[s])

            baseline_speeds = speed_df.loc[baseline_mask, s].dropna()
            if len(baseline_speeds) < MIN_POINTS_BASELINE:
                continue

            mean_closure = closure_speeds.mean()
            mean_baseline = baseline_speeds.mean()

            if mean_baseline != 0:
                rel_change = (mean_closure - mean_baseline) / mean_baseline
            else:
                rel_change = np.nan

            results.append(
                {
                    "closure_id": closure_id,
                    "sensor_id": s,
                    "start": start,
                    "end": end,
                    "n_points_closure": len(closure_speeds),
                    "n_points_baseline": len(baseline_speeds),
                    "mean_speed_closure": mean_closure,
                    "mean_speed_baseline": mean_baseline,
                    "rel_change": rel_change,
                }
            )

    return pd.DataFrame(results)


# ======================================================
# ENTRY POINT
# ======================================================

def main():
    speed_df = load_speed_data(SPEED_CSV, TIME_COL)
    closures_df = load_selected_closures(CLOSURE_CSV)

    if closures_df.empty:
        print("[ERROR] No selected closures after filtering. Check CLOSURE_EVENTS.")
        return

    is_closed = build_closure_mask(speed_df, closures_df)

    print("[ANALYSIS] computing closure vs. baseline stats ...")
    res_df = compute_closure_vs_baseline(speed_df, closures_df, is_closed)

    print(f"[RESULT] {len(res_df)} (closure_id, sensor_id) rows")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"[SAVE] {OUT_CSV}")


if __name__ == "__main__":
    main()
