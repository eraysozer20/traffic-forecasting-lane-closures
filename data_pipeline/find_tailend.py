#!/usr/bin/env python3
"""
Dense scan for a tail-start date T_start from lane-closure matches.

We still use the same definition of "tail-only sensor", but instead of only
checking Jan 1 of each year, we scan T_start on a regular grid (e.g. every day
or every N days) over [T_min, T_max).

This will tell us:
  - What is the maximum number of tail-only sensors we can get for any T_start?
  - For which T_start(s) does that happen?

Then you can pick an N that is actually achievable and (if you want) choose
the earliest T_start that meets that N.
"""

from pathlib import Path
from collections import defaultdict
from datetime import timedelta
from bisect import bisect_left

import pandas as pd


# =======================================================
# SCRIPT CONFIG
# =======================================================

DISTRICT = 8           # PeMS district (int)
TAIL_PERCENT = 30.0    # last % of [T_start, T_max] that counts as tail
STEP_DAYS = 1          # step in days when scanning T_start (1 = daily)
MIN_WINDOW_DAYS = 365  # require [T_start, T_max] to be at least this long

ROOT = Path(__file__).resolve().parent / "data"


# =======================================================
# HELPERS (now also return df)
# =======================================================

def load_closure_times(matches_csv: Path):
    if not matches_csv.exists():
        raise FileNotFoundError(f"Matches CSV not found: {matches_csv}")

    print(f"[LOAD] lane closure matches: {matches_csv}")
    df = pd.read_csv(matches_csv, low_memory=False)

    if "Status" not in df.columns:
        raise KeyError("Expected 'Status' column in lane closure CSV")
    df = df[df["Status"] == "Approved"].copy()
    print(f"  → after Status == 'Approved': {len(df)} rows")

    if "Start Date" not in df.columns:
        raise KeyError("Expected 'Start Date' column in lane closure CSV")
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

    # optional: also parse End Date if present, for nice printing
    if "End Date" in df.columns:
        df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")

    df = df.dropna(subset=["Start Date"])
    print(f"  → after dropping NaT Start Date: {len(df)} rows")

    if "matched_station_ids" not in df.columns:
        raise KeyError("Expected 'matched_station_ids' column in lane closure CSV")

    closure_times = defaultdict(list)

    for _, row in df.iterrows():
        start = row["Start Date"]
        station_str = row["matched_station_ids"]
        if pd.isna(station_str):
            continue
        stations = [s.strip() for s in str(station_str).split(";") if s.strip()]
        for s in stations:
            closure_times[s].append(start)

    if not closure_times:
        raise RuntimeError("No closure events found after filtering; check your CSV.")

    for s in closure_times:
        closure_times[s].sort()

    all_times = [t for times in closure_times.values() for t in times]
    T_min = min(all_times)
    T_max = max(all_times)

    print(f"[INFO] global closure time range: {T_min} → {T_max}")
    print(f"[INFO] sensors with any closure: {len(closure_times)}")

    # return closure_times + filtered df
    return closure_times, T_min, T_max, df


def evaluate_T_start(
    closure_times: dict,
    T_start: pd.Timestamp,
    T_max: pd.Timestamp,
    tail_percent: float,
    return_sensors: bool = False,
):
    if tail_percent <= 0 or tail_percent >= 100:
        raise ValueError("tail_percent must be between 0 and 100 (exclusive).")

    tail_fraction = tail_percent / 100.0

    T_start = pd.Timestamp(T_start)
    T_max = pd.Timestamp(T_max)
    total_span = (T_max - T_start).total_seconds()
    if total_span <= 0:
        return (0, []) if return_sensors else 0

    head_fraction = 1.0 - tail_fraction
    head_seconds = total_span * head_fraction
    T_tail_start = T_start + pd.to_timedelta(head_seconds, unit="s")

    count = 0
    tail_sensors = []

    for s, times in closure_times.items():
        if not times:
            continue

        i_start = bisect_left(times, T_start)
        if i_start >= len(times):
            continue

        i_tail = bisect_left(times, T_tail_start)

        # tail-only if:
        #  - at least one event in tail → i_tail < len(times)
        #  - no event in head → i_tail == i_start
        if i_tail < len(times) and i_tail == i_start:
            count += 1
            if return_sensors:
                tail_sensors.append(s)

    if return_sensors:
        return count, tail_sensors
    else:
        return count


# =======================================================
# DENSE SCAN
# =======================================================

def dense_scan_tail_start(
    closure_times: dict,
    T_min: pd.Timestamp,
    T_max: pd.Timestamp,
    tail_percent: float,
    step_days: int,
    min_window_days: int,
):
    """
    Scan T_start from T_min to T_max with a fixed day step, and record:
      - the maximum number of tail-only sensors
      - the T_start that achieves this maximum (and a few examples)

    Returns:
      best_T_start, best_count, best_sensors
    """
    T_min = pd.Timestamp(T_min)
    T_max = pd.Timestamp(T_max)

    step = timedelta(days=step_days)
    min_window = timedelta(days=min_window_days)

    cur = T_min
    best_T_start = None
    best_count = -1
    best_sensors = []

    print(f"\n[SCAN] Dense scan with step={step_days} days, "
          f"tail_percent={tail_percent}, min_window={min_window_days} days\n")

    n_steps = 0
    while cur < T_max:
        if (T_max - cur) < min_window:
            break

        count = evaluate_T_start(
            closure_times, cur, T_max, tail_percent, return_sensors=False
        )

        if count > best_count:
            best_count = count
            best_T_start = cur
            # re-evaluate once to get the sensors list
            _, sensors = evaluate_T_start(
                closure_times, cur, T_max, tail_percent, return_sensors=True
            )
            best_sensors = sensors
            print(f"  New best: T_start={cur}, tail-only sensors={best_count}")

        if n_steps % 365 == 0:
            print(f"  ... checked {n_steps} steps; current T_start={cur}")

        cur += step
        n_steps += 1

    print("\n[SCAN RESULT]")
    if best_T_start is None:
        print("  No valid T_start found (this should not happen if closure_times is non-empty).")
        return None, 0, []

    print(f"  Best T_start: {best_T_start}")
    print(f"  Best tail-only sensors count: {best_count}")
    print(f"  Example tail-only sensors (first 20): {best_sensors[:20]}")
    return best_T_start, best_count, best_sensors


# =======================================================
# MAIN
# =======================================================

def main():
    district_code = f"{DISTRICT:02d}"
    matches_csv = (
        ROOT
        / "lane_closures"
        / f"pems{district_code}"
        / f"pems{district_code}_lane_closure_matches.csv"
    )

    print(f"=== Dense tail-start scan for PeMS district {district_code} ===")
    print(f"Matches CSV:     {matches_csv}")
    print(f"TAIL_PERCENT   = {TAIL_PERCENT}")
    print(f"STEP_DAYS      = {STEP_DAYS}")
    print(f"MIN_WINDOW_DAYS= {MIN_WINDOW_DAYS}")

    closure_times, T_min, T_max, df = load_closure_times(matches_csv)

    best_T_start, best_count, best_sensors = dense_scan_tail_start(
        closure_times=closure_times,
        T_min=T_min,
        T_max=T_max,
        tail_percent=TAIL_PERCENT,
        step_days=STEP_DAYS,
        min_window_days=MIN_WINDOW_DAYS,
    )

    print("\n[SUMMARY]")
    print(f"  T_min (global):   {T_min}")
    print(f"  T_max (global):   {T_max}")
    print(f"  Best T_start:     {best_T_start}")
    print(f"  Max tail-only sensors: {best_count}")

    # ====================================================
    # NEW: Extract ONLY the true tail-only closure events
    # and print ID + Start/End dates
    # ====================================================
    if best_T_start is not None and best_count > 0:
        tail_fraction = TAIL_PERCENT / 100.0
        head_fraction = 1.0 - tail_fraction
        total_span = (T_max - best_T_start).total_seconds()
        head_seconds = total_span * head_fraction
        T_tail_start = best_T_start + pd.to_timedelta(head_seconds, unit="s")

        print(f"\n[TAIL WINDOW]")
        print(f"  Tail start for best T_start: {T_tail_start}")
        print(f"  Tail end (T_max):            {T_max}")

        tail_only_set = set(best_sensors)

        # Rows inside tail window
        tail_df = df[df["Start Date"] >= T_tail_start].copy()

        def contains_tail_only(st_str):
            if pd.isna(st_str):
                return False
            stations = [x.strip() for x in str(st_str).split(";") if x.strip()]
            return any(s in tail_only_set for s in stations)

        tail_df = tail_df[tail_df["matched_station_ids"].apply(contains_tail_only)]

        print(f"  Tail-only closure events (rows): {len(tail_df)}")

        if "ID" in tail_df.columns:
            # Deduplicate by (ID, Start Date, End Date) and print
            cols = ["ID", "Start Date"]
            if "End Date" in tail_df.columns:
                cols.append("End Date")

            uniq_rows = (
                tail_df[cols]
                .drop_duplicates()
                .sort_values(by=cols)
            )

            print("\n[STRICT TAIL-ONLY CLOSURE EVENTS]")
            print(f"  Unique (ID, Start, End) rows: {len(uniq_rows)}")
            for _, r in uniq_rows.iterrows():
                cid = r["ID"]
                s = r["Start Date"]
                e = r.get("End Date", None)
                if pd.isna(e):
                    print(f"    ID={cid}, start={s}")
                else:
                    print(f"    ID={cid}, start={s}, end={e}")
        else:
            print("[WARN] 'ID' column not found; cannot list closure IDs explicitly.")


if __name__ == "__main__":
    main()
