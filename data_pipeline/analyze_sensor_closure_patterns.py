#!/usr/bin/env python3
"""
Analyze sensor closure patterns:

1) For each station_id (sensor), compute:
   - number of closures
   - first and last closure date
   - maximum gap (days) between consecutive closures

2) Find sensors that:
   - have NO closures in the first 70% of the global time range
   - have AT LEAST one closure in the last 30% of the time range

3) Write a global per-district summary CSV:
   data/lane_closures/pems_all_districts_sensor_closure_summary.csv

Input:
  data/lane_closures/pemsXX/pemsXX_closure_speed_impacts.csv

Outputs:
  data/lane_closures/pemsXX/pemsXX_sensor_closure_gaps.csv
  data/lane_closures/pemsXX/pemsXX_sensors_closures_only_late.csv
  data/lane_closures/pems_all_districts_sensor_closure_summary.csv

Plus: prints a human-readable summary at the end.
"""

import os
from pathlib import Path

import pandas as pd

# -------------------------------------------------------
# CONFIG – District via env var PEMS_DISTRICT_ID
# -------------------------------------------------------
DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d: int) -> str:
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DISTRICT_STR = district_code(DISTRICT)

ROOT_DIR = Path(__file__).resolve().parent
LC_DIR   = ROOT_DIR / "data" / "lane_closures" / f"pems{DISTRICT_STR}"

IN_FILE  = LC_DIR / f"pems{DISTRICT_STR}_closure_speed_impacts.csv"

OUT_GAPS = LC_DIR / f"pems{DISTRICT_STR}_sensor_closure_gaps.csv"
OUT_LATE = LC_DIR / f"pems{DISTRICT_STR}_sensors_closures_only_late.csv"

# Global summary (one row per district)
GLOBAL_SUMMARY = ROOT_DIR / "data" / "lane_closures" / "pems_all_districts_sensor_closure_summary.csv"


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

def load_closure_station_data() -> pd.DataFrame:
    """
    Load closure × station table and keep all columns,
    but ensure station_id, Start Date, End Date are usable.
    """
    print(f"[LOAD] {IN_FILE}")
    df = pd.read_csv(IN_FILE, low_memory=False)

    if "station_id" not in df.columns:
        raise KeyError("Expected 'station_id' column in closure speed impacts file.")

    if "Start Date" not in df.columns:
        raise KeyError("Expected 'Start Date' column in closure speed impacts file.")

    if "End Date" not in df.columns:
        raise KeyError("Expected 'End Date' column in closure speed impacts file.")

    # Parse dates (NEW: also End Date)
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"]   = pd.to_datetime(df["End Date"], errors="coerce")

    # Drop rows missing Start Date or station_id; keep End Date even if NaT
    before = len(df)
    df = df.dropna(subset=["Start Date", "station_id"])
    after = len(df)
    print(f"[FILTER] drop rows with NaT Start Date or null station_id: {before} → {after}")

    # Ensure station_id is treated as string (consistent with other scripts)
    df["station_id"] = df["station_id"].astype(str)

    return df


# -------------------------------------------------------
# 1) PER-SENSOR GAP STATS
# -------------------------------------------------------

def compute_sensor_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station_id, compute:
      - n_closures
      - first_start
      - last_start
      - max_gap_days between consecutive closures
    """
    # Sort by station_id + Start Date
    df_sorted = df.sort_values(["station_id", "Start Date"])

    group = df_sorted.groupby("station_id")["Start Date"]

    first_start = group.min()
    last_start  = group.max()
    counts      = group.size()

    # Compute max gap for each station
    def _max_gap_in_days(ts: pd.Series) -> float:
        # ts is already sorted
        if len(ts) < 2:
            return 0.0
        diffs = ts.diff().dt.total_seconds().dropna() / (24 * 3600.0)
        return float(diffs.max())

    max_gap_days = group.apply(_max_gap_in_days)

    out = pd.DataFrame({
        "station_id": max_gap_days.index,
        "n_closures": counts.values,
        "first_start": first_start.values,
        "last_start": last_start.values,
        "max_gap_days": max_gap_days.values,
    })

    return out


# -------------------------------------------------------
# 2) SENSORS WITH ONLY LATE CLOSURES
# -------------------------------------------------------

def find_sensors_only_late(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find sensors that:
      - have NO closures in the first 70% of global time
      - have at least one closure in the last 30% of global time
    """
    # Global time range from all closures in this file
    global_min = df["Start Date"].min()
    global_max = df["Start Date"].max()

    print(f"[TIME RANGE] global_min = {global_min}, global_max = {global_max}")

    if pd.isna(global_min) or pd.isna(global_max):
        print("[WARN] Cannot compute global time range; returning empty.")
        return pd.DataFrame(columns=["station_id", "has_early", "has_late",
                                     "n_closures", "first_start", "last_start"])

    total_span = global_max - global_min
    cutoff_70 = global_min + 0.7 * total_span

    print(f"[THRESHOLD] 70% cutoff at {cutoff_70}")

    # For each station, see whether it has early / late closures
    def _early_late_flags(group: pd.DataFrame) -> pd.Series:
        starts = group["Start Date"]
        has_early = (starts < cutoff_70).any()
        has_late  = (starts >= cutoff_70).any()
        return pd.Series({
            "has_early": has_early,
            "has_late": has_late,
            "n_closures": len(group),
            "first_start": starts.min(),
            "last_start": starts.max(),
        })

    per_station = (
        df.groupby("station_id", as_index=True)
          .apply(_early_late_flags)
          .reset_index()
    )

    # We want: no early, some late
    filtered = per_station[(per_station["has_early"] == False) & (per_station["has_late"] == True)]

    print(f"[RESULT] Sensors with only late closures: {len(filtered)}")

    return filtered


# -------------------------------------------------------
# 3) CLOSURE DURATIONS FROM START/END
# -------------------------------------------------------

def _closure_durations_minutes(df_cl: pd.DataFrame):
    """
    Compute closure durations in minutes from:
        End Date - Start Date

    Returns:
      (durations_series_in_minutes, unit_str)

    Filters out NaT and negative durations.
    """
    if "Start Date" not in df_cl.columns or "End Date" not in df_cl.columns:
        return None, "N/A (missing Start Date or End Date columns)"

    # Timedelta -> minutes
    delta = df_cl["End Date"] - df_cl["Start Date"]
    # If End Date is NaT, delta will be NaT
    dur = delta.dt.total_seconds() / 60.0
    dur = dur.dropna()

    # Filter negative durations (weird data / same-start-end edge cases if any)
    dur = dur[dur >= 0]

    if dur.empty:
        return None, "N/A (no valid Start/End pairs)"

    return dur, "minutes (End Date - Start Date)"


# -------------------------------------------------------
# 4) SUMMARY PRINTER + GLOBAL SUMMARY
# -------------------------------------------------------

def print_summary(df_cl: pd.DataFrame,
                  gaps_df: pd.DataFrame,
                  late_df: pd.DataFrame) -> dict:
    """
    Print a concise console summary of the results and
    return a dict of summary statistics for global CSV.
    """
    total_rows = len(df_cl)
    unique_sensors = df_cl["station_id"].nunique()

    # Sensors with exactly one closure vs multiple
    sensors_one_closure = (gaps_df["n_closures"] == 1).sum()
    sensors_multi_closure = (gaps_df["n_closures"] > 1).sum()

    # Basic gap stats (only where more than one closure)
    multi_gap_df = gaps_df[gaps_df["n_closures"] > 1]
    if not multi_gap_df.empty:
        max_gap = multi_gap_df["max_gap_days"].max()
        median_gap = multi_gap_df["max_gap_days"].median()
        mean_gap = multi_gap_df["max_gap_days"].mean()
    else:
        max_gap = median_gap = mean_gap = 0.0

    # Closures count stats
    avg_closures_per_sensor = gaps_df["n_closures"].mean()
    median_closures_per_sensor = gaps_df["n_closures"].median()

    # Global time range
    global_min = df_cl["Start Date"].min()
    global_max = df_cl["Start Date"].max()

    only_late_count = len(late_df)

    # Closure duration stats from Start/End
    duration_min = duration_max = duration_mean = duration_median = None
    dur_series, duration_units = _closure_durations_minutes(df_cl)

    if dur_series is not None and not dur_series.empty:
        duration_min = float(dur_series.min())
        duration_max = float(dur_series.max())
        duration_mean = float(dur_series.mean())
        duration_median = float(dur_series.median())

    # --------- PRINT SUMMARY ----------
    print("\n" + "=" * 60)
    print(f"SUMMARY — District {DISTRICT_STR}")
    print("=" * 60)
    print(f"Total closure×station rows used        : {total_rows}")
    print(f"Total sensors with at least 1 closure  : {unique_sensors}")
    print(f"Sensors with exactly 1 closure         : {sensors_one_closure}")
    print(f"Sensors with >1 closure                : {sensors_multi_closure}")
    print(f"Sensors with only late closures (last 30% of time) : {only_late_count}")
    print("")
    print(f"Avg closures per sensor                : {avg_closures_per_sensor:.2f}")
    print(f"Median closures per sensor             : {median_closures_per_sensor:.2f}")
    print("")
    print("Gap stats (only sensors with >1 closure):")
    print(f"  Max gap days                         : {max_gap:.1f}")
    print(f"  Mean gap days                        : {mean_gap:.1f}")
    print(f"  Median gap days                      : {median_gap:.1f}")
    print("")
    print("Global closure time range:")
    print(f"  First closure                        : {global_min}")
    print(f"  Last closure                         : {global_max}")
    print("")
    print("Closure duration stats (from End - Start):")
    print(f"  Units                                : {duration_units}")
    print(f"  Min                                  : {duration_min}")
    print(f"  Max                                  : {duration_max}")
    print(f"  Mean                                 : {duration_mean}")
    print(f"  Median                               : {duration_median}")
    print("=" * 60 + "\n")

    # Pack stats into a dict for global CSV
    summary = {
        "district": DISTRICT,
        "district_str": DISTRICT_STR,
        "total_rows": total_rows,
        "unique_sensors": unique_sensors,
        "sensors_one_closure": int(sensors_one_closure),
        "sensors_multi_closure": int(sensors_multi_closure),
        "sensors_only_late_closures": int(only_late_count),
        "avg_closures_per_sensor": float(avg_closures_per_sensor),
        "median_closures_per_sensor": float(median_closures_per_sensor),
        "max_gap_days": float(max_gap),
        "mean_gap_days": float(mean_gap),
        "median_gap_days": float(median_gap),
        "global_first_closure": global_min,
        "global_last_closure": global_max,
        # duration (minutes, from End-Start)
        "duration_unit": duration_units,
        "duration_min": duration_min,
        "duration_max": duration_max,
        "duration_mean": duration_mean,
        "duration_median": duration_median,
    }

    return summary


def update_global_summary(summary: dict) -> None:
    """
    Append/update the per-district summary row in the global CSV.
    If the file exists and this district is already there, its row is replaced.
    Otherwise a new row is appended.
    """
    GLOBAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    if GLOBAL_SUMMARY.exists():
        df_global = pd.read_csv(GLOBAL_SUMMARY)
    else:
        df_global = pd.DataFrame()

    # Convert to DataFrame
    new_row = pd.DataFrame([summary])

    if not df_global.empty and "district" in df_global.columns:
        # Drop any existing rows for this district
        df_global = df_global[df_global["district"] != summary["district"]]
        df_global = pd.concat([df_global, new_row], ignore_index=True)
    else:
        df_global = new_row

    df_global.to_csv(GLOBAL_SUMMARY, index=False)
    print(f"[SAVE] Global per-district summary → {GLOBAL_SUMMARY}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    df_cl = load_closure_station_data()

    # 1) Compute per-sensor gaps
    print("[STEP 1] Computing per-sensor closure gaps...")
    gaps_df = compute_sensor_gaps(df_cl)
    gaps_df.to_csv(OUT_GAPS, index=False)
    print(f"[SAVE] Sensor gap stats → {OUT_GAPS}")

    # 2) Find sensors with only late closures
    print("[STEP 2] Finding sensors with no early closures but at least one late closure...")
    late_df = find_sensors_only_late(df_cl)
    late_df.to_csv(OUT_LATE, index=False)
    print(f"[SAVE] Sensors with only late closures → {OUT_LATE}")

    # 3) Print summary + collect stats
    summary = print_summary(df_cl, gaps_df, late_df)

    # 4) Update global CSV
    update_global_summary(summary)

    print("[DONE]")


if __name__ == "__main__":
    main()
