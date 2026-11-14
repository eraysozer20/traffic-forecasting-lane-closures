#!/usr/bin/env python3
"""
Time-aware matching of lane closures to PeMS stations (sensors) for a given district.

Inputs (for DISTRICT_ID = d, code XX):

Lane closures:
    data/lane_closures/pemsXX/pemsXX_lane_closures_*.csv
  (we pick the latest matching file automatically)

Station metadata (merged + compressed with change points only):
    data/meta/metaXX/station_meta_distd_YYYY-YYYY.csv
  (e.g. station_meta_dist3_2001-2024.csv)

Output:
    data/lane_closures/pemsXX/pemsXX_lane_closure_matches.csv

Matching is time-aware:
    - same route+direction string (Fwy-Dir in closures, Fwy in meta)
    - station Abs PM within [Begin Abs PM, End Abs PM]
    - closure Start Date within station's [valid_from, valid_to) interval
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import os

# ====================== USER CONFIG ======================

# Change this to switch districts
DISTRICT_ID = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d: int) -> str:
    return f"{int(d):02d}"


DIST_CODE = district_code(DISTRICT_ID)

SCRIPT_DIR = Path(__file__).resolve().parent

LANE_DIR = SCRIPT_DIR / "data" / "lane_closures" / f"pems{DIST_CODE}"
META_DIR = SCRIPT_DIR / "data" / "meta" / f"meta{DIST_CODE}"

# ---- Column names in lane closures: ADJUST IF NEEDED ----
# From your sample:
# ID, Log #, District, Fwy-Dir, Begin County, Begin Abs PM, ...
C_CLS_ROUTE = "Fwy-Dir"         # combined freeway + direction, e.g. "I80-W", "SR1-S"
C_CLS_BEG_PM = "Begin Abs PM"   # begin absolute postmile (float)
C_CLS_END_PM = "End Abs PM"     # end absolute postmile (float)
C_CLS_START_TIME = "Start Date" # closure start date/time

# ---- Column names in station meta: ADJUST IF NEEDED ----
# From your sample:
# Fwy, District, County, City, CA PM, Abs PM, Length, station_id, ...
C_STN_ROUTE = "Fwy"             # combined freeway + direction, e.g. "SR1-S"
C_STN_ABS_PM = "Abs PM"
C_STN_ID = "station_id"         # we ensured this when building meta

# For the last interval, we cap valid_to at:
#   (max snapshot_date in meta) + padding days
INTERVAL_END_PADDING_DAYS = 30

# ========================================================


def find_latest_lane_csv() -> Path:
    """
    Find the most recent lane-closures CSV for this district in LANE_DIR.
    Pattern: pemsXX_lane_closures_*.csv
    """
    pattern = f"pems{DIST_CODE}_lane_closures_*.csv"
    files = sorted(LANE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No lane closures CSV matching {pattern} found in {LANE_DIR}"
        )
    latest = files[-1]
    print(f"[LANE] Using lane closure file: {latest}")
    return latest


def find_station_meta_merged() -> Path:
    """
    Find the merged station meta CSV for this district in META_DIR.
    We prefer a file whose name includes a year range, e.g. station_meta_dist3_2001-2024.csv.
    """
    pattern = f"station_meta_dist{DISTRICT_ID}_*.csv"
    files = sorted(META_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No station meta CSV matching {pattern} found in {META_DIR}"
        )

    candidates: List[Path] = []
    for f in files:
        tail = f.stem.split(f"station_meta_dist{DISTRICT_ID}_", 1)[-1]
        if "-" in tail:
            candidates.append(f)
    if candidates:
        chosen = candidates[-1]
    else:
        chosen = files[-1]

    print(f"[META] Using station meta file: {chosen}")
    return chosen


# ----------------- STATION META PREP ----------------- #

def prepare_station_meta(meta_path: Path) -> pd.DataFrame:
    """
    Load station metadata CSV. Normalizes station_id and snapshot_date.
    Expects meta to be the merged+compressed file (first + changed snapshots).
    """
    df = pd.read_csv(meta_path)

    # Detect station ID column if C_STN_ID isn't there yet
    station_col = C_STN_ID
    if station_col not in df.columns:
        candidates = ["ID", "Id", "id", "Station", "STATION", "Station ID", "station_id"]
        for c in candidates:
            if c in df.columns:
                station_col = c
                break
        if station_col not in df.columns:
            raise KeyError(
                f"Could not find station ID column among {list(df.columns)}; "
                f"update C_STN_ID or candidates."
            )

    df[station_col] = df[station_col].astype(str)
    if station_col != "station_id":
        df = df.rename(columns={station_col: "station_id"})

    if "snapshot_date" not in df.columns:
        raise KeyError(
            "Expected 'snapshot_date' column in station meta; "
            "make sure you're using the merged+compressed meta file."
        )
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    # Normalize Abs PM
    if C_STN_ABS_PM in df.columns:
        df[C_STN_ABS_PM] = pd.to_numeric(df[C_STN_ABS_PM], errors="coerce")

    # Route string as-is (SR1-S etc.) – no splitting needed
    if C_STN_ROUTE not in df.columns:
        raise KeyError(
            f"Expected station route column '{C_STN_ROUTE}' not found in meta."
        )

    return df


def build_station_intervals(df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build station intervals from meta:

    For each station_id, sorted by snapshot_date, produce rows:

        station_id, route, Abs PM, valid_from, valid_to

    valid_from = snapshot_date of row i
    valid_to   = snapshot_date of row i+1 (next change), or
                 (max_snapshot_date + padding) for the last row.
    """
    max_date = df_meta["snapshot_date"].max()
    global_end = max_date + pd.Timedelta(days=INTERVAL_END_PADDING_DAYS)

    df_meta = df_meta.sort_values(["station_id", "snapshot_date"])

    intervals = []

    for station_id, g in df_meta.groupby("station_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(n):
            row = g.iloc[i]

            valid_from = row["snapshot_date"]
            if i < n - 1:
                valid_to = g.iloc[i + 1]["snapshot_date"]
            else:
                valid_to = global_end

            intervals.append(
                {
                    "station_id": station_id,
                    C_STN_ROUTE: row.get(C_STN_ROUTE, None),
                    C_STN_ABS_PM: row.get(C_STN_ABS_PM, None),
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                }
            )

    df_int = pd.DataFrame(intervals)

    # Sanity
    df_int[C_STN_ABS_PM] = pd.to_numeric(df_int[C_STN_ABS_PM], errors="coerce")

    print(f"[META] Built station intervals table with {len(df_int)} rows.")
    return df_int


# ----------------- CLOSURE TABLE PREP ----------------- #

def prepare_closure_table(lane_path: Path) -> pd.DataFrame:
    """
    Load lane closure table and normalize:
    - Begin/End Abs PM to numeric
    - Start Date to datetime
    """
    # low_memory=False to avoid DtypeWarning (mixed columns)
    df = pd.read_csv(lane_path, low_memory=False)

    for col in (C_CLS_ROUTE, C_CLS_BEG_PM, C_CLS_END_PM):
        if col not in df.columns:
            raise KeyError(
                f"Expected closure column '{col}' not found in {lane_path}.\n"
                f"Adjust C_CLS_ROUTE / C_CLS_BEG_PM / C_CLS_END_PM."
            )

    if C_CLS_START_TIME not in df.columns:
        raise KeyError(
            f"Expected closure start-time column '{C_CLS_START_TIME}' not found in {lane_path}.\n"
            "Adjust C_CLS_START_TIME to match your CSV header."
        )

    # Route string e.g. "I80-W", just strip spaces
    df[C_CLS_ROUTE] = df[C_CLS_ROUTE].astype(str).str.strip()

    # PMs
    df[C_CLS_BEG_PM] = pd.to_numeric(df[C_CLS_BEG_PM], errors="coerce")
    df[C_CLS_END_PM] = pd.to_numeric(df[C_CLS_END_PM], errors="coerce")

    # Time
    df[C_CLS_START_TIME] = pd.to_datetime(df[C_CLS_START_TIME], errors="coerce")

    print(f"[LANE] Loaded {len(df)} closures.")
    return df


# ----------------- MATCHING ----------------- #

def match_time_aware(
    closures: pd.DataFrame,
    stations_int: pd.DataFrame,
) -> pd.DataFrame:
    """
    Time-aware matching using station intervals.

    For each closure row, with closure_time = Start Date:

        - same route string (Fwy-Dir vs Fwy)
        - station Abs PM in [Begin Abs PM, End Abs PM]
        - valid_from <= closure_time < valid_to
    """
    matched_ids_list: List[str] = []
    matched_counts: List[int] = []

    # Pre-group station intervals by route
    stations_group = stations_int.groupby(
        C_STN_ROUTE, dropna=False
    )

    total = len(closures)
    for idx, row in closures.iterrows():
        if idx % 5000 == 0 and idx > 0:
            print(f"[MATCH] processed {idx}/{total} closures...")

        route = row[C_CLS_ROUTE]
        beg_pm = row[C_CLS_BEG_PM]
        end_pm = row[C_CLS_END_PM]
        t_c = row[C_CLS_START_TIME]

        if pd.isna(route) or pd.isna(beg_pm) or pd.isna(end_pm) or pd.isna(t_c):
            matched_ids_list.append("")
            matched_counts.append(0)
            continue

        lo = min(beg_pm, end_pm)
        hi = max(beg_pm, end_pm)

        if route not in stations_group.groups:
            matched_ids_list.append("")
            matched_counts.append(0)
            continue

        stn_sub = stations_group.get_group(route)

        # Spatial filter
        mask_pm = stn_sub[C_STN_ABS_PM].between(lo, hi)
        stn_hits = stn_sub.loc[mask_pm]

        # Time filter
        mask_time = (stn_hits["valid_from"] <= t_c) & (t_c < stn_hits["valid_to"])
        stn_hits = stn_hits.loc[mask_time]

        if stn_hits.empty:
            matched_ids_list.append("")
            matched_counts.append(0)
        else:
            ids_str = ";".join(stn_hits["station_id"].astype(str).tolist())
            matched_ids_list.append(ids_str)
            matched_counts.append(len(stn_hits))

    closures = closures.copy()
    closures["matched_station_ids"] = matched_ids_list
    closures["matched_station_count"] = matched_counts

    print(
        f"[MATCH] {sum(c > 0 for c in matched_counts)} closures matched "
        f"to at least one station (time-aware)."
    )

    return closures


# ----------------- MAIN ----------------- #

def main():
    lane_path = find_latest_lane_csv()
    meta_path = find_station_meta_merged()

    closures = prepare_closure_table(lane_path)
    df_meta = prepare_station_meta(meta_path)
    stations_int = build_station_intervals(df_meta)

    matched = match_time_aware(closures, stations_int)

    out_path = LANE_DIR / f"pems{DIST_CODE}_lane_closure_matches.csv"
    matched.to_csv(out_path, index=False)
    print(f"[SAVE] Wrote matches to {out_path}")


if __name__ == "__main__":
    main()
