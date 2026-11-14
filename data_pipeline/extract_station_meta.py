#!/usr/bin/env python3
"""
Fetch station metadata (elv/stations) for a given PeMS district, year by year,
filter to the station IDs used in your aggregated data, and detect changes
in station info over time.

- Station IDs are taken from: data/pems/pemsXX/pemsXX_d.csv
  * First column is 'date' → ignored.
  * Remaining column names are treated as station IDs (strings).

- Per-year snapshots are saved under: data/meta/metaXX/
  * station_meta_dist{DISTRICT_ID}_{YEAR}.csv

- Final merged + change-flagged file:
  * station_meta_dist{DISTRICT_ID}_{startyear}-{endyear}.csv
"""

from __future__ import annotations
import os

import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from io import BytesIO

import pandas as pd

# =========================================================
# CONFIG
# =========================================================

PEMS_BASE_URL = "https://pems.dot.ca.gov"
LOGIN_URL = f"{PEMS_BASE_URL}/?dnode=login"

# 🔁 Change this to switch districts (3,4,5,6,7,8,10,11,12)
DISTRICT_ID = 12

# Station export URL: root with query params (like lane-closure script)
STATIONS_URL = f"{PEMS_BASE_URL}/"

# PeMS credentials are read from the PEMS_USERNAME / PEMS_PASSWORD environment variables.
# ❌ Do NOT push these to git.
PEMS_USERNAME = os.environ.get("PEMS_USERNAME", "")
PEMS_PASSWORD = os.environ.get("PEMS_PASSWORD", "")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR

# Where your aggregated pemsXX_d.csv files live
PEMS_DATA_ROOT = ROOT_DIR / "data" / "pems"

# Where to store meta outputs
META_ROOT = ROOT_DIR / "data" / "meta"

# Per-district date ranges (month/year) for station metadata snapshots
# Format: district_id: (start_year, start_month, end_year, end_month)
DISTRICT_DATE_RANGES: Dict[int, Tuple[int, int, int, int]] = {
    3:  (2001, 3,  2024, 3),   # PEMS03_Agg 03/2001 - 03/2024
    4:  (2002, 6,  2024, 3),   # PEMS04_Agg 06/2002 - 03/2024
    5:  (2012, 3,  2024, 3),   # PEMS05_Agg 03/2012 - 03/2024
    6:  (2009, 12, 2024, 3),   # PEMS06_Agg 12/2009 - 03/2024
    7:  (2002, 6,  2024, 3),   # PEMS07_Agg 06/2002 - 03/2024
    8:  (2001, 3,  2024, 3),   # PEMS08_Agg 03/2001 - 03/2024
    10: (2007, 6,  2024, 3),   # PEMS10_Agg 06/2007 - 03/2024
    11: (2002, 9,  2024, 3),   # PEMS11_Agg 09/2002 - 03/2024
    12: (2002, 1,  2024, 3),   # PEMS12_Agg 01/2002 - 03/2024
}

# Toggle:
#   True  → login + fetch XLSX from PeMS, write per-year CSVs, then merge
#   False → only read existing per-year CSVs and merge
FETCH_FROM_PEMS = True

# =========================================================
# BASIC LOGIN (same style as lane-closure script)
# =========================================================

def create_session() -> requests.Session:
    s = requests.Session()
    print("Session created:", s)
    return s


def login(session: requests.Session) -> None:
    payload = {
        "username": PEMS_USERNAME,
        "password": PEMS_PASSWORD,
        "login": "Login",
    }

    print(f"\n[LOGIN] POST {LOGIN_URL}")
    resp = session.post(LOGIN_URL, data=payload)

    print("[LOGIN] HTTP status:", resp.status_code)
    print("[LOGIN] Response length:", len(resp.text))

    print("\n[LOGIN] Response snippet (first 300 chars):\n")
    print(resp.text[:300])
    print("\n--- END LOGIN SNIPPET ---\n")


# =========================================================
# HELPERS
# =========================================================

def district_code(d: int) -> str:
    """Return two-digit district code, e.g. 3 → '03'."""
    return f"{d:02d}"


def get_district_dates(d: int) -> Tuple[datetime, datetime]:
    """
    Convert the month/year range for a district into concrete start/end datetimes.
    We only use them to generate yearly snapshot dates.
    """
    if d not in DISTRICT_DATE_RANGES:
        raise ValueError(f"No date range configured for district {d}")
    sy, sm, ey, em = DISTRICT_DATE_RANGES[d]
    start_dt = datetime(sy, sm, 1, 0, 0)
    end_dt = datetime(ey, em, 31, 23, 59)
    return start_dt, end_dt


def get_station_ids_for_district(d: int) -> List[str]:
    """
    Read station IDs from data/pems/pemsXX/pemsXX_d.csv.
    - The first column is named 'date' → ignore it.
    - All other column names are treated as station IDs (strings).
    """
    code = district_code(d)
    csv_path = PEMS_DATA_ROOT / f"pems{code}" / f"pems{code}_d.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find station CSV: {csv_path}")

    print(f"[STATIONS] Reading header from {csv_path}")
    df_header = pd.read_csv(csv_path, nrows=0)
    cols = list(df_header.columns)

    station_cols = [c for c in cols if c.lower() != "date"]
    print(f"[STATIONS] Found {len(station_cols)} station IDs (ignoring 'date').")
    return station_cols


def iter_snapshot_dates(start_dt: datetime, end_dt: datetime):
    """
    Yield snapshot dates year by year, using the same month/day as start_dt,
    until we pass end_dt.
    """
    month = start_dt.month
    day = start_dt.day
    year = start_dt.year

    while True:
        snap = datetime(year, month, day, 0, 0)
        if snap > end_dt:
            break
        yield snap
        year += 1


def detect_station_id_column(df: pd.DataFrame) -> str:
    """
    Try to guess which column is the station ID column.
    Adjust candidate list if PeMS uses different naming.
    """
    candidates = [
        "station_id", "STATION_ID",
        "ID", "Id", "id",
        "Station", "STATION", "station",
        "ID#", "Station ID"
    ]
    for c in candidates:
        if c in df.columns:
            print(f"[META] Using '{c}' as station ID column.")
            return c
    raise KeyError(
        f"Could not find a station ID column in columns: {list(df.columns)}. "
        "Update detect_station_id_column() with the correct column name."
    )


# =========================================================
# STATION EXPORT QUERY (EXPORT XLSX)
# =========================================================

def build_station_query(snapshot_dt: datetime) -> dict:
    """
    Build query params for station metadata EXPORT (xlsx),
    mimicking the "Export XLSX" button.

    Example button URL:
    https://pems.dot.ca.gov/?report_form=1&dnode=District&content=elv&tab=stations
        &export=xlsx&district_id=3&_time_id=1067990400&_time_id_f=11/05/2003
        &eqpo=&tag=&st_cd=on&st_ch=on&st_ff=on&st_hv=on&st_ml=on&st_fr=on&st_or=on
    """
    unix_ts = int(snapshot_dt.replace(tzinfo=timezone.utc).timestamp())

    params = {
        "report_form": "1",
        "dnode": "District",
        "content": "elv",
        "tab": "stations",

        "district_id": str(DISTRICT_ID),

        # export as real Excel (xlsx)
        "export": "xlsx",

        # time fields used by the stations export
        "_time_id": str(unix_ts),
        "_time_id_f": snapshot_dt.strftime("%m/%d/%Y"),

        # extra fields/filters from the export form
        "eqpo": "",
        "tag": "",

        # station type checkboxes – all enabled like your URL
        "st_cd": "on",
        "st_ch": "on",
        "st_ff": "on",
        "st_hv": "on",
        "st_ml": "on",
        "st_fr": "on",
        "st_or": "on",
    }

    return params


def fetch_station_snapshot(
    session: requests.Session,
    snapshot_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch a single station-metadata snapshot for DISTRICT_ID at snapshot_dt.

    Uses the EXPORT XLSX endpoint (same style as your lane-closure script),
    then reads the binary Excel content with pandas.read_excel.
    """
    params = build_station_query(snapshot_dt)

    print(f"\n[FETCH] Station metadata for district {DISTRICT_ID} at {snapshot_dt}")
    print("[FETCH] GET", STATIONS_URL)
    print("[FETCH] Query params:", params)

    resp = session.get(STATIONS_URL, params=params)

    print("[FETCH] HTTP status:", resp.status_code)
    print("[FETCH] Content-Type:", resp.headers.get("Content-Type"))

    # Debug preview of first 300 bytes (as text)
    snippet = resp.content[:300]
    try:
        txt = snippet.decode("utf-8", errors="replace")
    except Exception:
        txt = str(snippet)
    print("\n[FETCH] Preview (first 300 bytes as text):\n")
    print(txt)
    print("\n--- END FETCH PREVIEW ---\n")

    # Parse as Excel (xlsx)
    df = pd.read_excel(BytesIO(resp.content))
    print(f"[FETCH] Retrieved {len(df)} rows, {len(df.columns)} columns from XLSX.")
    return df


# =========================================================
# MERGE + CHANGE DETECTION
# =========================================================

def filter_and_flag_changes(
    frames: List[pd.DataFrame],
    station_ids: List[str],
) -> pd.DataFrame:
    """
    - Concatenate all yearly snapshot frames.
    - Detect station ID column.
    - Filter to given list of station IDs.
    - Normalize values (NaNs, whitespace) to avoid false positives.
    - For each station:
        * First snapshot row -> changed_from_prev = True
        * Later row -> True if any metadata field changed vs previous snapshot
    - Finally, KEEP ONLY rows with changed_from_prev == True.
    """
    if not frames:
        raise ValueError("No frames to merge – did any snapshots succeed?")

    # 1) Stack all snapshots
    merged = pd.concat(frames, ignore_index=True)

    # 2) Detect and normalize station_id
    station_col = detect_station_id_column(merged)
    merged[station_col] = merged[station_col].astype(str)
    merged = merged.rename(columns={station_col: "station_id"})

    # 3) Filter to the stations you actually use
    station_ids_str = {str(s) for s in station_ids}
    before = len(merged)
    merged = merged[merged["station_id"].isin(station_ids_str)]
    after = len(merged)
    print(f"[FILTER] Kept {after}/{before} rows after filtering to station IDs.")

    if "snapshot_date" not in merged.columns:
        raise KeyError("Expected 'snapshot_date' column not found in merged data.")

    merged["snapshot_date"] = pd.to_datetime(merged["snapshot_date"])

    # 4) Sort by station + time
    merged = merged.sort_values(["station_id", "snapshot_date"]).reset_index(drop=True)

    # 5) Decide which columns to compare (exclude ID + date)
    ignore_cols = {"station_id", "snapshot_date"}
    compare_cols = [c for c in merged.columns if c not in ignore_cols]

    # 6) Normalize values to avoid NaN/whitespace false positives
    #    - convert everything to string
    #    - strip whitespace
    #    - treat "nan", "NaN", "None" as empty
    merged_norm = merged.copy()
    if compare_cols:
        merged_norm[compare_cols] = (
            merged_norm[compare_cols]
            .apply(lambda col: col.astype("string"))
            .fillna("")
            .apply(lambda col: col.str.strip())
            .replace({"nan": "", "NaN": "", "None": ""})
        )

    # 7) Per-station change detection
    if compare_cols:
        def _group_changes(g: pd.DataFrame) -> pd.Series:
            g_cmp = g[compare_cols]
            changed = g_cmp.ne(g_cmp.shift()).any(axis=1)
            # First snapshot is the base definition -> mark True
            if len(changed) > 0:
                changed.iloc[0] = True
            return changed

        changed_series = (
            merged_norm.groupby("station_id", group_keys=False).apply(_group_changes)
        )
        merged["changed_from_prev"] = changed_series.values
    else:
        # If for some reason there are no compare columns, just keep first row per station.
        merged["changed_from_prev"] = (
            merged.groupby("station_id").cumcount() == 0
        )

    total_true = merged["changed_from_prev"].sum()
    print(f"[CHANGES] {total_true} rows with changed_from_prev=True before filtering.")

    # 8) Keep only changed rows (including first snapshot per station)
    merged = merged[merged["changed_from_prev"]].reset_index(drop=True)
    print(f"[CHANGES] After filtering, {len(merged)} rows kept (only informative snapshots).")

    return merged


# =========================================================
# MAIN
# =========================================================

def main():
    # Date range for this district
    start_dt, end_dt = get_district_dates(DISTRICT_ID)
    print(f"[INFO] District {DISTRICT_ID} date range: {start_dt.date()} → {end_dt.date()}")

    # Station IDs from pemsXX_d.csv (ignoring 'date')
    station_ids = get_station_ids_for_district(DISTRICT_ID)

    # Output directory
    code = district_code(DISTRICT_ID)
    meta_dir = META_ROOT / f"meta{code}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Meta output directory: {meta_dir}")

    frames: List[pd.DataFrame] = []

    if FETCH_FROM_PEMS:
        # ----------------- FETCH FROM PEMS ----------------- #
        session = create_session()
        login(session)

        for snap_dt in iter_snapshot_dates(start_dt, end_dt):
            try:
                df_snap = fetch_station_snapshot(session, snap_dt)
            except Exception as e:
                print(f"[WARN] Failed to fetch snapshot for {snap_dt.date()}: {e}")
                continue

            df_snap["snapshot_date"] = snap_dt
            frames.append(df_snap)

            year = snap_dt.year
            raw_path = meta_dir / f"station_meta_dist{DISTRICT_ID}_{year}.csv"
            df_snap.to_csv(raw_path, index=False)
            print(f"[SAVE] Saved raw snapshot to {raw_path}")
    else:
        # ----------------- LOAD EXISTING SNAPSHOTS ----------------- #
        pattern = f"station_meta_dist{DISTRICT_ID}_*.csv"
        csv_files = sorted(meta_dir.glob(pattern))

        for f in csv_files:
            # Skip any final merged file like station_meta_dist3_2001-2024.csv
            tail = f.stem.split(f"station_meta_dist{DISTRICT_ID}_", 1)[-1]
            if "-" in tail:
                print(f"[SKIP] Skipping merged/range file {f.name}")
                continue

            print(f"[LOAD] Reading snapshot file {f.name}")
            df = pd.read_csv(f)

            if "snapshot_date" in df.columns:
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
            else:
                print(f"[WARN] snapshot_date column missing in {f.name}; "
                      "you may need to reconstruct it if required.")
            frames.append(df)

    if not frames:
        print("[ERROR] No snapshot data available – nothing to merge.")
        return

    # Merge + change detection (this already keeps only 'changed' rows)
    merged = filter_and_flag_changes(frames, station_ids)

    sy, sm, ey, em = DISTRICT_DATE_RANGES[DISTRICT_ID]
    merged_path = meta_dir / f"station_meta_dist{DISTRICT_ID}_{sy}-{ey}.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[SAVE] Saved merged station metadata with changes to {merged_path}")

    # ----------------- CLEAN UP YEARLY SNAPSHOTS ----------------- #
    pattern = f"station_meta_dist{DISTRICT_ID}_*.csv"
    csv_files = list(meta_dir.glob(pattern))

    print("[CLEANUP] Deleting yearly snapshot files...")
    for f in csv_files:
        tail = f.stem.split(f"station_meta_dist{DISTRICT_ID}_", 1)[-1]
        # Skip the merged/range file (it has a dash like '2001-2024')
        if "-" in tail:
            print(f"[CLEANUP] Keeping merged file {f.name}")
            continue

        try:
            f.unlink()
            print(f"[CLEANUP] Deleted {f.name}")
        except Exception as e:
            print(f"[CLEANUP] Could not delete {f.name}: {e}")



if __name__ == "__main__":
    main()
