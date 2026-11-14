#!/usr/bin/env python3
"""
Analyze speed impacts of lane closures using:
  - matched closure file
  - PeMS speed data (wide CSV)

This script:
  ✓ keeps ONLY closures with matched sensors
  ✓ removes only rejected/canceled closures
  ✓ expands closure rows (1 per sensor)
  ✓ loads ONLY the needed station columns from PeMS CSV
  ✓ extracts before/during/after speeds
  ✓ outputs a clean analysis file
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import os

# -------------------------------------------------------
# CONFIG – CHANGE DISTRICT HERE
# -------------------------------------------------------
DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d):
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DISTRICT = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data"
CLOSURE_FILE = ROOT / "lane_closures" / f"pems{DISTRICT}" / f"pems{DISTRICT}_lane_closure_matches.csv"
PEMS_FILE    = ROOT / "pems" / f"pems{DISTRICT}" / f"pems{DISTRICT}_all_common_flow.csv"

OUTPUT_FILE  = ROOT / "lane_closures" / f"pems{DISTRICT}" / f"pems{DISTRICT}_closure_speed_impacts.csv"

# -------------------------------------------------------
# STATUS FILTER – Keep only actual performed work
# -------------------------------------------------------
GOOD_STATUSES = {
    "Approved", "Completed",
    "A", "C", "Standard", "Permitted"
}

# -------------------------------------------------------
# 1. Load data
# -------------------------------------------------------

def load_closure_data() -> pd.DataFrame:
    df = pd.read_csv(CLOSURE_FILE, low_memory=False)

    # Convert timestamps
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"]   = pd.to_datetime(df["End Date"], errors="coerce")

    # Filter closures with matched sensors
    df = df[df["matched_station_count"] > 0]

    # Filter status
    df = df[df["Status"].isin(GOOD_STATUSES)]

    return df


def load_pems_data(required_station_ids) -> pd.DataFrame:
    """
    Load PeMS data but ONLY for the needed station ids + 'date' column,
    using chunked reading and the Python engine to avoid out-of-memory
    errors in the C parser.
    """
    print(f"[PEMS] Reading header from {PEMS_FILE} to discover columns...")
    header = pd.read_csv(PEMS_FILE, nrows=0)
    available_cols = set(header.columns)

    # Ensure station IDs are strings
    required_station_ids = [str(s) for s in required_station_ids]

    usecols = ["date"]
    for sid in required_station_ids:
        if sid in available_cols:
            usecols.append(sid)

    print(f"[PEMS] Will load {len(usecols)-1} station columns + 'date' (total {len(usecols)} columns).")

    # Dtype: keep 'date' default, everything else as float32 to save memory
    dtype_map = {col: "float32" for col in usecols if col != "date"}

    chunks = []
    print("[PEMS] Reading CSV in chunks with python engine...")
    for i, chunk in enumerate(
        pd.read_csv(
            PEMS_FILE,
            usecols=usecols,
            dtype=dtype_map,
            engine="python",   # <- python engine, no low_memory
            chunksize=50_000   # <- tune smaller if needed (e.g., 20_000)
        ),
        start=1
    ):
        chunks.append(chunk)
        if i % 20 == 0:
            print(f"  [PEMS] {i} chunks loaded...")

    df = pd.concat(chunks, ignore_index=True)
    print(f"[PEMS] Concatenated chunks → shape {df.shape}")

    # Convert and set index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)

    print(f"[PEMS] Final speed table shape: {df.shape}")
    return df



# -------------------------------------------------------
# 2. Expand closures into one row per station
# -------------------------------------------------------

def explode_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert matched_station_ids into one row per station.
    """
    def split_ids(val):
        if pd.isna(val) or val == "":
            return []
        return [s for s in str(val).split(";") if s.strip() != ""]

    df = df.copy()
    df["sensor_list"] = df["matched_station_ids"].apply(split_ids)

    df_expanded = df.explode("sensor_list")
    df_expanded["sensor_list"] = df_expanded["sensor_list"].astype(str)

    return df_expanded.rename(columns={"sensor_list": "station_id"})


# -------------------------------------------------------
# 3. Extract BEFORE / DURING / AFTER speeds
# -------------------------------------------------------

def extract_speed_windows(df_cl: pd.DataFrame, df_sp: pd.DataFrame, log_every: int = 5000) -> pd.DataFrame:
    """
    For each closure row (closure × station):
      - before_speed: mean of 15–60 min before
      - during_speed: mean during the closure
      - after_speed: mean of 15–60 min after

    log_every: print progress every N rows.
    """
    records = []
    total = len(df_cl)
    print(f"[ANALYZE] Total closure×station rows to process: {total}")

    for i, (_, row) in enumerate(df_cl.iterrows(), start=1):
        sid = row["station_id"]

        if sid not in df_sp.columns:
            # skip if station has no speed data
            if (i % log_every) == 0:
                print(f"  [PROGRESS] {i}/{total} rows processed (station {sid} missing in speed table)")
            continue

        start = row["Start Date"]
        end   = row["End Date"]

        # define windows
        before_start = start - timedelta(hours=1)
        before_end   = start - timedelta(minutes=15)

        after_start = end + timedelta(minutes=15)
        after_end   = end + timedelta(hours=1)

        # extract
        try:
            before_speed = df_sp.loc[before_start:before_end, sid].mean()
            during_speed = df_sp.loc[start:end, sid].mean()
            after_speed  = df_sp.loc[after_start:after_end, sid].mean()
        except Exception:
            before_speed, during_speed, after_speed = None, None, None

        out = row.to_dict()
        out["before_speed"] = before_speed
        out["during_speed"]  = during_speed
        out["after_speed"]   = after_speed

        # effect sizes (check for NaN instead of truthiness)
        if pd.notna(before_speed) and pd.notna(during_speed):
            out["drop_from_before"] = float(before_speed) - float(during_speed)
        else:
            out["drop_from_before"] = None

        if pd.notna(during_speed) and pd.notna(after_speed):
            out["recovery_after"] = float(after_speed) - float(during_speed)
        else:
            out["recovery_after"] = None

        records.append(out)

        # progress log
        if (i % log_every) == 0:
            print(f"  [PROGRESS] {i}/{total} rows processed")

    print("[ANALYZE] Finished processing all rows.")
    return pd.DataFrame(records)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print(f"[LOAD] Closures: {CLOSURE_FILE}")
    df_cl = load_closure_data()
    print(f"  → {len(df_cl)} closures with matched sensors and valid status")

    print("[EXPAND] Exploding matched sensors...")
    df_expanded = explode_sensors(df_cl)
    print(f"  → {len(df_expanded)} closure × station rows")

    # Determine which station IDs we actually need from the big PeMS file
    station_ids = sorted(df_expanded["station_id"].unique())
    print(f"[INFO] Unique station IDs to load from PeMS: {len(station_ids)}")

    print(f"[LOAD] PEMS subset: {PEMS_FILE}")
    df_sp = load_pems_data(station_ids)

    print("[ANALYZE] Extracting before/during/after speeds...")
    df_out = extract_speed_windows(df_expanded, df_sp, log_every=5000)
    print(f"  → {len(df_out)} rows analyzed")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"[DONE] Saved analysis to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
