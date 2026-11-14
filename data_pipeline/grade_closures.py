#!/usr/bin/env python3
"""
Script 1: Grade Closures by V-Shape Impact (Drop + Recovery)
------------------------------------------------------------
Calculates a continuous 'closure_grade' [0-1] based on:
1. Drop Score: How much flow dropped compared to 'Before'.
2. Recovery Score: How much flow bounced back in 'After'.
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import os
import numpy as np

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "8"))
def district_code(d): return f"{int(d):02d}"

DISTRICT = district_code(DISTRICT)
ROOT = Path(__file__).resolve().parent / "data"

# INPUTS
CLOSURE_FILE = ROOT / "lane_closures" / f"pems{DISTRICT}" / f"pems{DISTRICT}_lane_closure_matches.csv"
PEMS_FILE    = ROOT / "pems" / f"pems{DISTRICT}" / f"pems{DISTRICT}_all_common_flow.csv"

# OUTPUT
OUTPUT_FILE  = ROOT / "lane_closures" / f"pems{DISTRICT}" / f"pems{DISTRICT}_graded_closures_v2.csv"

GOOD_STATUSES = {"Approved", "Completed", "A", "C", "Standard", "Permitted"}

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
def load_closure_data() -> pd.DataFrame:
    print(f"[LOAD] Reading {CLOSURE_FILE}...")
    df = pd.read_csv(CLOSURE_FILE, low_memory=False)
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"]   = pd.to_datetime(df["End Date"], errors="coerce")
    df = df[df["matched_station_count"] > 0]
    df = df[df["Status"].isin(GOOD_STATUSES)]
    return df

def explode_sensors(df: pd.DataFrame) -> pd.DataFrame:
    def split_ids(val):
        if pd.isna(val) or val == "": return []
        return [s for s in str(val).split(";") if s.strip() != ""]

    df = df.copy()
    df["sensor_list"] = df["matched_station_ids"].apply(split_ids)
    df = df.explode("sensor_list")
    df["sensor_list"] = df["sensor_list"].astype(str)
    return df.rename(columns={"sensor_list": "station_id"})

def load_pems_data(required_station_ids) -> pd.DataFrame:
    print(f"[PEMS] Scanning columns in {PEMS_FILE}...")
    header = pd.read_csv(PEMS_FILE, nrows=0)
    available_cols = set(header.columns)
    required_station_ids = [str(s) for s in required_station_ids]
    
    usecols = ["date"] + [sid for sid in required_station_ids if sid in available_cols]
    print(f"[PEMS] Loading {len(usecols)} columns...")
    
    dtype_map = {col: "float32" for col in usecols if col != "date"}
    
    chunks = []
    for chunk in pd.read_csv(PEMS_FILE, usecols=usecols, dtype=dtype_map, engine="python", chunksize=50_000):
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    return df

# -------------------------------------------------------
# ANALYSIS & GRADING LOGIC
# -------------------------------------------------------
def extract_speed_windows(df_cl: pd.DataFrame, df_sp: pd.DataFrame) -> pd.DataFrame:
    records = []
    print(f"[ANALYZE] Processing {len(df_cl)} rows...")

    for i, (_, row) in enumerate(df_cl.iterrows(), start=1):
        sid = row["station_id"]
        if sid not in df_sp.columns: continue

        start = row["Start Date"]
        end   = row["End Date"]

        # Define Windows
        # Before: 1 hour prior to 15 mins prior
        before_start = start - timedelta(hours=1)
        before_end   = start - timedelta(minutes=15)
        
        # After: 15 mins after to 1 hour after
        after_start = end + timedelta(minutes=15)
        after_end   = end + timedelta(hours=1)
        
        try:
            # Calculate MEAN FLOW for each window
            before_val = df_sp.loc[before_start:before_end, sid].mean()
            during_val = df_sp.loc[start:end, sid].mean()
            after_val  = df_sp.loc[after_start:after_end, sid].mean()
        except:
            before_val, during_val, after_val = None, None, None

        out = row.to_dict()
        out["before_flow"] = before_val
        out["during_flow"] = during_val
        out["after_flow"]  = after_val
        
        # --- GRADING MATH ---
        grade = 0.0
        drop_score = 0.0
        rec_score = 0.0
        
        if pd.notna(before_val) and pd.notna(during_val) and pd.notna(after_val):
            # 1. Drop Score: How much did it fall from Before?
            # Range: 0.0 (No drop) to 1.0 (Total stop)
            if before_val > 10: # Only grade if road was active
                drop_score = (before_val - during_val) / before_val
            
            # 2. Recovery Score: How much did it bounce back compared to After?
            # Range: 0.0 (No recovery) to 1.0 (Full recovery from zero)
            if after_val > 10:
                rec_score = (after_val - during_val) / after_val
            
            # Clip negative values (if flow INCREASED during closure, grade is 0)
            drop_score = max(0.0, drop_score)
            rec_score  = max(0.0, rec_score)
            
            # 3. Final Grade: Average of Drop and Recovery
            # We enforce a "V-Shape": If recovery is 0, grade is penalized heavily
            grade = (drop_score + rec_score) / 2.0
            
            # Penalty for "Night/Empty" Ambiguity
            # If absolute flow was tiny, the ratio is noisy and unreliable.
            if before_val < 50 or after_val < 50:
                grade = grade * 0.5  # Penalize low-flow events

        out["drop_score"] = drop_score
        out["recovery_score"] = rec_score
        out["closure_grade"] = grade

        records.append(out)
        if i % 5000 == 0: print(f"  Processed {i}...")

    return pd.DataFrame(records)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    # 1. Load & Prep
    df_cl = load_closure_data()
    df_expanded = explode_sensors(df_cl)
    
    # 2. Load Flow Data
    station_ids = sorted(df_expanded["station_id"].unique())
    df_sp = load_pems_data(station_ids)
    
    # 3. Analyze
    df_out = extract_speed_windows(df_expanded, df_sp)
    
    # 4. Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    
    # Stats
    high_grade = df_out[df_out['closure_grade'] > 0.4]
    print(f"\n[RESULTS]")
    print(f"Total Events: {len(df_out)}")
    print(f"High Grade Events (>0.4): {len(high_grade)}")
    print(f"Average Grade: {df_out['closure_grade'].mean():.3f}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()