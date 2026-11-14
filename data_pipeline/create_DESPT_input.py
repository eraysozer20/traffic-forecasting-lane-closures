#!/usr/bin/env python3
"""
Create DESPT-style H5 inputs (speed + closure labels) from PeMS wide CSVs
and lane closure matches.

Output format:
  - speed  -> speed.h5, key '/sensor_readings'
  - labels -> closure_labels.h5, key '/closure_labels'
  - index: DatetimeIndex named 'timestamp'
  - rows: time (5-min steps)
  - columns: sensor IDs as strings

Mode control:
  - both
  - speed
  - closure
"""

import os
from pathlib import Path
import argparse

import pandas as pd
import numpy as np


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

DISTRICT = "03"

_env_d = os.getenv("PEMS_DISTRICT_ID")
if _env_d:
    DISTRICT = f"{int(_env_d):02d}"

MODE = "both"
_env_mode = os.getenv("PEMS_MODE")
if _env_mode:
    MODE = _env_mode.lower().strip()

ROOT = Path(__file__).resolve().parent / "data"

H5_KEY_SPEED = "sensor_readings"
H5_KEY_LABELS = "closure_labels"

CSV_CHUNKSIZE = 20000
H5_COMPRESSION_LEVEL = 5


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def _detect_time_col(columns):
    for cand in ["timestamp", "date", "time"]:
        if cand in columns:
            return cand
    return None


def _write_speed_csv_to_h5(csv_path: Path, out_path: Path, key: str, what: str, chunksize: int):
    """
    Stream a wide CSV directly into H5 chunk by chunk.
    Returns basic metadata.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{what} CSV not found: {csv_path}")

    print(f"[LOAD] {what} CSV: {csv_path}")

    header_df = pd.read_csv(csv_path, nrows=0)
    original_cols = list(header_df.columns)

    time_col = _detect_time_col(original_cols)
    if time_col is None:
        raise ValueError(
            f"{what} CSV must have a 'date', 'timestamp', or 'time' column. "
            f"Got columns: {original_cols}"
        )

    data_cols = [c for c in original_cols if c != time_col]
    dtype_map = {c: "float32" for c in data_cols}

    print(f"  detected time column: {time_col}")
    print(f"  detected {len(data_cols)} sensor columns")
    print(f"  reading in chunks of {chunksize} rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    total_rows_raw = 0
    total_rows_kept = 0
    min_time = None
    max_time = None
    wrote_any = False

    with pd.HDFStore(
        out_path,
        mode="w",
        complevel=H5_COMPRESSION_LEVEL,
        complib="zlib",
    ) as store:
        for i, chunk in enumerate(
            pd.read_csv(
                csv_path,
                chunksize=chunksize,
                dtype=dtype_map,
                low_memory=True,
            )
        ):
            total_rows_raw += len(chunk)
            print(f"  [chunk {i + 1}] raw shape: {chunk.shape}")

            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
            bad_time = chunk[time_col].isna().sum()
            if bad_time:
                print(f"    dropped {bad_time} rows with invalid timestamps")

            chunk = chunk.dropna(subset=[time_col])

            if chunk.empty:
                continue

            chunk = chunk.set_index(time_col)
            chunk.index.name = "timestamp"
            chunk.columns = chunk.columns.astype(str)

            for c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")

            chunk = chunk.sort_index()

            cmin = chunk.index.min()
            cmax = chunk.index.max()
            min_time = cmin if min_time is None else min(min_time, cmin)
            max_time = cmax if max_time is None else max(max_time, cmax)

            store.append(
                key,
                chunk,
                format="table",
                data_columns=None,
                index=True,
            )

            total_rows_kept += len(chunk)
            wrote_any = True

    if not wrote_any:
        raise ValueError(f"No valid rows found in {csv_path}")

    print(f"✅ Wrote {what} H5 incrementally:")
    print(f"   file:  {out_path}")
    print(f"   key:   /{key}")
    print(f"   rows kept: {total_rows_kept}")
    print(f"   time range: {min_time} → {max_time}")

    return {
        "time_col": time_col,
        "sensor_cols": [str(c) for c in data_cols],
        "rows_raw": total_rows_raw,
        "rows_kept": total_rows_kept,
        "time_min": min_time,
        "time_max": max_time,
    }


def _load_speed_h5(speed_h5: Path, key: str) -> pd.DataFrame:
    if not speed_h5.exists():
        raise FileNotFoundError(f"speed H5 not found: {speed_h5}")

    print(f"[LOAD] existing speed H5: {speed_h5}")
    df = pd.read_hdf(speed_h5, key=key)

    if not df.index.is_monotonic_increasing:
        print("[INFO] Sorting speed H5 by timestamp...")
        df = df.sort_index()

    df.index.name = "timestamp"
    df.columns = df.columns.astype(str)

    print(f"  → speed shape: {df.shape}")
    print(f"    time range: {df.index.min()} → {df.index.max()}")
    return df


def _write_h5_fixed(df: pd.DataFrame, out_path: Path, key: str, what: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(
        out_path,
        mode="w",
        complevel=H5_COMPRESSION_LEVEL,
        complib="zlib",
    ) as store:
        store.put(key, df, format="fixed")

    print(f"  Wrote {what} H5:")
    print(f"   file:  {out_path}")
    print(f"   key:   /{key}")
    print(f"   shape: {df.shape}")


def _parse_station_list(station_str):
    if pd.isna(station_str):
        return []

    s = str(station_str).strip()
    if not s:
        return []

    for sep in [";", ",", "|"]:
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]

    return [s]


def _build_closure_labels_from_matches(speed_df: pd.DataFrame, matches_csv: Path) -> pd.DataFrame:
    if not matches_csv.exists():
        raise FileNotFoundError(f"Lane closure matches CSV not found: {matches_csv}")

    print(f"[LOAD] lane closure matches: {matches_csv}")
    lc = pd.read_csv(matches_csv, low_memory=False)

    if "Status" not in lc.columns:
        raise KeyError(f"'Status' column not found in {matches_csv}")

    lc = lc[lc["Status"] == "Approved"].copy()
    print(f"  → after Status == 'Approved': {len(lc)} rows")

    if "Start Date" not in lc.columns or "End Date" not in lc.columns:
        raise KeyError("Expected 'Start Date' and 'End Date' columns in lane closure CSV")

    lc["Start Date"] = pd.to_datetime(lc["Start Date"], errors="coerce")
    lc["End Date"] = pd.to_datetime(lc["End Date"], errors="coerce")

    before_drop = len(lc)
    lc = lc.dropna(subset=["Start Date", "End Date"])
    print(f"  → after dropping NaT dates: {len(lc)} rows (dropped {before_drop - len(lc)})")

    if "matched_station_ids" not in lc.columns:
        raise KeyError("'matched_station_ids' column not found in lane closure CSV")

    time_index = speed_df.index
    cols = speed_df.columns
    cols_set = set(cols)

    labels = pd.DataFrame(
        np.zeros((len(time_index), len(cols)), dtype=np.uint8),
        index=time_index,
        columns=cols,
    )

    print(f"[INIT] labels dataframe with zeros, shape {labels.shape}, dtype uint8")

    t_min = time_index.min()
    t_max = time_index.max()

    used_events = 0
    used_assignments = 0

    for _, row in lc.iterrows():
        start = row["Start Date"]
        end = row["End Date"]

        if pd.isna(start) or pd.isna(end):
            continue

        if end < start:
            start, end = end, start

        if end < t_min or start > t_max:
            continue

        start_eff = max(start, t_min)
        end_eff = min(end, t_max)

        stations = _parse_station_list(row.get("matched_station_ids", ""))
        stations = [s for s in stations if s in cols_set]

        if not stations:
            continue

        mask = (time_index >= start_eff) & (time_index <= end_eff)
        if not mask.any():
            continue

        labels.loc[mask, stations] = 1
        used_events += 1
        used_assignments += len(stations)

    print("[DONE] Built closure labels from matches.")
    print(f"  used closure events: {used_events}")
    print(f"  total station assignments across events: {used_assignments}")

    return labels


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    global MODE

    parser = argparse.ArgumentParser(description="Build DESPT-style speed / closure H5 files.")
    parser.add_argument("--mode", choices=["both", "speed", "closure"])
    parser.add_argument("--district", type=int, help="District ID, e.g. 4 for pems04")
    parser.add_argument("--chunksize", type=int, default=CSV_CHUNKSIZE)
    args = parser.parse_args()

    if args.mode is not None:
        MODE = args.mode

    district = DISTRICT
    if args.district is not None:
        district = f"{int(args.district):02d}"

    mode = MODE
    do_speed = mode in ("both", "speed")
    do_closure = mode in ("both", "closure")

    root = Path(__file__).resolve().parent / "data"

    speed_csv = root / "pems" / f"pems{district}" / f"pems{district}_all_common_flow.csv"
    lane_closure_csv = root / "lane_closures" / f"pems{district}" / f"pems{district}_lane_closure_matches.csv"

    out_dir = root / "despt_inputs" / f"pems{district}"
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_h5 = out_dir / "speed.h5"
    labels_h5 = out_dir / "closure_labels.h5"

    print(f"=== DESPT input builder for PeMS district {district} ===")
    print(f"Mode (final): {mode}")
    print(f"Speed CSV:        {speed_csv}")
    print(f"Lane closure CSV: {lane_closure_csv}")
    print(f"Speed H5:         {speed_h5}")
    print(f"Labels H5:        {labels_h5}")
    print(f"Chunk size:       {args.chunksize}")

    if do_speed:
        _write_speed_csv_to_h5(
            csv_path=speed_csv,
            out_path=speed_h5,
            key=H5_KEY_SPEED,
            what="speed",
            chunksize=args.chunksize,
        )
    else:
        print("[SKIP] Speed H5 (mode is 'closure' only)")

    if do_closure:
        speed_df = _load_speed_h5(speed_h5, H5_KEY_SPEED)
        labels_df = _build_closure_labels_from_matches(speed_df, lane_closure_csv)
        _write_h5_fixed(labels_df, labels_h5, H5_KEY_LABELS, "closure labels")
    else:
        print("[SKIP] Closure labels H5 (mode is 'speed' only)")

    print("\n[DONE] DESPT-style inputs created.")


if __name__ == "__main__":
    main()