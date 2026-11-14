#!/usr/bin/env python3
"""
Convert both:
  - closure_cut.h5 -> closure_cut.csv
  - speed_cut.h5   -> speed_cut.csv

Output CSV format:
,110-04174,110+04177,...
1/3/2019 1:00,59.16,63.62,...

No CLI args needed.
"""

import os
from pathlib import Path
import pandas as pd

# ======================================================
# PATHS
# ======================================================
DISTRICT = "03"   # default

_env_d = os.getenv("PEMS_DISTRICT_ID")
if _env_d:
    DISTRICT = f"{int(_env_d):02d}"

BASE_DIR = Path(r"data\despt_inputs\pems" + DISTRICT)

FILES = [
    {
        "h5": BASE_DIR / "closure_cut.h5",
        "csv": BASE_DIR / "closure_cut.csv",
        "key": None,
    },
    {
        "h5": BASE_DIR / "speed_cut.h5",
        "csv": BASE_DIR / "speed_cut.csv",
        "key": None,
    },
]

# If an H5 has multiple keys, set it here instead of None.
# Example:
# FILES[0]["key"] = "closure_labels"
# FILES[1]["key"] = "speed"

# Windows datetime format: 1/3/2019 1:00
DTFMT = "%#m/%#d/%Y %#H:%M"


def convert_one(h5_path: Path, out_csv: Path, key=None):
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    df = pd.read_hdf(h5_path) if key is None else pd.read_hdf(h5_path, key=key)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        bad = df.index.isna().sum()
        raise ValueError(f"{bad} index values could not be parsed as datetimes in {h5_path}.")

    df = df.copy()
    df.index = df.index.strftime(DTFMT)
    df.columns = df.columns.astype(str)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True, index_label="")

    print(f"[OK] Wrote:  {out_csv}")
    print(f"[INFO] H5:    {h5_path}")
    print(f"[INFO] shape: {df.shape} (rows, sensors)")
    print("-" * 40)


def main():
    print(f"[INFO] District: {DISTRICT}")
    print(f"[INFO] Base dir:  {BASE_DIR}")
    print("=" * 40)

    for item in FILES:
        convert_one(item["h5"], item["csv"], item["key"])

    print("[DONE] Converted all cut H5 files to CSV.")


if __name__ == "__main__":
    main()