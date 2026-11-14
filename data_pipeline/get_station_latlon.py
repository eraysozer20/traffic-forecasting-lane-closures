#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd


# ======================================================
# 1. District + paths
# ======================================================

DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "3"))

def district_code(d):
    return f"{int(d):02d}"   # 3 -> "03", 10 -> "10"

DISTRICT = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data"

# meta folder: data/meta/meta{DISTRICT}, e.g. data/meta/meta03
META_DIR = ROOT / "meta" / f"meta{DISTRICT}"


def find_station_id_csv(meta_dir: Path) -> Path:
    """
    Find the CSV file used by the old script to determine which station_ids
    are needed. Excludes previously generated station_latlon files.
    """
    csv_files = [
        p for p in meta_dir.glob("*.csv")
        if not p.name.startswith("station_latlon_")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No station-id CSV files found in {meta_dir}")

    csv_files = sorted(csv_files)
    print(f"[INFO] Using station-id CSV: {csv_files[0].name}")
    return csv_files[0]


def find_text_meta_file(meta_dir: Path, district: str) -> Path:
    """
    Find downloaded PeMS text metadata file, e.g.
    d03_text_meta_2025_12_30.txt

    Prefer the latest one by filename sort.
    """
    txt_files = sorted(meta_dir.glob(f"d{district}_text_meta_*.txt"))
    if not txt_files:
        # fallback: any txt
        txt_files = sorted(meta_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No text metadata files found in {meta_dir}")

    chosen = txt_files[-1]   # latest by sorted filename
    print(f"[INFO] Using text metadata file: {chosen.name}")
    return chosen


META_FILE = find_station_id_csv(META_DIR)
TEXT_META_FILE = find_text_meta_file(META_DIR, DISTRICT)

# output: same as old script
OUT_FILE = META_DIR / f"station_latlon_meta{DISTRICT}.csv"


# ======================================================
# 2. Main
# ======================================================

def main():
    print(f"[INFO] District: {DISTRICT}")
    print(f"[INFO] Meta dir: {META_DIR}")
    print(f"[INFO] Station-id CSV: {META_FILE.name}")
    print(f"[INFO] Text meta file: {TEXT_META_FILE.name}")

    # -----------------------------
    # A. Get the station IDs we need
    # -----------------------------
    df_meta = pd.read_csv(META_FILE)

    if "station_id" not in df_meta.columns:
        raise KeyError(f"'station_id' column not found in {META_FILE}")

    station_ids = (
        df_meta["station_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"[INFO] Found {len(station_ids)} unique station IDs from CSV")

    # -----------------------------
    # B. Load downloaded text metadata
    # -----------------------------
    df_txt = pd.read_csv(TEXT_META_FILE, sep="\t")

    required_cols = ["ID", "Latitude", "Longitude"]
    missing = [c for c in required_cols if c not in df_txt.columns]
    if missing:
        raise KeyError(f"Missing required columns in {TEXT_META_FILE}: {missing}")

    df_txt = df_txt[["ID", "Latitude", "Longitude"]].copy()
    df_txt = df_txt.rename(columns={
        "ID": "station_id",
        "Latitude": "latitude",
        "Longitude": "longitude",
    })

    df_txt["station_id"] = pd.to_numeric(df_txt["station_id"], errors="coerce")
    df_txt["latitude"] = pd.to_numeric(df_txt["latitude"], errors="coerce")
    df_txt["longitude"] = pd.to_numeric(df_txt["longitude"], errors="coerce")

    df_txt = df_txt.dropna(subset=["station_id", "latitude", "longitude"]).copy()
    df_txt["station_id"] = df_txt["station_id"].astype(int)

    # if duplicates exist in text meta, keep first
    df_txt = df_txt.drop_duplicates(subset=["station_id"])

    print(f"[INFO] Text metadata has {len(df_txt)} unique stations with lat/lon")

    # -----------------------------
    # C. Filter to only needed stations
    # -----------------------------
    needed_df = pd.DataFrame({"station_id": station_ids})
    out_df = needed_df.merge(df_txt, on="station_id", how="left")

    matched = out_df["latitude"].notna().sum()
    missing_count = out_df["latitude"].isna().sum()

    print(f"[INFO] Matched lat/lon for {matched} / {len(out_df)} requested stations")

    if missing_count > 0:
        missing_ids = out_df.loc[out_df["latitude"].isna(), "station_id"].tolist()
        print(f"[WARN] Missing lat/lon for {missing_count} stations")
        print(f"[WARN] Missing station IDs: {missing_ids[:20]}{' ...' if len(missing_ids) > 20 else ''}")

    # save same format as old script
    out_df.to_csv(OUT_FILE, index=False)

    print(f"[DONE] Saved {len(out_df)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()