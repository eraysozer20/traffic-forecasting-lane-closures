#!/usr/bin/env python3
import os
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ======================================================
# 1. District + paths (same style as your other scripts)
# ======================================================

DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "3"))

def district_code(d):
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DISTRICT = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data"

# meta folder: data/meta/meta{DISTRICT}, e.g. data/meta/meta08
META_DIR   = ROOT / "meta" / f"meta{DISTRICT}"

# we’ll auto-detect the meta CSV in this folder
def find_meta_csv(meta_dir: Path) -> Path:
    """
    Find a CSV file in meta_dir to read station_id from.
    If you have a specific name, hardcode it here instead.
    """
    csv_files = list(meta_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {meta_dir}")
    # if you prefer a specific one, filter here:
    # csv_files = [f for f in csv_files if "meta" in f.name.lower()]
    # and then pick csv_files[0]
    print(f"[INFO] Using meta file: {csv_files[0].name}")
    return csv_files[0]

META_FILE  = find_meta_csv(META_DIR)

# output: stays inside the same meta folder
OUT_FILE   = META_DIR / f"station_latlon_meta{DISTRICT}.csv"


# ======================================================
# 2. PeMS session cookie (from env)
# ======================================================
# Set this before running, e.g.:
#   $env:PEMS_PHPSESSID="d4f4c4a1babc83db0f35308360781385"
PEMS_PHPSESSID = os.getenv("PEMS_PHPSESSID", "d57f9797deba93cd76076de2ea7b38c5")


if not PEMS_PHPSESSID:
    raise RuntimeError(
        "PEMS_PHPSESSID env var not set. "
        "Set it to your current PHPSESSID (session cookie) from PeMS."
    )

HEADERS = {
    "Cookie": f"PHPSESSID={PEMS_PHPSESSID}"
}


# ======================================================
# 3. HTML scraper
# ======================================================

def fetch_lat_lon(station_id: int):
    """
    Fetch latitude and longitude for a given station_id
    from the PeMS VDS page.
    """
    url = f"https://pems.dot.ca.gov/?report_form=1&dnode=VDS&station_id={station_id}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"[WARN] station {station_id}: HTTP {r.status_code}")
        return None, None

    soup = BeautifulSoup(r.text, "html.parser")

    lat = None
    lng = None

    # Heuristic: find a row whose last two <td> cells are floats (Lat, Lng)
    for row in soup.find_all("tr"):
        tds = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(tds) >= 4:
            try:
                maybe_lat = float(tds[-2])
                maybe_lng = float(tds[-1])
                lat, lng = maybe_lat, maybe_lng
                break
            except ValueError:
                continue

    if lat is None or lng is None:
        print(f"[WARN] station {station_id}: could not parse lat/lon")

    return lat, lng


# ======================================================
# 4. Main: load station IDs, scrape, save
# ======================================================

def main():
    print(f"[INFO] District: {DISTRICT}")
    print(f"[INFO] Meta dir: {META_DIR}")
    print(f"[INFO] Meta file: {META_FILE.name}")

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

    print(f"[INFO] Found {len(station_ids)} unique station IDs")

    rows = []
    for i, sid in enumerate(station_ids, start=1):
        print(f"[{i}/{len(station_ids)}] Fetching {sid}...")
        lat, lng = fetch_lat_lon(sid)
        rows.append(
            {
                "station_id": sid,
                "latitude": lat,
                "longitude": lng,
            }
        )
        time.sleep(0.5)   # be nice to the server

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"[DONE] Saved {len(out_df)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
