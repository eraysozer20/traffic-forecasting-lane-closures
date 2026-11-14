#!/usr/bin/env python3
import os
import csv
import time
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ======================================================
# 1) Config
# ======================================================

DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d: int) -> str:
    return f"{int(d):02d}"

DISTRICT_CODE = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data"

META_DIR = ROOT / "meta" / f"meta{DISTRICT_CODE}"
LATLON_CSV = META_DIR / f"station_latlon_meta{DISTRICT_CODE}.csv"

PEMS_DIR = ROOT / "despt_inputs" / f"pems{DISTRICT_CODE}"

# use CUT CSVs as the source of truth
SPEED_CUT_CSV = PEMS_DIR / "speed_cut.csv"
CLOSURE_CUT_CSV = PEMS_DIR / "closure_cut.csv"

DIST_DIR = ROOT / "dist" / f"pems{DISTRICT_CODE}"
DIST_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS_CSV = DIST_DIR / f"graph_sensor_locations_d{DISTRICT_CODE}.csv"
DISTANCES_CSV = DIST_DIR / f"distances_d{DISTRICT_CODE}_route.csv"

ADJ_DIRECT_CSV = DIST_DIR / "adj.csv"
ADJ1_DIST_CSV  = DIST_DIR / "adj_1.csv"
ADJ1_MILES_CSV = DIST_DIR / "adj_1_miles.csv"

# final grouped folder per district
FINAL_DIR = ROOT / "final" / f"pems{DISTRICT_CODE}"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OSRM_SERVER = os.getenv("OSRM_SERVER_URL", "http://127.0.0.1:5000")

STATION_META_CSV_ENV = os.getenv("PEMS_STATION_META_CSV", "").strip()
STATION_META_CSV = Path(STATION_META_CSV_ENV) if STATION_META_CSV_ENV else None


# ======================================================
# 2) Helpers: locations + OSRM
# ======================================================

def ensure_locations_csv():
    if not LATLON_CSV.exists():
        raise FileNotFoundError(f"Lat/lon meta file not found: {LATLON_CSV}")

    print(f"[INFO] Rebuilding locations file from {LATLON_CSV}")
    df = pd.read_csv(LATLON_CSV)

    required_cols = {"station_id", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing columns in {LATLON_CSV}: {missing}")

    out = (
        df[["station_id", "latitude", "longitude"]]
        .dropna(subset=["station_id", "latitude", "longitude"])
        .drop_duplicates(subset=["station_id"])
        .rename(columns={"station_id": "sensor_id"})
    )
    out["sensor_id"] = out["sensor_id"].astype(str).str.strip()

    out.to_csv(LOCATIONS_CSV, index=False)
    print(f"[INFO] Saved sensor locations to {LOCATIONS_CSV}")


def read_sensors(csv_file_path: Path):
    sensors = []
    with csv_file_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sensors.append(
                {
                    "sensor_id": str(row["sensor_id"]).strip(),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
    return sensors


def test_osrm():
    try:
        url = f"{OSRM_SERVER}/nearest/v1/driving/-121.48412,38.409782"
        r = requests.get(url, timeout=3)
        print(f"[INFO] OSRM probe ({url}) -> status {r.status_code}")
    except Exception as e:
        print(f"[WARN] OSRM probe failed: {e}")


def calculate_distances_osrm(sensors, sleep_between=0.02, max_retries=3):
    results = []
    session = requests.Session()

    for a in tqdm(sensors, desc="Calculating OSRM distances"):
        for b in sensors:
            if a["sensor_id"] == b["sensor_id"]:
                results.append({"from": a["sensor_id"], "to": b["sensor_id"], "cost": 0.0})
                continue

            request_url = (
                f"{OSRM_SERVER}/route/v1/driving/"
                f"{a['longitude']},{a['latitude']};{b['longitude']},{b['latitude']}?overview=false"
            )

            cost = float("inf")
            for attempt in range(max_retries):
                try:
                    resp = session.get(request_url, timeout=10)
                    route_data = resp.json()
                    if route_data.get("code") == "Ok":
                        cost = float(route_data["routes"][0]["distance"])  # meters
                    else:
                        cost = float("inf")
                    break
                except requests.exceptions.RequestException as e:
                    print(
                        f"[WARN] OSRM error {a['sensor_id']} -> {b['sensor_id']} "
                        f"(attempt {attempt+1}/{max_retries}): {e}"
                    )
                    time.sleep(0.5 * (attempt + 1))

            results.append({"from": a["sensor_id"], "to": b["sensor_id"], "cost": cost})
            time.sleep(sleep_between)

    return results


# ======================================================
# 3) Helpers: station meta -> latest -> adjacency
# ======================================================

def find_station_meta_csv(meta_dir: Path) -> Path:
    if STATION_META_CSV is not None:
        if not STATION_META_CSV.exists():
            raise FileNotFoundError(f"PEMS_STATION_META_CSV set but not found: {STATION_META_CSV}")
        return STATION_META_CSV

    candidates = sorted(meta_dir.glob("*.csv"))
    scored = []
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=5)
            cols = set(head.columns)
            need = {"station_id", "snapshot_date", "Fwy", "Abs PM"}
            if need.issubset(cols):
                bonus = 0
                for extra in ["Type", "HOV", "District"]:
                    if extra in cols:
                        bonus += 1
                scored.append((bonus, p))
        except Exception:
            continue

    if not scored:
        raise FileNotFoundError(
            f"Could not auto-find station meta CSV with columns "
            f"station_id,snapshot_date,Fwy,Abs PM in {meta_dir}. "
            f"Set env var PEMS_STATION_META_CSV."
        )

    scored.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    best = scored[0][1]
    print(f"[INFO] Auto-selected station meta CSV: {best}")
    return best


def load_latest_station_meta(meta_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(meta_csv)
    required = {"station_id", "snapshot_date", "Fwy", "Abs PM"}
    missing = required - set(m.columns)
    if missing:
        raise KeyError(f"Station meta missing required columns {missing} in {meta_csv}")

    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"], errors="coerce")
    m = m.dropna(subset=["snapshot_date"]).copy()
    m["station_id"] = m["station_id"].astype(str).str.strip()

    latest = (
        m.sort_values(["station_id", "snapshot_date"])
         .drop_duplicates("station_id", keep="last")
         .reset_index(drop=True)
    )
    return latest


def parse_route_dir(fwy: str):
    if not isinstance(fwy, str):
        return None, None
    fwy = fwy.strip()
    mm = re.match(r"^(.*)-([NSEW])$", fwy)
    if mm:
        return mm.group(1), mm.group(2)
    return fwy, None


def build_adj_from_meta(latest_meta: pd.DataFrame, sensor_ids: list[str], pm_tol: float = 0.02) -> pd.DataFrame:
    nodes = list(map(str, sensor_ids))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    m = latest_meta.copy()
    m["station_id"] = m["station_id"].astype(str).str.strip()
    m = m[m["station_id"].isin(node_to_idx)].copy()

    m["Abs PM"] = pd.to_numeric(m["Abs PM"], errors="coerce")
    m = m.dropna(subset=["Abs PM", "Fwy"]).copy()

    rd = m["Fwy"].apply(parse_route_dir)
    m["route"] = rd.apply(lambda x: x[0])
    m["dir"]   = rd.apply(lambda x: x[1])
    m = m.dropna(subset=["dir"]).copy()

    m["Type_norm"] = m["Type"].astype(str).str.strip().str.lower() if "Type" in m.columns else ""
    m["HOV_norm"]  = m["HOV"].astype(str).str.strip().str.lower() if "HOV" in m.columns else "no"

    def is_hov(row) -> bool:
        return row["HOV_norm"] in ["yes", "y", "true", "1"]

    def is_mainline_like(row) -> bool:
        if is_hov(row):
            return True
        return row["Type_norm"] in ["mainline", "ml"]

    def is_on_ramp(row) -> bool:
        return row["Type_norm"] in ["on ramp", "or"]

    adj = np.zeros((N, N), dtype=np.float32)

    group_cols = ["route", "dir"]
    if "District" in m.columns:
        group_cols.append("District")

    for _, g in m.groupby(group_cols):
        ddir = str(g["dir"].iloc[0]).upper()
        ascending = ddir in ["N", "E"]

        g_sorted = g.sort_values("Abs PM", ascending=ascending).reset_index(drop=True)

        mainline = g_sorted[g_sorted.apply(is_mainline_like, axis=1)].copy().reset_index(drop=True)
        onramps  = g_sorted[g_sorted.apply(is_on_ramp, axis=1)].copy().reset_index(drop=True)

        main_ids = mainline["station_id"].tolist()
        for u, v in zip(main_ids[:-1], main_ids[1:]):
            ui, vi = node_to_idx.get(u), node_to_idx.get(v)
            if ui is not None and vi is not None:
                adj[ui, vi] = 1.0

        if len(mainline) == 0:
            continue

        main_pm  = mainline["Abs PM"].to_numpy(dtype=float)
        main_sid = mainline["station_id"].tolist()

        def attach_mainline_idx(pm_val: float):
            same = np.where(np.abs(main_pm - pm_val) <= pm_tol)[0]
            if same.size > 0:
                return int(same[0])
            return int(np.argmin(np.abs(main_pm - pm_val)))

        for _, r in onramps.iterrows():
            ramp_id = r["station_id"]
            pm_val = float(r["Abs PM"])

            j = attach_mainline_idx(pm_val)
            target_main = main_sid[j]

            ri, ti = node_to_idx.get(ramp_id), node_to_idx.get(target_main)
            if ri is not None and ti is not None:
                adj[ri, ti] = 1.0

    return pd.DataFrame(adj, index=nodes, columns=nodes)


# ======================================================
# 4) Sensor alignment helpers
# ======================================================

def load_cut_csv_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required cut CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=0)
    cols = [str(c).strip() for c in df.columns.tolist()]
    if len(cols) == 0:
        raise ValueError(f"No columns found in {csv_path}")

    return cols[1:]  # first column is timestamp


def load_location_sensor_ids() -> set[str]:
    """
    Use only location sensors that have valid lat/lon.
    """
    if LATLON_CSV.exists():
        df = pd.read_csv(LATLON_CSV)

        required = {"station_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing columns in {LATLON_CSV}: {required - set(df.columns)}")

        df = df.dropna(subset=["station_id", "latitude", "longitude"]).copy()
        df["station_id"] = df["station_id"].astype(str).str.strip()
        return set(df["station_id"].tolist())

    if LOCATIONS_CSV.exists():
        df = pd.read_csv(LOCATIONS_CSV)

        required = {"sensor_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing columns in {LOCATIONS_CSV}: {required - set(df.columns)}")

        df = df.dropna(subset=["sensor_id", "latitude", "longitude"]).copy()
        df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
        return set(df["sensor_id"].tolist())

    raise FileNotFoundError(
        f"Neither {LATLON_CSV} nor {LOCATIONS_CSV} exists for location sensor matching."
    )


def get_common_sensor_ids() -> list[str]:
    speed_ids = load_cut_csv_columns(SPEED_CUT_CSV)
    closure_ids = load_cut_csv_columns(CLOSURE_CUT_CSV)
    location_ids = load_location_sensor_ids()

    speed_set = set(speed_ids)
    closure_set = set(closure_ids)

    # preserve speed_cut order
    common = [sid for sid in speed_ids if sid in closure_set and sid in location_ids]

    if not common:
        raise ValueError("No common sensors across speed_cut.csv, closure_cut.csv, and location data")

    print(f"[INFO] speed_cut sensors:      {len(speed_ids)}")
    print(f"[INFO] closure_cut sensors:    {len(closure_ids)}")
    print(f"[INFO] location sensors:       {len(location_ids)}")
    print(f"[INFO] final common sensors:   {len(common)}")
    print(f"[INFO] removed from speed:     {len(speed_set - set(common))}")
    print(f"[INFO] removed from closure:   {len(closure_set - set(common))}")
    print(f"[INFO] missing in locations:   {len((speed_set & closure_set) - location_ids)}")

    return common


def rewrite_cut_csv_to_common(csv_path: Path, common_sensor_ids: list[str]):
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    keep_cols = [time_col] + [sid for sid in common_sensor_ids if sid in df.columns]
    df = df[keep_cols]
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Rewrote {csv_path.name} with {len(keep_cols)-1} common sensors")


def copy_final_files():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        (SPEED_CUT_CSV, FINAL_DIR / "speed_cut.csv"),
        (CLOSURE_CUT_CSV, FINAL_DIR / "closure_cut.csv"),
        (ADJ_DIRECT_CSV, FINAL_DIR / "adj.csv"),
        (ADJ1_DIST_CSV, FINAL_DIR / "adj_1.csv"),
        (ADJ1_MILES_CSV, FINAL_DIR / "adj_1_miles.csv"),
    ]

    print(f"[INFO] Grouping final files into {FINAL_DIR}")

    for src, dst in files_to_copy:
        if not src.exists():
            raise FileNotFoundError(f"Cannot group missing file: {src}")
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src.name} -> {dst}")

    print(f"[INFO] Final grouped folder ready: {FINAL_DIR}")


# ======================================================
# 5) Main
# ======================================================

def main():
    print(f"[INFO] District: {DISTRICT} (code={DISTRICT_CODE})")
    print(f"[INFO] ROOT:     {ROOT}")
    print(f"[INFO] OSRM:     {OSRM_SERVER}")

    if not SPEED_CUT_CSV.exists():
        raise FileNotFoundError(f"Missing: {SPEED_CUT_CSV}")
    if not CLOSURE_CUT_CSV.exists():
        raise FileNotFoundError(f"Missing: {CLOSURE_CUT_CSV}")

    # final sensor set = intersection of speed_cut, closure_cut, and valid location data
    sensor_ids = get_common_sensor_ids()

    # rewrite speed_cut.csv and closure_cut.csv so they match exactly
    rewrite_cut_csv_to_common(SPEED_CUT_CSV, sensor_ids)
    rewrite_cut_csv_to_common(CLOSURE_CUT_CSV, sensor_ids)

    # always rebuild locations CSV from lat/lon source so it is not stale
    ensure_locations_csv()

    # build sensor list only for final common sensors
    all_sensors = read_sensors(LOCATIONS_CSV)
    sensor_map = {s["sensor_id"]: s for s in all_sensors if s["sensor_id"] in set(sensor_ids)}

    missing_after_rebuild = [sid for sid in sensor_ids if sid not in sensor_map]
    if missing_after_rebuild:
        raise ValueError(
            f"Still missing after rebuild: {missing_after_rebuild[:20]}"
            f"{' ...' if len(missing_after_rebuild) > 20 else ''}"
        )

    sensors = [sensor_map[sid] for sid in sensor_ids]

    # (A) distances: compute/reuse OSRM pairwise route distances
    if DISTANCES_CSV.exists():
        print(f"[INFO] Using existing distances file: {DISTANCES_CSV}")
    else:
        print("[INFO] Distances file not found; computing via OSRM...")
        test_osrm()
        print(f"[INFO] Loaded {len(sensors)} sensor locations")
        distance_results = calculate_distances_osrm(sensors)

        with DISTANCES_CSV.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["from", "to", "cost"])
            writer.writeheader()
            for r in distance_results:
                writer.writerow(r)

        print(f"[INFO] Distances saved to {DISTANCES_CSV}")

    # (B) build distance matrix aligned to common sensor order
    sensor_id_to_index = {sid: idx for idx, sid in enumerate(sensor_ids)}
    N = len(sensor_ids)
    print(f"[INFO] Final aligned sensors: {N}")

    dist_df = pd.read_csv(DISTANCES_CSV)
    dist_df["from"] = dist_df["from"].astype(str).str.strip()
    dist_df["to"]   = dist_df["to"].astype(str).str.strip()

    dist_mx = np.full((N, N), np.inf, dtype=float)
    for _, row in dist_df.iterrows():
        f = row["from"]
        t = row["to"]
        if f in sensor_id_to_index and t in sensor_id_to_index:
            i = sensor_id_to_index[f]
            j = sensor_id_to_index[t]
            dist_mx[i, j] = float(row["cost"])
    np.fill_diagonal(dist_mx, 0.0)

    # save adj_1.csv in meters
    adj1_df = pd.DataFrame(dist_mx, index=sensor_ids, columns=sensor_ids)
    adj1_df = adj1_df.replace(np.inf, 0.0)
    adj1_df.to_csv(ADJ1_DIST_CSV)
    print(f"[INFO] Saved adj_1.csv (meters): {ADJ1_DIST_CSV}")

    # save adj_1_miles.csv
    adj1_miles_df = adj1_df / 1609.344
    adj1_miles_df.to_csv(ADJ1_MILES_CSV)
    print(f"[INFO] Saved adj_1_miles.csv: {ADJ1_MILES_CSV}")

    # build adj.csv with same sensor order
    meta_csv = find_station_meta_csv(META_DIR)
    latest_meta = load_latest_station_meta(meta_csv)

    adj_df = build_adj_from_meta(latest_meta, sensor_ids, pm_tol=0.02)
    adj_df.to_csv(ADJ_DIRECT_CSV)
    print(f"[INFO] Saved adj.csv (direct): {ADJ_DIRECT_CSV}")

    out_deg = (adj_df.values > 0).sum(axis=1)
    in_deg  = (adj_df.values > 0).sum(axis=0)
    print(
        f"[INFO] adj.csv degree stats: avg out={out_deg.mean():.2f}, max out={out_deg.max()}, "
        f"avg in={in_deg.mean():.2f}, max in={in_deg.max()}"
    )

    # (C) group final files in one folder per district
    copy_final_files()

    print("[DONE]")


if __name__ == "__main__":
    main()