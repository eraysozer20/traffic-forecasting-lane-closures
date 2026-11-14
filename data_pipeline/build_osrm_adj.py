#!/usr/bin/env python3
import os
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import requests
from tqdm import tqdm


# ======================================================
# 1. Config: districts + paths (matching your style)
# ======================================================

DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d):
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DISTRICT_CODE = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data"

# Meta + lat/lon
META_DIR   = ROOT / "meta" / f"meta{DISTRICT_CODE}"
LATLON_CSV = META_DIR / f"station_latlon_meta{DISTRICT_CODE}.csv"

# Time-series HDF5 (speed data)
PEMS_DIR   = ROOT / "despt_inputs" / f"pems{DISTRICT_CODE}"
PEMS_H5    = PEMS_DIR / "speed.h5"

# Output directory for CSVs
DIST_DIR   = ROOT / "dist" / f"pems{DISTRICT_CODE}"
DIST_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS_CSV = DIST_DIR / f"graph_sensor_locations_d{DISTRICT_CODE}.csv"
DISTANCES_CSV = DIST_DIR / f"distances_d{DISTRICT_CODE}_route.csv"
ADJ_CSV       = DIST_DIR / f"adj_mx_d{DISTRICT_CODE}.csv"

# Two ADJ pkls in despt_inputs/pemsXX
ADJ_MX_DIAG0_PKL = PEMS_DIR / "adj_mx_diag0.pkl"  # diag = 0
ADJ_MX_DIAG1_PKL = PEMS_DIR / "adj_mx_diag1.pkl"  # diag = 1

OSRM_SERVER = os.getenv("OSRM_SERVER_URL", "http://127.0.0.1:5000")


# ======================================================
# 2. Helpers
# ======================================================

def ensure_locations_csv():
    """
    Ensure graph_sensor_locations_dXX.csv exists in DIST_DIR.
    If not, build it from station_latlon_metaXX.csv in META_DIR.
    """
    if LOCATIONS_CSV.exists():
        print(f"[INFO] Using existing locations file: {LOCATIONS_CSV}")
        return

    if not LATLON_CSV.exists():
        raise FileNotFoundError(f"Lat/lon meta file not found: {LATLON_CSV}")

    print(f"[INFO] Creating locations file from {LATLON_CSV}")

    df = pd.read_csv(LATLON_CSV)
    required_cols = {"station_id", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing columns in {LATLON_CSV}: {missing}")

    df_filtered = (
        df[["station_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"station_id": "sensor_id"})
    )

    df_filtered.to_csv(LOCATIONS_CSV, index=False)
    print(f"[INFO] Saved sensor locations to {LOCATIONS_CSV}")


def read_sensors(csv_file_path: Path):
    """Read sensor_id, lat, lon from CSV into a list of dicts."""
    sensors = []
    with csv_file_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sensors.append(
                {
                    "sensor_id": str(row["sensor_id"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
    return sensors


def test_osrm():
    """Quick sanity check that OSRM is reachable."""
    url = f"{OSRM_SERVER}/health"
    try:
        r = requests.get(url, timeout=2)
        print(f"[INFO] OSRM health check ({url}) -> status {r.status_code}")
    except Exception as e:
        print(f"[WARN] OSRM health check failed: {e}")


def calculate_distances_osrm(sensors, sleep_between=0.02, max_retries=3):
    """
    Compute OSRM route distance (meters) between all sensor pairs.

    - Uses a single requests.Session to reuse TCP connections
    - Sleeps a bit between requests to avoid port exhaustion
    - Retries a few times on network errors
    """
    results = []
    session = requests.Session()

    for sensor_a in tqdm(sensors, desc="Calculating OSRM distances"):
        for sensor_b in sensors:
            if sensor_a["sensor_id"] == sensor_b["sensor_id"]:
                cost = 0.0
                results.append({
                    "from": sensor_a["sensor_id"],
                    "to": sensor_b["sensor_id"],
                    "cost": cost,
                })
                continue

            request_url = (
                f"{OSRM_SERVER}/route/v1/driving/"
                f"{sensor_a['longitude']},{sensor_a['latitude']};"
                f"{sensor_b['longitude']},{sensor_b['latitude']}?overview=false"
            )

            cost = float("inf")
            for attempt in range(max_retries):
                try:
                    resp = session.get(request_url, timeout=10)
                    route_data = resp.json()
                    if route_data.get("code") == "Ok":
                        cost = route_data["routes"][0]["distance"]  # meters
                    else:
                        cost = float("inf")
                    break
                except requests.exceptions.RequestException as e:
                    print(
                        f"Error between {sensor_a['sensor_id']} and {sensor_b['sensor_id']} "
                        f"(attempt {attempt+1}/{max_retries}): {e}"
                    )
                    time.sleep(0.5 * (attempt + 1))

            results.append({
                "from": sensor_a["sensor_id"],
                "to": sensor_b["sensor_id"],
                "cost": cost,
            })

            time.sleep(sleep_between)

    return results


def get_adjacency_matrix(dist_mx_array, normalized_k=0.1):
    """
    Turn distance matrix into adjacency via Gaussian kernel.

    adj[i,j] = exp( - (d/std)^2 )
    entries below normalized_k are zeroed for sparsity.
    """
    finite = ~np.isinf(dist_mx_array)
    distances = dist_mx_array[finite].flatten()
    std = distances.std() if distances.size > 0 else 1.0
    adj_mx_array = np.exp(-np.square(dist_mx_array / std))
    adj_mx_array[adj_mx_array < normalized_k] = 0
    return adj_mx_array


# ======================================================
# 3. Main pipeline (reuses distances if already there)
# ======================================================

def main():
    print(f"[INFO] District: {DISTRICT} (code={DISTRICT_CODE})")
    print(f"[INFO] ROOT:     {ROOT}")
    print(f"[INFO] OSRM:     {OSRM_SERVER}")
    print(f"[INFO] PEMS_H5:  {PEMS_H5}")

    if not PEMS_H5.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {PEMS_H5}\n"
            "Make sure speed.h5 is at data/despt_inputs/pemsXX/speed.h5"
        )

    ensure_locations_csv()

    # --- STEP 1: distances (reuse if already computed) ---
    if DISTANCES_CSV.exists():
        print(f"[INFO] Using existing distances file: {DISTANCES_CSV}")
    else:
        print(f"[INFO] Distances file not found, will compute via OSRM.")
        test_osrm()

        sensors = read_sensors(LOCATIONS_CSV)
        print(f"[INFO] Loaded {len(sensors)} sensor locations")

        distance_results = calculate_distances_osrm(sensors)

        with DISTANCES_CSV.open(mode="w", newline="") as csvfile:
            fieldnames = ["from", "to", "cost"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in distance_results:
                writer.writerow(result)

        print(f"[INFO] Distances saved to {DISTANCES_CSV}")

    # --- STEP 2: build distance matrix aligned with HDF5 ---
    df = pd.read_hdf(PEMS_H5)
    sensor_ids = df.columns.astype(str).tolist()
    sensor_id_to_index = {sid: idx for idx, sid in enumerate(sensor_ids)}
    num_sensors = len(sensor_ids)
    print(f"[INFO] HDF5 sensors: {num_sensors}")

    dist_df = pd.read_csv(DISTANCES_CSV)
    dist_df["from"] = dist_df["from"].astype(str)
    dist_df["to"]   = dist_df["to"].astype(str)

    dist_mx_array = np.full((num_sensors, num_sensors), np.inf, dtype=float)

    for _, row in dist_df.iterrows():
        from_sensor = row["from"]
        to_sensor   = row["to"]
        if from_sensor in sensor_id_to_index and to_sensor in sensor_id_to_index:
            i = sensor_id_to_index[from_sensor]
            j = sensor_id_to_index[to_sensor]
            dist_mx_array[i, j] = row["cost"]

    np.fill_diagonal(dist_mx_array, 0.0)

    # --- STEP 3: distance → adjacency ---
    adj_mx_array = get_adjacency_matrix(dist_mx_array).astype(np.float32)

    # diag1: keep self-connections (diagonal = 1)
    adj_diag1 = adj_mx_array.copy()
    np.fill_diagonal(adj_diag1, 1.0)

    # diag0: remove self-connections (diagonal = 0)
    adj_diag0 = adj_mx_array.copy()
    np.fill_diagonal(adj_diag0, 0.0)


    # --- STEP 4: save two ADJ pkls in despt_inputs/pemsXX ---
    payload_diag0 = [sensor_ids, sensor_id_to_index, adj_diag0]
    payload_diag1 = [sensor_ids, sensor_id_to_index, adj_diag1]

    with ADJ_MX_DIAG0_PKL.open("wb") as f:
        pickle.dump(payload_diag0, f)
    with ADJ_MX_DIAG1_PKL.open("wb") as f:
        pickle.dump(payload_diag1, f)

    print(f"[INFO] Adjacency diag0 PKL saved to {ADJ_MX_DIAG0_PKL}")
    print(f"[INFO] Adjacency diag1 PKL saved to {ADJ_MX_DIAG1_PKL}")


    # --- STEP 5: also save adjacency CSV in dist/pemsXX ---
    np.savetxt(ADJ_CSV, adj_diag1, delimiter=",", fmt="%.6f")
    print(f"[INFO] Adjacency CSV saved to {ADJ_CSV}")



if __name__ == "__main__":
    main()
