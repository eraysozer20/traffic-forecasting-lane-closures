#!/usr/bin/env python3
"""
Map PeMS District 8 stations (lat/lon-only file) to SUMO edges.

Input CSV columns:
    station_id, latitude, longitude

Output CSV columns:
    station_id, latitude, longitude, edge_id, dist_to_edge_m
"""

import os
import sys
from pathlib import Path

import pandas as pd

# --- SUMO tools / sumolib import ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise RuntimeError("Please set SUMO_HOME before running this script")

import sumolib  # type: ignore


def map_stations_to_edges(
    net_path: Path,
    station_csv: Path,
    out_csv: Path,
    search_radius_m: float = 500.0,
):
    print(f"[LOAD] net: {net_path}")
    net = sumolib.net.readNet(str(net_path))

    print(f"[LOAD] station info (lat/lon only): {station_csv}")
    df = pd.read_csv(station_csv)

    # Expect exactly these columns
    required_cols = ["station_id", "latitude", "longitude"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} in {df.columns.tolist()}")

    records = []

    for idx, row in df.iterrows():
        sid = row["station_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # plain-geo: lon/lat are used directly as x/y
        x, y = net.convertLonLat2XY(lon, lat)
        edges = net.getNeighboringEdges(x, y, search_radius_m)

        if not edges:
            print(f"[WARN] No edge within {search_radius_m} m for station {sid} at ({lat},{lon})")
            continue

        # edges is list of (edge, dist); invert to sort by dist
        dist, closest_edge = sorted([(d, e) for (e, d) in edges], key=lambda t: t[0])[0]
        edge_id = closest_edge.getID()

        records.append(
            {
                "station_id": sid,
                "latitude": lat,
                "longitude": lon,
                "edge_id": edge_id,
                "dist_to_edge_m": dist,
            }
        )

    out_df = pd.DataFrame(records)
    print(f"[SAVE] {len(out_df)} mapped stations → {out_csv}")
    out_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    NET_PATH = Path(r"/mnt/c/Users/erays/Desktop/pemsdata/data/osrm/d08_freeways.net.xml")
    STATION_CSV = Path(r"/mnt/c/Users/erays/Desktop/pemsdata/data/meta/meta08/station_latlon_meta08.csv")
    OUT_CSV = Path(r"/mnt/c/Users/erays/Desktop/pemsdata/data/meta/meta08/d08_station_edge_map_latlon.csv")

    map_stations_to_edges(NET_PATH, STATION_CSV, OUT_CSV)
