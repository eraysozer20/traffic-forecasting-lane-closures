#!/usr/bin/env python3
"""
Build a detectors.add.xml for D08 based on station→edge mapping (lat/lon).
Each station becomes an e1Detector on lane 0 at mid-edge.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# SUMO / sumolib
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise RuntimeError("Please set SUMO_HOME before running this script")

import sumolib  # type: ignore


def build_detectors(
    net_path: Path,
    mapping_csv: Path,
    out_xml: Path,
    freq_s: int = 300,
    out_file_name: str = "d08_detector_output.xml",
):
    print(f"[LOAD] net: {net_path}")
    net = sumolib.net.readNet(str(net_path))

    print(f"[LOAD] station-edge map: {mapping_csv}")
    df = pd.read_csv(mapping_csv)

    required = ["station_id", "edge_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in mapping CSV {mapping_csv}")

    with open(out_xml, "w") as f:
        f.write("<additional>\n")
        for _, row in df.iterrows():
            sid = row["station_id"]
            edge_id = row["edge_id"]

            try:
                edge = net.getEdge(edge_id)
            except Exception:
                print(f"[WARN] edge {edge_id} not in net, skip station {sid}")
                continue

            lane0 = edge.getLane(0)
            lane_id = lane0.getID()
            pos = edge.getLength() / 2.0  # middle of edge

            det_id = f"det_{sid}"
            f.write(
                f'  <e1Detector id="{det_id}" lane="{lane_id}" '
                f'pos="{pos:.1f}" freq="{freq_s}" file="{out_file_name}"/>\n'
            )
        f.write("</additional>\n")

    print(f"[SAVE] detectors → {out_xml}")


if __name__ == "__main__":
    NET_PATH = Path("/mnt/c/Users/erays/Desktop/pemsdata/data/osrm/d08_freeways.net.xml")
    MAP_CSV = Path("/mnt/c/Users/erays/Desktop/pemsdata/data/meta/meta08/d08_station_edge_map_latlon.csv")
    OUT_XML = Path("/mnt/c/Users/erays/Desktop/pemsdata/data/osrm/d08_detectors.add.xml")

    build_detectors(NET_PATH, MAP_CSV, OUT_XML)
