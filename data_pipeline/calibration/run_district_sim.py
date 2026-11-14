# calibration/run_district_sim.py
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List

import pandas as pd

from config import CalibConfig  # or from .config import CalibConfig if you run as a package


def run_command(cmd: list, cwd: Optional[Path] = None):
    """Small helper to run external commands and print errors nicely."""
    print("▶", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                            text=True, capture_output=True)
    if result.returncode != 0:
        print("❌ SUMO error:\n", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def ensure_sumocfg_matches_cfg(cfg: CalibConfig) -> Path:
    """
    Make sure we have a SUMO config file that points to:
      - net = sumo_net_file
      - route-files = sumo_route_file
      - additional-files = sumo_detectors_add

    If cfg.sumo_cfg_file already exists, we KEEP it.
    If you want to overwrite, delete it manually.
    """
    sumocfg_path = Path(cfg.sumo_cfg_file)
    os.makedirs(sumocfg_path.parent, exist_ok=True)

    if sumocfg_path.exists():
        print(f"[INFO] Using existing SUMO config: {sumocfg_path}")
        return sumocfg_path

    # Otherwise, create a minimal configuration like in clean_grid.py
    net_basename      = Path(cfg.sumo_net_file).name
    route_basename    = Path(cfg.sumo_route_file).name
    detectors_basename= Path(cfg.sumo_detectors_add).name

    xml = f"""<configuration>
    <input>
        <net-file value="{net_basename}"/>
        <route-files value="{route_basename}"/>
        <additional-files value="{detectors_basename}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{cfg.sim_duration_sec}"/>
        <step-length value="1.0"/>
    </time>
</configuration>
"""
    sumocfg_path.write_text(xml, encoding="utf-8")
    print(f"[INFO] Created SUMO config at: {sumocfg_path}")
    return sumocfg_path


def convert_detector_xml_to_csv(
    xml_file: Path,
    output_csv: Path,
    expected_sensors: Optional[List[str]] = None
):
    """
    Parse e1Detector output XML -> wide CSV:
      rows = sensor_id
      cols = minute (0, 5, 10, ..., 1435 for a full day at freq=300s)
    This matches the style used in clean_grid.py.
    """
    xml_file = Path(xml_file)
    if not xml_file.exists():
        raise FileNotFoundError(f"Detector output XML not found: {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []

    for interval in root.findall("interval"):
        begin_sec = float(interval.attrib.get("begin", "0"))
        sensor_id = interval.attrib.get("id", "")
        count = int(float(interval.attrib.get("nVehEntered", 0)))
        minute = int(begin_sec // 60)  # 0, 5, 10, ... if freq=300
        records.append({
            "minute": minute,
            "sensor_id": sensor_id,
            "vehicle_count": count,
        })

    if not records:
        print(f"⚠️ No <interval> records in {xml_file}; writing empty CSV.")
        df = pd.DataFrame()
        os.makedirs(output_csv.parent, exist_ok=True)
        df.to_csv(output_csv)
        return

    df_long = pd.DataFrame(records)

    # wide pivot: index = sensor, columns = minute
    df = (
        df_long
        .pivot(index="sensor_id", columns="minute", values="vehicle_count")
        .fillna(0)
        .astype(int)
    )

    # optional padding to a known sensor set
    if expected_sensors is not None and len(df.columns) > 0:
        missing = [s for s in expected_sensors if s not in df.index]
        for s in missing:
            df.loc[s] = [0] * len(df.columns)
        df = df.sort_index()

    df.columns = [str(c) for c in df.columns]  # make column names strings
    os.makedirs(output_csv.parent, exist_ok=True)
    df.to_csv(output_csv)

    print(f"✅ Wrote detector CSV: {output_csv}")
    print(f"   shape = {df.shape[0]} sensors × {df.shape[1]} time bins")


def run_open_district_simulation(use_gui: bool = True):
    """
    One open, no-closure run on the district freeway network.

    1) Ensure SUMO config exists & points to correct files.
    2) Run sumo / sumo-gui with that config.
    3) Convert detector XML -> 5-min CSV.
    """
    cfg = CalibConfig()

    net_file   = Path(cfg.sumo_net_file)
    route_file = Path(cfg.sumo_route_file)
    det_add    = Path(cfg.sumo_detectors_add)
    det_xml    = Path(cfg.sumo_detector_output)
    out_csv    = Path(cfg.sim_open_csv)

    # Basic sanity checks
    for p in [net_file, route_file, det_add]:
        if not p.exists():
            raise FileNotFoundError(f"Required SUMO file not found: {p}")

    sumocfg_path = ensure_sumocfg_matches_cfg(cfg)

    # Run SUMO from the osrm_dir so relative paths in the cfg are correct
    print("🏁 Running open (no-closure) district simulation...")
    run_command(
        ["sumo-gui" if use_gui else "sumo", "-c", sumocfg_path.name],
        cwd=Path(cfg.osrm_dir),
    )
    print("✅ SUMO finished, converting detector output...")

    convert_detector_xml_to_csv(det_xml, out_csv, expected_sensors=None)
    print("🎉 Open district simulation pipeline completed.")


if __name__ == "__main__":
    # change to False when you don't need to visually inspect anymore
    run_open_district_simulation(use_gui=True)
