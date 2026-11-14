# calibration/run_open_sim_scale.py
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, List

import pandas as pd

from calibration.config import CalibConfig


# =====================================================
# Utility: run a command with optional step label
# =====================================================

def run_cmd(cmd: list, cwd: Optional[Path] = None, step_info: str = ""):
    """
    Run a command with logging. Optionally include a human-readable step label.
    """
    if step_info:
        print(step_info)
    print("▶", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print("❌ ERROR while running command:")
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result


# =====================================================
# Detector XML → CSV conversion
# =====================================================

def convert_detector_xml_to_csv(
    xml_file: Path,
    output_csv: Path,
    expected_sensors: Optional[List[str]] = None,
):
    """
    Convert e1Detector output from SUMO into a wide CSV:
      rows: sensor_id
      cols: minute (0, 5, 10, ...) as strings
      vals: nVehEntered (int)
    """
    if not xml_file.exists():
        raise FileNotFoundError(f"Detector XML not found: {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []

    for interval in root.findall("interval"):
        begin_sec = float(interval.attrib.get("begin", "0"))
        sensor_id = interval.attrib.get("id", "")
        count = int(float(interval.attrib.get("nVehEntered", 0)))
        minute = int(begin_sec // 60)
        records.append(
            {"minute": minute, "sensor_id": sensor_id, "vehicle_count": count}
        )

    if not records:
        print(f"⚠️ No intervals found in {xml_file}, writing empty CSV.")
        df = pd.DataFrame()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv)
        return

    df = (
        pd.DataFrame(records)
        .pivot(index="sensor_id", columns="minute", values="vehicle_count")
        .fillna(0)
        .astype(int)
    )

    # Optional: pad missing sensors
    if expected_sensors is not None and len(df.columns) > 0:
        missing = [s for s in expected_sensors if s not in df.index]
        for s in missing:
            df.loc[s] = 0
        df = df.sort_index()

    df.columns = [str(c) for c in df.columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    print(f"✅ Sensor data saved to CSV: {output_csv}")


# =====================================================
# Build detectors on all non-internal lanes
# =====================================================

def add_detectors_for_all_non_internal_lanes(
    net_file: Path,
    detector_additional_file: Path,
    detector_output_file: Path,
    freq_sec: int,
):
    """
    Add an e1Detector to every non-internal lane.
    Detector ID: det_<lane_id> (e.g. det_801230.0)
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    additional = ET.Element("additional")

    for edge in root.findall("edge"):
        eid = edge.attrib.get("id", "")
        if eid.startswith(":"):
            continue  # skip internal edges

        for lane in edge.findall("lane"):
            lane_id = lane.attrib["id"]
            length = float(lane.attrib.get("length", "0"))
            pos = length / 2.0
            ET.SubElement(
                additional,
                "e1Detector",
                {
                    "id": f"det_{lane_id}",
                    "lane": lane_id,
                    "pos": str(pos),
                    "freq": str(freq_sec),
                    "file": detector_output_file.name,
                },
            )

    detector_additional_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(additional).write(
        detector_additional_file,
        encoding="utf-8",
        xml_declaration=True,
    )
    print(f"✅ Detectors written to: {detector_additional_file}")


# =====================================================
# Write SUMO config (.sumocfg)
# =====================================================

def write_sumocfg(net_file: Path, route_file: Path, additional_file: Path, cfg_file: Path):
    text = f"""<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="{route_file.name}"/>
        <additional-files value="{additional_file.name}"/>
    </input>
</configuration>
"""
    cfg_file.write_text(text, encoding="utf-8")
    print(f"✅ SUMO config written: {cfg_file}")


# =====================================================
# Find randomTrips.py via SUMO_HOME
# =====================================================

def get_random_trips_script() -> Path:
    """
    Locate randomTrips.py using SUMO_HOME/tools.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set. Please set it to your SUMO installation root.")
    tools = Path(sumo_home) / "tools"
    rt = tools / "randomTrips.py"
    if not rt.exists():
        raise FileNotFoundError(f"randomTrips.py not found at {rt}")
    return rt


# =====================================================
# TraCI helper: import traci from SUMO tools
# =====================================================

def _ensure_traci_on_path():
    """
    Add SUMO_HOME/tools to sys.path so we can import traci.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set (needed for TraCI).")
    tools = Path(sumo_home) / "tools"
    if not tools.exists():
        raise FileNotFoundError(f"SUMO tools dir not found at {tools}")
    if str(tools) not in sys.path:
        sys.path.append(str(tools))

    import traci  # type: ignore
    return traci


# =====================================================
# Run SUMO with progress logging via TraCI
# =====================================================

def run_sumo_with_progress(sumocfg_file: Path, total_sim_sec: int, use_gui: bool, cwd: Path):
    """
    Run SUMO through TraCI and log how many simulated minutes have passed/left.
    """
    traci = _ensure_traci_on_path()

    sumo_bin = CalibConfig.sumo_gui_bin if use_gui else CalibConfig.sumo_bin
    port = 8873  # arbitrary port

    cmd = [
        sumo_bin,
        "-c", sumocfg_file.name,
        "--remote-port", str(port),
        "--step-length", "1.0",   # 1 sec per simulation step
        "--start",
    ]
    print("▶", " ".join(cmd))
    print("[4/5] Starting SUMO with TraCI for progress logging...")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        traci.init(port)
        sim_step = 0
        log_interval_sec = 300  # log every 5 simulated minutes

        total_min = total_sim_sec / 60.0
        print(f"    → Simulation horizon: {total_min:.1f} minutes")

        while sim_step < total_sim_sec:
            traci.simulationStep()
            sim_step += 1

            if sim_step % log_interval_sec == 0 or sim_step == total_sim_sec:
                done_min = sim_step / 60.0
                left_min = total_min - done_min
                pct = 100.0 * sim_step / total_sim_sec
                print(
                    f"    ⏱ sim time: {done_min:6.1f} / {total_min:.1f} min  "
                    f"({pct:5.1f}% done, ~{left_min:6.1f} min left)"
                )

        print("✅ SUMO simulation finished (100% of horizon).")

    finally:
        try:
            traci.close(False)
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()


# =====================================================
# Main: run open sim for a given demand scale
# =====================================================

def run_open_sim_for_scale(
    demand_scale: float,
    use_gui: bool = False,
    reuse_inputs: bool = False,
) -> Path:
    """
    Run a single open SUMO simulation for given demand_scale.
    If reuse_inputs=True, existing trips/routes/detectors/config are reused when present.

    Returns:
        Path to sensor_output.csv
    """
    cfg = CalibConfig

    # ---------- STEP 0: run directory ----------
    scale_tag = f"{demand_scale:.2f}".replace(".", "p")
    run_dir = cfg.sim_base_dir / f"open_scale_{scale_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"🗂  Run dir: {run_dir}")

    # Paths inside run_dir
    net_src = Path(cfg.sumo_net_file)
    if not net_src.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_src}")

    net_file      = run_dir / net_src.name
    trips_file    = run_dir / "trips.rou.xml"
    routed_file   = run_dir / "routes.rou.xml"
    detectors_add = run_dir / "detectors.add.xml"
    detector_xml  = run_dir / "detector_output.xml"
    sumocfg_file  = run_dir / "sumo.sumocfg"
    sensor_csv    = run_dir / "sensor_output.csv"

    # Copy network into run dir if needed
    if not net_file.exists():
        shutil.copyfile(net_src, net_file)
        print(f"📄 Copied net: {net_src} → {net_file}")
    else:
        print(f"ℹ️ Net already present: {net_file}")

    total_sim_sec = int(cfg.sim_duration_sec)
    total_sim_hours = total_sim_sec / 3600.0
    print(f"ℹ️ Simulation horizon: {total_sim_sec} sec (~{total_sim_hours:.1f} hours)")

    # ---------- STEP 1/5: Trips ----------
    if reuse_inputs and trips_file.exists():
        print(f"[1/5] Trips already exist, reusing: {trips_file}")
    else:
        print(f"[1/5] Generating trips (demand_scale={demand_scale:.3f})...")
        random_trips_py = get_random_trips_script()
        base_period = float(cfg.baseline_trip_period_sec)
        effective_period = base_period / demand_scale

        run_cmd(
            [
                sys.executable,
                str(random_trips_py),
                "-n", net_file.name,
                "-r", trips_file.name,
                "--seed", "42",
                "--begin", "0",
                "--end", str(total_sim_sec),
                "--period", f"{effective_period}",
                "--binomial", "1",
            ],
            cwd=run_dir,
        )
        print("✅ Trips generated.")

    # ---------- STEP 2/5: Routes ----------
    if reuse_inputs and routed_file.exists():
        print(f"[2/5] Routes already exist, reusing: {routed_file}")
    else:
        print("[2/5] Running duarouter (routing trips)...")
        run_cmd(
            [
                "duarouter",
                "--net-file", net_file.name,
                "--route-files", trips_file.name,
                "--output-file", routed_file.name,
                "--ignore-errors",
            ],
            cwd=run_dir,
        )
        print("✅ Routes generated.")

    # ---------- STEP 3/5: Detectors + SUMO config ----------
    if reuse_inputs and detectors_add.exists() and sumocfg_file.exists():
        print("[3/5] Detector + SUMO config already exist, reusing.")
    else:
        print("[3/5] Creating detectors and SUMO config...")
        add_detectors_for_all_non_internal_lanes(
            net_file=net_file,
            detector_additional_file=detectors_add,
            detector_output_file=detector_xml,
            freq_sec=int(cfg.sensor_freq_sec),
        )
        write_sumocfg(net_file, routed_file, detectors_add, sumocfg_file)
        print("✅ Detectors + SUMO config ready.")

    # ---------- STEP 4/5: Run SUMO with progress ----------
    print("[4/5] Running SUMO simulation with TraCI progress logging...")
    print(f"    → This is a full-day run (~{total_sim_hours:.1f} hours of sim time).")
    run_sumo_with_progress(
        sumocfg_file=sumocfg_file,
        total_sim_sec=total_sim_sec,
        use_gui=use_gui,
        cwd=run_dir,
    )

    # ---------- STEP 5/5: Convert XML → CSV ----------
    print("[5/5] Converting detector XML → CSV...")
    convert_detector_xml_to_csv(detector_xml, sensor_csv)
    print(f"🎯 Pipeline finished for demand_scale = {demand_scale:.3f}")

    return sensor_csv


# =====================================================
# CLI
# =====================================================

def main():
    ap = argparse.ArgumentParser(description="Run open SUMO simulation for a given demand scale.")
    ap.add_argument(
        "--demand-scale",
        type=float,
        required=True,
        help="Demand scaling factor α (1.0 = baseline, >1 more demand, <1 less).",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Use sumo-gui instead of sumo.",
    )
    ap.add_argument(
        "--reuse-inputs",
        action="store_true",
        help="If set, reuse existing trips/routes/detectors/config if they exist.",
    )
    args = ap.parse_args()

    run_open_sim_for_scale(
        demand_scale=args.demand_scale,
        use_gui=args.gui,
        reuse_inputs=args.reuse_inputs,
    )


if __name__ == "__main__":
    main()
