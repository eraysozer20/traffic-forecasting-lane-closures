#!/usr/bin/env python3
"""
Run a no-closure SUMO simulation for District 8:

- Uses existing:
    d08_freeways.net.xml
    d08_detectors.add.xml

- Generates:
    d08_random.rou.xml          (random demand)
    d08_noclosure.sumocfg       (config)
    d08_detector_output.xml     (from detectors.add.xml)

Usage (Ubuntu or Windows):

    python3 run_d08_noclosure.py --mode gui
    python3 run_d08_noclosure.py --mode cli

Make sure SUMO_HOME is set before running, e.g.:

    export SUMO_HOME=/usr/share/sumo       # Ubuntu
    # or in PowerShell:
    # $env:SUMO_HOME = "C:\\Program Files (x86)\\Eclipse\\Sumo"
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess


# =======================
# 1. CONFIGURE PATHS HERE
# =======================

# Root folder where your files live
ROOT = Path("/mnt/c/Users/erays/Desktop/pemsdata/data/osrm")

NET_FILE = ROOT / "d08_freeways.net.xml"
DET_FILE = ROOT / "d08_detectors.add.xml"

ROUTES_FILE = ROOT / "d08_random.rou.xml"
SUMOCFG_FILE = ROOT / "d08_noclosure.sumocfg"

# Simulation time (seconds)
SIM_BEGIN = 0
SIM_END = 3600   # 1 hour

# RandomTrips options
TRIP_BEGIN = 0
TRIP_END = 3600
TRIP_PERIOD = 1.0    # avg 1 departure per second
TRIP_BINOMIAL = 2
TRIP_SEED = 42


# ==========================
# 2. HELPER / RUN FUNCTIONS
# ==========================

def require_sumo_home() -> Path:
    """Ensure SUMO_HOME is set and return its Path."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME is not set. Please set it before running this script:\n"
            "  Ubuntu:   export SUMO_HOME=/usr/share/sumo\n"
            "  Windows:  $env:SUMO_HOME = \"C:\\Program Files (x86)\\Eclipse\\Sumo\""
        )
    return Path(sumo_home)


def run_random_trips(force: bool = False):
    """Generate random routes with randomTrips.py if needed."""
    if ROUTES_FILE.exists() and not force:
        print(f"[SKIP] routes already exist: {ROUTES_FILE}")
        return

    sumo_home = require_sumo_home()
    random_trips = sumo_home / "tools" / "randomTrips.py"

    if not random_trips.exists():
        raise FileNotFoundError(f"randomTrips.py not found at {random_trips}")

    print(f"[ROUTES] Generating random routes → {ROUTES_FILE}")
    cmd = [
        sys.executable,  # use current python
        str(random_trips),
        "-n", str(NET_FILE),
        "-r", str(ROUTES_FILE),
        "-b", str(TRIP_BEGIN),
        "-e", str(TRIP_END),
        "-p", str(TRIP_PERIOD),
        "--binomial", str(TRIP_BINOMIAL),
        "--seed", str(TRIP_SEED),
    ]
    print("       ", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print("[ROUTES] Done.")


def write_sumocfg():
    """Write a simple no-closure SUMO config."""
    print(f"[CFG] Writing SUMO config → {SUMOCFG_FILE}")
    cfg_xml = f"""<configuration>
    <input>
        <net-file value="{NET_FILE.name}"/>
        <route-files value="{ROUTES_FILE.name}"/>
        <!-- detectors only, no closures yet -->
        <additional-files value="{DET_FILE.name}"/>
    </input>

    <time>
        <begin value="{SIM_BEGIN}"/>
        <end value="{SIM_END}"/>
    </time>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    SUMOCFG_FILE.write_text(cfg_xml, encoding="utf-8")
    print("[CFG] Done.")


def run_sumo(mode: str):
    """
    Run SUMO simulation.

    mode = 'gui' → sumo-gui
    mode = 'cli' → sumo (no GUI)
    """
    sumo_bin = "sumo-gui" if mode == "gui" else "sumo"

    print(f"[SUMO] Running {sumo_bin} with {SUMOCFG_FILE}")
    cmd = [sumo_bin, "-c", str(SUMOCFG_FILE)]
    print("       ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[SUMO] Finished simulation.")


# ===================
# 3. MAIN ENTRYPOINT
# ===================

def main():
    parser = argparse.ArgumentParser(description="Run D08 no-closure SUMO simulation.")
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="Run with SUMO GUI (gui) or command-line (cli). Default: gui",
    )
    parser.add_argument(
        "--force-routes",
        action="store_true",
        help="Regenerate routes even if d08_random.rou.xml already exists.",
    )
    args = parser.parse_args()

    # Sanity checks
    if not NET_FILE.exists():
        raise FileNotFoundError(f"Network file not found: {NET_FILE}")
    if not DET_FILE.exists():
        raise FileNotFoundError(f"Detectors file not found: {DET_FILE}")

    # Ensure we are in the same directory as the files (for relative paths in cfg)
    os.chdir(ROOT)
    print(f"[CD] Working directory: {ROOT}")

    # 1) Generate routes (if needed)
    run_random_trips(force=args.force_routes)

    # 2) Write SUMO config
    write_sumocfg()

    # 3) Run SUMO
    run_sumo(args.mode)


if __name__ == "__main__":
    main()
