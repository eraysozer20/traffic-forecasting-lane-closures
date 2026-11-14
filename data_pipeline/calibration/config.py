# calibration/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, adjust if needed


class CalibConfig:
    district = "08"

    # ---------- REAL DATA ----------
    real_flow_wide_csv = ROOT / "data" / "pems" / f"pems{district}" / f"pems{district}_all_common_flow.csv"
    hourly_flow_wide_csv = ROOT / "data" / "pems" / f"pems{district}" / f"pems{district}_h.csv"
    daily_flow_wide_csv = ROOT / "data" / "pems" / f"pems{district}" / f"pems{district}_d.csv"

    # ---------- LANE CLOSURES ----------
    lane_closures_raw_csv = (
        ROOT / "data" / "lane_closures" / f"pems{district}" / f"pems{district}_lane_closure_matches.csv"
    )

    lane_closures_daily_csv = (
        ROOT / "data" / "lane_closures" / f"pems{district}" / f"pems{district}_daily_summary.csv"
    )

    # ---------- CALIBRATION OUTPUTS ----------
    out_dir = ROOT / "calibration" / "outputs" / f"pems{district}"

    # ---------- THRESHOLDS ----------
    max_closed_minutes_for_open = 24 * 60.0
    min_alive_sensor_ratio = 0.6
    n_calib_days_target = 10
    recent_window_days = 365 * 7
    max_per_dow = 3
    min_gap_days = 14

    # ---------- SUMO / SIM SETTINGS ----------
    # Existing freeway network from OSRM/netconvert
    sumo_net_file = ROOT / "data" / "osrm" / f"d{district}_freeways.net.xml"

    # Where to put sim runs (per-scale subfolders)
    sim_base_dir = ROOT / "calibration" / "sim_runs" / f"pems{district}"

    # Simulation horizon: one day, 5-min measurements
    sim_duration_sec = 24 * 3600
    sensor_freq_sec = 300  # 5 min

    # Base demand: period (sec) for scale=1.0 (adjust later if needed)
    baseline_trip_period_sec = 2.0

    # SUMO binaries (assume in PATH; change if needed)
    sumo_bin = "sumo"
    sumo_gui_bin = "sumo-gui"
