# calibration/calibrate_open_sim.py
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import CalibConfig   # run: python calibration/calibrate_open_sim.py (from repo root)


# ---------- Helpers ----------

def parse_station_id_from_detector(det_id: str) -> str:
    """
    Example: 'det_801230.0' -> '801230'
             'det_806617'   -> '806617'
    We just take the first continuous digit block.
    """
    m = re.search(r"(\d+)", det_id)
    if not m:
        return det_id  # fallback: leave as-is
    return m.group(1)


# ---------- Load SIM (open) ----------

def load_sim_baseline_station_minute(cfg: CalibConfig) -> pd.DataFrame:
    """
    Load SUMO baseline open sim (detector x minute) and map
    detector IDs -> station IDs via pattern matching.

    Returns DataFrame with:
      index: station_id (string)
      columns: minutes (0,5,10,...) as strings
    """
    sim_path = Path(cfg.sim_open_csv)
    if not sim_path.exists():
        raise FileNotFoundError(f"Sim open CSV not found: {sim_path}")

    df_det = pd.read_csv(sim_path, index_col=0)  # index = detector_id like det_801230.0
    df_det.index = df_det.index.astype(str)

    # map each detector to a station id using digits in the name
    station_ids = [parse_station_id_from_detector(i) for i in df_det.index]
    df_det["station_id"] = station_ids

    # aggregate detectors that map to the same station id
    value_cols = [c for c in df_det.columns if c != "station_id"]
    df_sta = (
        df_det
        .reset_index(drop=True)
        .groupby("station_id")[value_cols]
        .sum()
    )
    df_sta.index = df_sta.index.astype(str)
    df_sta.index.name = "station_id"

    # ensure columns are strings (minutes)
    df_sta.columns = [str(c) for c in df_sta.columns]

    print(f"[INFO] SIM baseline: {df_sta.shape[0]} stations x {df_sta.shape[1]} minutes")
    return df_sta


# ---------- Load REAL (typical open day) ----------

def load_real_typical_open_day(cfg: CalibConfig) -> pd.DataFrame:
    """
    From the full 5-min PeMS matrix and selected calibration days,
    build a station x minute matrix:
      R[s, t] = average flow over calibration days on minute-of-day t.
    """
    real_path = Path(cfg.real_flow_wide_csv)
    if not real_path.exists():
        raise FileNotFoundError(f"Real flow CSV not found: {real_path}")

    df = pd.read_csv(real_path)
    if "date" not in df.columns:
        raise RuntimeError("Expected a 'date' column in real flow CSV.")

    df["date"] = pd.to_datetime(df["date"])

    # calibration days file from earlier step
    calib_days_path = cfg.out_dir / "calibration_days.csv"
    if not calib_days_path.exists():
        raise FileNotFoundError(f"Calibration days file not found: {calib_days_path}")

    calib_days = pd.read_csv(calib_days_path)
    calib_days["day"] = pd.to_datetime(calib_days["day"]).dt.date

    mask = df["date"].dt.date.isin(calib_days["day"])
    df_calib = df[mask].copy()
    if df_calib.empty:
        raise RuntimeError("No rows in real data match calibration days.")

    # minute-of-day (0..1435)
    df_calib["minute"] = df_calib["date"].dt.hour * 60 + df_calib["date"].dt.minute

    value_cols = [c for c in df_calib.columns if c not in ("date", "minute")]

    # long format: one row per (day, minute, station)
    df_long = df_calib.melt(
        id_vars=["date", "minute"],
        value_vars=value_cols,
        var_name="station_id",
        value_name="flow",
    )
    df_long["station_id"] = df_long["station_id"].astype(str)

    # average over days: group by (station_id, minute)
    df_typ = (
        df_long
        .groupby(["station_id", "minute"])["flow"]
        .mean()
        .reset_index()
    )

    df_wide = df_typ.pivot(index="station_id", columns="minute", values="flow")
    df_wide = df_wide.fillna(0.0)
    df_wide.index = df_wide.index.astype(str)
    df_wide.index.name = "station_id"
    df_wide.columns = [str(c) for c in df_wide.columns]

    print(f"[INFO] REAL typical open day: {df_wide.shape[0]} stations x {df_wide.shape[1]} minutes")
    return df_wide


# ---------- Alignment ----------

def align_real_and_sim(
    real_sta_min: pd.DataFrame,
    sim_sta_min: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align real and sim matrices on common station IDs and minutes.
    Returns (R_aligned, S_aligned) with same shape.
    """
    common_stations = sorted(set(real_sta_min.index) & set(sim_sta_min.index))

    if not common_stations:
        print("[WARN] No overlapping stations between real and sim IDs.")
        print("       Falling back to network-level totals only.")
        R_tot = real_sta_min.sum(axis=0).to_frame(name="total").T
        S_tot = sim_sta_min.sum(axis=0).to_frame(name="total").T
        # align on minutes
        common_minutes = sorted(set(R_tot.columns) & set(S_tot.columns))
        R = R_tot[common_minutes]
        S = S_tot[common_minutes]
        print(f"[INFO] Alignment (network-level): 1 series, {len(common_minutes)} minutes")
        return R, S

    common_minutes = sorted(set(real_sta_min.columns) & set(sim_sta_min.columns))

    R = real_sta_min.loc[common_stations, common_minutes].astype(float)
    S = sim_sta_min.loc[common_stations, common_minutes].astype(float)

    print(f"[INFO] Alignment: {len(common_stations)} stations, {len(common_minutes)} minutes")
    return R, S


# ---------- Metrics ----------

def compute_mae_metrics(R: pd.DataFrame, S_scaled: pd.DataFrame) -> dict:
    """
    MAE metrics between real R and scaled sim S_scaled.
    """
    assert R.shape == S_scaled.shape

    diff = (R - S_scaled).abs()
    mae_all = diff.to_numpy().mean()

    per_station_mae = diff.mean(axis=1)
    mae_station_mean = per_station_mae.mean()
    mae_station_median = per_station_mae.median()

    R_tot = R.sum(axis=0)
    S_tot = S_scaled.sum(axis=0)
    mae_network = (R_tot - S_tot).abs().mean()

    return {
        "mae_all": mae_all,
        "mae_station_mean": mae_station_mean,
        "mae_station_median": mae_station_median,
        "mae_network": mae_network,
    }


# ---------- Main calibration ----------

def run_open_calibration():
    cfg = CalibConfig()

    print("🔹 Loading SIM baseline (station x minute) ...")
    sim_sta_min = load_sim_baseline_station_minute(cfg)

    print("🔹 Loading REAL typical open day (station x minute) ...")
    real_sta_min = load_real_typical_open_day(cfg)

    print("🔹 Aligning real and sim matrices ...")
    R, S = align_real_and_sim(real_sta_min, sim_sta_min)

    scales = getattr(cfg, "demand_scales", [0.5, 0.75, 1.0, 1.25, 1.5])
    rows = []

    for alpha in scales:
        S_scaled = S * float(alpha)
        metrics = compute_mae_metrics(R, S_scaled)
        metrics["alpha"] = alpha
        rows.append(metrics)
        print(
            f"  α={alpha}: "
            f"mae_all={metrics['mae_all']:.3f}, "
            f"mae_station_mean={metrics['mae_station_mean']:.3f}, "
            f"mae_network={metrics['mae_network']:.3f}"
        )

    df_res = (
        pd.DataFrame(rows)
        .set_index("alpha")
        .sort_values("mae_all")
    )
    out_metrics = Path(cfg.open_calibration_results_csv)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_metrics)
    print(f"✅ Saved calibration results to: {out_metrics}")

    best_alpha = df_res.index[0]
    print(f"⭐ Best alpha (by mae_all) = {best_alpha}")

    # Save calibrated S for later use
    S_cal = S * float(best_alpha)
    out_cal = Path(cfg.sim_open_calibrated_csv)
    S_cal.to_csv(out_cal)
    print(f"✅ Saved calibrated SIM matrix to: {out_cal}")
    print("   shape =", S_cal.shape)


if __name__ == "__main__":
    run_open_calibration()
