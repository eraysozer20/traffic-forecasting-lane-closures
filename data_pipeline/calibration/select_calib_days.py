# calibration/select_calib_days.py
import pandas as pd
from datetime import timedelta
from config import CalibConfig   # or from .config import CalibConfig if using package style


def load_daily_flow_wide(cfg: CalibConfig) -> pd.DataFrame:
    """
    Read daily per-sensor aggregates: columns [date, sensor1, sensor2, ...].
    Assumes each row is one day.
    """
    df = pd.read_csv(cfg.daily_flow_wide_csv)
    df["day"] = pd.to_datetime(df["date"]).dt.date
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek  # Monday = 0
    return df


def build_daily_descriptors_from_daily(df: pd.DataFrame, cfg: CalibConfig) -> pd.DataFrame:
    """
    From daily per-sensor data, compute for each day:
      - total_flow (sum over alive sensors)
      - alive_sensor_ratio
      - n_alive_sensors, n_total_sensors
    We treat a sensor as 'alive that day' if its daily aggregate > 0.
    """
    sensor_cols = [c for c in df.columns if c not in ("date", "day", "dow")]

    # Drop globally dead sensors (all zeros across the whole period)
    global_sums = df[sensor_cols].sum(axis=0)
    alive_global_mask = global_sums > 0
    alive_global_sensors = [s for s in sensor_cols if alive_global_mask[s]]

    if not alive_global_sensors:
        raise ValueError("All sensors appear dead (sum == 0) in daily_flow_wide_csv.")

    print(f"[INFO] {len(alive_global_sensors)} / {len(sensor_cols)} sensors ever report non-zero daily flow.")

    sensor_cols = alive_global_sensors

    rows = []
    for _, row in df.iterrows():
        day = row["day"]
        dow = int(row["dow"])

        # daily per-sensor values for that day
        sums = row[sensor_cols].astype(float)
        alive_mask = sums > 0
        n_total = len(sensor_cols)
        n_alive = int(alive_mask.sum())
        alive_ratio = n_alive / n_total if n_total else 0.0

        # total network flow that day (only over alive sensors)
        total_flow = float(sums[alive_mask].sum()) if n_alive > 0 else 0.0

        rows.append(
            {
                "day": day,
                "dow": dow,
                "total_flow": total_flow,
                "alive_sensor_ratio": alive_ratio,
                "n_alive_sensors": n_alive,
                "n_total_sensors": n_total,
            }
        )

    desc_df = pd.DataFrame(rows)
    return desc_df


def load_daily_closures(cfg: CalibConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.lane_closures_daily_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def pick_candidate_open_days(desc_df: pd.DataFrame, closure_df: pd.DataFrame, cfg: CalibConfig) -> pd.DataFrame:
    merged = desc_df.merge(closure_df, left_on="day", right_on="date", how="left")

    merged["total_closed_minutes"] = merged["total_closed_minutes"].fillna(0.0)
    merged["num_closures"] = merged["num_closures"].fillna(0).astype(int)

    mask_open = merged["total_closed_minutes"] <= cfg.max_closed_minutes_for_open
    mask_sensor_good = merged["alive_sensor_ratio"] >= cfg.min_alive_sensor_ratio

    candidates = merged[mask_open & mask_sensor_good].copy()
    return candidates


def pick_representative_days(candidates: pd.DataFrame, cfg: CalibConfig) -> pd.DataFrame:
    """
    From candidate open days, select calibration days with:
      - preference for recency (last cfg.recent_window_days),
      - diversity across days-of-week and time when we have many candidates,
      - 'typical' total_flow (close to median).

    If the number of recent candidates is <= n_calib_days_target,
    we simply return ALL recent candidates (no extra diversity constraints).
    """
    if candidates.empty:
        return candidates.copy()

    # 1) restrict to recent window
    day_dt_all = pd.to_datetime(candidates["day"])
    max_day = day_dt_all.max()
    cutoff_day = max_day - pd.Timedelta(days=cfg.recent_window_days)

    recent_mask = day_dt_all >= cutoff_day
    recent_candidates = candidates[recent_mask].copy()

    if recent_candidates.empty:
        print(
            f"[WARN] No candidate days in the last {cfg.recent_window_days} days; "
            f"falling back to entire history."
        )
        base = candidates.copy()
    else:
        print(
            f"[INFO] Using {len(recent_candidates)} candidate days "
            f"from the last {cfg.recent_window_days} days."
        )
        base = recent_candidates

    # If we already have <= target days, just use all of them
    if len(base) <= cfg.n_calib_days_target:
        print(f"[INFO] {len(base)} <= target {cfg.n_calib_days_target}; "
              "using all these days as calibration days (no extra diversity filtering).")
        base_sorted = base.sort_values("day")
        return base_sorted.copy()

    # ----------------------------------------------------
    # From here on: many candidates -> apply diversity logic
    # ----------------------------------------------------
    base = base.copy()
    base["day_dt"] = pd.to_datetime(base["day"])

    # 2) 'typicalness': distance to median flow
    median_flow = base["total_flow"].median()
    base["dist_flow"] = (base["total_flow"] - median_flow).abs()

    # 3) sort by: typical first, then more recent, then lower closure minutes
    sort_cols = ["dist_flow", "day_dt", "total_closed_minutes"]
    sort_asc = [True, False, True]
    base = base.sort_values(sort_cols, ascending=sort_asc)

    # 4) greedy diversity selection
    max_per_dow = getattr(cfg, "max_per_dow", 3)
    min_gap_days = getattr(cfg, "min_gap_days", 14)

    selected_rows = []
    selected_dates = []
    dow_counts = {d: 0 for d in range(7)}

    for _, row in base.iterrows():
        if len(selected_rows) >= cfg.n_calib_days_target:
            break

        dow = int(row["dow"])
        day = row["day_dt"].date()

        # weekday quota
        if dow_counts[dow] >= max_per_dow:
            continue

        # temporal spread
        too_close = False
        for sday in selected_dates:
            if abs((day - sday).days) < min_gap_days:
                too_close = True
                break
        if too_close:
            continue

        selected_rows.append(row)
        selected_dates.append(day)
        dow_counts[dow] += 1

    # If strict diversity rules selected nothing (unlikely here), just take top-N
    if not selected_rows:
        print("[WARN] Diversity constraints too strict; falling back to top-N without gap constraint.")
        selected_rows = list(base.head(cfg.n_calib_days_target).itertuples(index=False))

        # Convert from namedtuple to Series-like dicts
        selected_rows = [pd.Series(r._asdict()) for r in selected_rows]

    selected_df = pd.DataFrame(selected_rows)
    selected_df = selected_df.sort_values("day_dt")
    selected_df = selected_df.drop(columns=["day_dt", "dist_flow"], errors="ignore")

    return selected_df




def main():
    cfg = CalibConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading daily flow data from {cfg.daily_flow_wide_csv}")
    daily_df = load_daily_flow_wide(cfg)

    print("[INFO] Computing daily descriptors + sensor health (from daily aggregates)...")
    desc_df = build_daily_descriptors_from_daily(daily_df, cfg)
    print(f"[INFO] Got descriptors for {len(desc_df)} days")

    print(f"[INFO] Loading daily closure summary from {cfg.lane_closures_daily_csv}")
    closure_df = load_daily_closures(cfg)

    print("[INFO] Filtering candidate open days (low closure + good sensor coverage)...")
    candidates = pick_candidate_open_days(desc_df, closure_df, cfg)
    print(f"[INFO] Candidate open days: {len(candidates)}")

    selected = pick_representative_days(candidates, cfg)
    print(f"[INFO] Selected {len(selected)} calibration days:")
    for d in selected["day"]:
        print(f"  - {d}")

    out_candidates = cfg.out_dir / "candidate_open_days.csv"
    out_selected   = cfg.out_dir / "calibration_days.csv"

    candidates.to_csv(out_candidates, index=False)
    selected.to_csv(out_selected, index=False)

    print(f"[OK] Wrote candidate days to {out_candidates}")
    print(f"[OK] Wrote final calibration days to {out_selected}")


if __name__ == "__main__":
    main()