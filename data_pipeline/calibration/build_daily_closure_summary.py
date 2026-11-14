# calibration/build_daily_closure_summary.py
import pandas as pd
from pathlib import Path
from config import CalibConfig

STATUS_APPROVED_VALUES = ["Approved", "approved", "APPROVED"]  # adjust if needed


def parse_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def compute_daily_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each closure row, distribute its duration across the calendar days it spans.
    Returns DataFrame[date, total_closed_minutes, num_closures].
    """
    records = []

    for _, row in df.iterrows():
        start = parse_datetime(row["Start Date"])
        end   = parse_datetime(row["End Date"])

        if pd.isna(start):
            continue

        if pd.isna(end):
            dur = row.get("Duration", None)
            if pd.notna(dur):
                try:
                    minutes = float(dur)
                    if minutes <= 0:
                        continue
                    end = start + pd.Timedelta(minutes=minutes)
                except Exception:
                    continue
            else:
                continue

        if end <= start:
            continue

        day = start.normalize()
        while day < end.normalize() + pd.Timedelta(days=1):
            day_start = max(start, day)
            day_end   = min(end, day + pd.Timedelta(days=1))
            minutes   = (day_end - day_start).total_seconds() / 60.0
            if minutes > 0:
                records.append(
                    {
                        "date": day.date(),
                        "closed_minutes": minutes,
                    }
                )
            day += pd.Timedelta(days=1)

    if not records:
        return pd.DataFrame(columns=["date", "total_closed_minutes", "num_closures"])

    per_day = pd.DataFrame(records)
    summary = (
        per_day.groupby("date")
        .agg(
            total_closed_minutes=("closed_minutes", "sum"),
            num_closures=("closed_minutes", "size"),
        )
        .reset_index()
    )
    return summary


def main():
    cfg = CalibConfig()

    print(f"[INFO] Reading lane closures from: {cfg.lane_closures_raw_csv}")
    df = pd.read_csv(cfg.lane_closures_raw_csv)

    # Filter to approved (or at least non-rejected) statuses
    if "Status" in df.columns:
        mask_approved = df["Status"].isin(STATUS_APPROVED_VALUES)
        df = df[mask_approved].copy()
        print(f"[INFO] After Status filter: {len(df)} rows")

    daily = compute_daily_minutes(df)
    print(f"[INFO] Daily closure rows: {len(daily)}")

    out_path = cfg.lane_closures_daily_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"[OK] Wrote daily closure summary to: {out_path}")


if __name__ == "__main__":
    main()
