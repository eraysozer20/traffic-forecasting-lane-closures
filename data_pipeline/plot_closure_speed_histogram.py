#!/usr/bin/env python3
"""
Histograms for closure × station speed-impact analysis:

  1) drop_ratio      = drop_from_before / before_speed
  2) recovery_ratio  = recovery_after   / after_speed
  3) start date histogram (when closures start)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------
# CONFIG – Change district via env var PEMS_DISTRICT_ID
# -------------------------------------------------------
DISTRICT = int(os.getenv("PEMS_DISTRICT_ID", "12"))

def district_code(d: int) -> str:
    return f"{int(d):02d}"   # 3→"03", 10→"10"

DISTRICT_STR = district_code(DISTRICT)

ROOT = Path(__file__).resolve().parent / "data" / "lane_closures" / f"pems{DISTRICT_STR}"
IN_FILE = ROOT / f"pems{DISTRICT_STR}_closure_speed_impacts.csv"

OUT_FILE_DROP    = ROOT / f"pems{DISTRICT_STR}_hist_drop_ratio.png"
OUT_FILE_RECOVER = ROOT / f"pems{DISTRICT_STR}_hist_recovery_ratio.png"
OUT_FILE_START   = ROOT / f"pems{DISTRICT_STR}_hist_start_dates.png"

# -------------------------------------------------------
# LOAD
# -------------------------------------------------------
print(f"[LOAD] {IN_FILE}")
df = pd.read_csv(IN_FILE, low_memory=False)

# Try to parse Start Date if present
if "Start Date" in df.columns:
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

# -------------------------------------------------------
# HELPER TO MAKE A RATIO HISTOGRAM
# -------------------------------------------------------
def make_ratio_hist(
    df: pd.DataFrame,
    num_col: str,
    den_col: str,
    out_path: Path,
    title_prefix: str,
    ratio_name: str,
    clip_min: float = -1.0,
    clip_max: float = 5.0,
) -> None:
    """
    Build a histogram of (num_col / den_col) with some basic cleaning:
      - drop rows where numerator or denominator is NaN
      - require denominator > 0
      - optional sanity clipping
    """
    if num_col not in df.columns or den_col not in df.columns:
        print(f"[SKIP] Columns {num_col!r} or {den_col!r} not found; skipping {ratio_name}.")
        return

    sub = df.dropna(subset=[num_col, den_col]).copy()
    before = len(sub)
    sub = sub[sub[den_col] > 0]
    after = len(sub)
    print(f"[FILTER {ratio_name}] drop NaN & {den_col}>0: {before} → {after} rows")

    if sub.empty:
        print(f"[SKIP] No data left for {ratio_name} after filtering.")
        return

    ratio_col = f"{ratio_name}_ratio"
    sub[ratio_col] = sub[num_col] / sub[den_col]

    # sanity-filter insane values
    sub = sub[(sub[ratio_col] > clip_min) & (sub[ratio_col] < clip_max)]
    print(f"[FILTER {ratio_name}] clip to ({clip_min}, {clip_max}): {after} → {len(sub)} rows")

    if sub.empty:
        print(f"[SKIP] No data left for {ratio_name} after clipping.")
        return

    # ---------------------------------------------------
    # PLOT
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(sub[ratio_col], bins=50, edgecolor="black")

    plt.title(f"District {DISTRICT_STR} — {title_prefix}")
    plt.xlabel(ratio_name.replace("_", " ") + " (ratio)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[DONE] Saved {ratio_name} histogram to:\n{out_path}")


def plot_start_date_hist(df: pd.DataFrame, out_path: Path) -> None:
    """
    Histogram of closure start dates from the speed-impacts file.
    Assumes 'Start Date' column exists and has been parsed to datetime.
    """
    if "Start Date" not in df.columns:
        print("[SKIP] 'Start Date' column not found; cannot plot start date histogram.")
        return

    sub = df.dropna(subset=["Start Date"]).copy()
    if sub.empty:
        print("[SKIP] No valid Start Date values to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(sub["Start Date"], bins=50)
    plt.title(f"District {DISTRICT_STR} — Histogram of Closure Start Dates")
    plt.xlabel("Start Date")
    plt.ylabel("Number of closure × station rows")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[DONE] Saved start date histogram to:\n{out_path}")


# -------------------------------------------------------
# RUN ALL HISTOGRAMS
# -------------------------------------------------------

# 1) drop_from_before / before_speed
make_ratio_hist(
    df=df,
    num_col="drop_from_before",
    den_col="before_speed",
    out_path=OUT_FILE_DROP,
    title_prefix="Speed Drop Ratio Histogram\n(drop_from_before / before_speed)",
    ratio_name="drop_from_before",
)

# 2) recovery_after / after_speed
make_ratio_hist(
    df=df,
    num_col="recovery_after",
    den_col="after_speed",
    out_path=OUT_FILE_RECOVER,
    title_prefix="Speed Recovery Ratio Histogram\n(recovery_after / after_speed)",
    ratio_name="recovery_after",
)

# 3) start date histogram
plot_start_date_hist(df, OUT_FILE_START)

print("[ALL DONE]")
