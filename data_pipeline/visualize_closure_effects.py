#!/usr/bin/env python3
"""
Visualize speed impact vs adjacency-based distance to closed lanes.

Uses:
  - closure stats CSV with columns at least:
        closure_id, sensor_id, rel_change
  - adjacency matrix CSV:
        data/dist/pems08/adj_mx_d08.csv
  - speed.h5 for sensor ordering:
        data/despt_inputs/pems08/speed.h5
  - lane_closure_matches CSV:
        data/lane_closures/pems08/pems08_lane_closure_matches.csv

For each closure_id:
  1) Take all matched_station_ids from lane_closure_matches rows
     for that closure_id → set of "closed-lane sensors".
  2) Map them to indices in the adjacency matrix.
  3) For every sensor in the graph:
        sim_i  = max_j adj[i, j] over closed-lane sensors j
        dist_i = 1 - sim_i
  4) Attach dist_i as adj_dist_to_closure for all stats rows of that sensor.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG — update paths / filenames if needed
# ======================================================

DISTRICT = 8
DISTRICT_CODE = f"{DISTRICT:02d}"

ROOT = Path(__file__).resolve().parent / "data"

# your stats file (the one with closure_id,sensor_id,rel_change)
STATS_CSV = ROOT / "analysis" / f"pems{DISTRICT_CODE}_closure_speed_comparison_selected.csv"

# adjacency (diag=1 version)
ADJ_CSV = ROOT / "dist" / f"pems{DISTRICT_CODE}" / f"adj_mx_d{DISTRICT_CODE}.csv"

# speed.h5 for sensor order
PEMS_H5 = ROOT / "despt_inputs" / f"pems{DISTRICT_CODE}" / "speed.h5"

# lane-closure matches
MATCHES_CSV = (
    ROOT
    / "lane_closures"
    / f"pems{DISTRICT_CODE}"
    / f"pems{DISTRICT_CODE}_lane_closure_matches.csv"
)

# outputs
OUT_ENRICHED = (
    ROOT / "analysis" / f"pems{DISTRICT_CODE}" / "closure_effects_d08_with_distance.csv"
)
OUT_PLOT = (
    ROOT / "analysis" / f"pems{DISTRICT_CODE}" / "closure_effects_vs_distance_d08.png"
)


# ======================================================
# LOAD DATA
# ======================================================

print(f"[LOAD] stats:   {STATS_CSV}")
stats_df = pd.read_csv(STATS_CSV)

required_cols = {"closure_id", "sensor_id", "rel_change"}
missing = required_cols - set(stats_df.columns)
if missing:
    raise KeyError(f"Stats CSV is missing columns: {missing}")

# Ensure types
stats_df["sensor_id"] = stats_df["sensor_id"].astype(str)

print(f"[LOAD] speed.h5 for sensor order: {PEMS_H5}")
speed_df = pd.read_hdf(PEMS_H5)
sensor_ids = speed_df.columns.astype(str).tolist()
sensor_id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
num_sensors = len(sensor_ids)
print(f"  sensors in graph (from speed.h5): {num_sensors}")

print(f"[LOAD] adjacency CSV: {ADJ_CSV}")
adj = np.loadtxt(ADJ_CSV, delimiter=",").astype(float)
if adj.shape != (num_sensors, num_sensors):
    raise ValueError(
        f"Adjacency shape {adj.shape} does not match {num_sensors} sensors "
        f"(from speed.h5)."
    )

print(f"[LOAD] lane-closure matches: {MATCHES_CSV}")
matches_df = pd.read_csv(MATCHES_CSV, low_memory=False)

if "ID" not in matches_df.columns:
    raise KeyError("Expected 'ID' column in lane_closure_matches CSV")
if "matched_station_ids" not in matches_df.columns:
    raise KeyError("Expected 'matched_station_ids' column in lane_closure_matches CSV")

# Restrict matches to the closures that appear in stats
closure_ids_in_stats = stats_df["closure_id"].unique().tolist()
matches_df = matches_df[matches_df["ID"].isin(closure_ids_in_stats)].copy()

print(f"  closures in stats:   {len(closure_ids_in_stats)}")
print(f"  matching rows in LC: {len(matches_df)}")


# ======================================================
# BUILD CLOSED-LANE SENSOR SET PER CLOSURE
# ======================================================

closure_to_closed_sensors = {}

for cid in closure_ids_in_stats:
    rows = matches_df[matches_df["ID"] == cid]
    closed_set = set()

    for _, r in rows.iterrows():
        st_str = r.get("matched_station_ids", "")
        if pd.isna(st_str) or not str(st_str).strip():
            continue
        parts = [x.strip() for x in str(st_str).split(";") if x.strip()]
        closed_set.update(parts)

    closure_to_closed_sensors[cid] = sorted(closed_set)

# Log a little
for cid, sensors in closure_to_closed_sensors.items():
    print(f"[CLOSED SENSORS] {cid}: {len(sensors)} sensors from matches")


# ======================================================
# COMPUTE DISTANCE-LIKE MEASURE PER CLOSURE + SENSOR
# ======================================================

# Prepare container for distances: same rows as stats_df
stats_df["adj_dist_to_closure"] = np.nan

for cid in closure_ids_in_stats:
    closed_ids = closure_to_closed_sensors.get(cid, [])
    if not closed_ids:
        print(f"[WARN] Closure {cid} has no matched_station_ids. Skipping.")
        continue

    # Map to sensor indices that exist in the graph
    closed_idx = [
        sensor_id_to_idx[sid]
        for sid in closed_ids
        if sid in sensor_id_to_idx
    ]

    if not closed_idx:
        print(
            f"[WARN] Closure {cid}: none of its {len(closed_ids)} sensors "
            f"are in the graph sensor list. Skipping."
        )
        continue

    closed_idx = np.array(closed_idx, dtype=int)

    # For every sensor i: sim_i = max_j adj[i, j] over j in closed_idx
    # Then distance-like measure: 1 - sim_i
    # (Shape: [num_sensors, len(closed_idx)] -> [num_sensors])
    sims = adj[:, closed_idx]              # (N, K)
    max_sim = sims.max(axis=1)             # (N,)
    dist_like = 1.0 - max_sim              # (N,)

    # Now assign to all rows in stats_df for this closure
    mask = stats_df["closure_id"] == cid
    for idx, row in stats_df[mask].iterrows():
        sid = row["sensor_id"]
        if sid in sensor_id_to_idx:
            i = sensor_id_to_idx[sid]
            stats_df.at[idx, "adj_dist_to_closure"] = dist_like[i]
        else:
            # sensor in stats but not in graph; leave NaN
            pass

# Drop rows without a distance (if any)
before = len(stats_df)
stats_df = stats_df.dropna(subset=["adj_dist_to_closure"])
after = len(stats_df)
print(f"[INFO] Dropped {before - after} rows without adjacency distance.")


# ======================================================
# SAVE ENRICHED CSV
# ======================================================

OUT_ENRICHED.parent.mkdir(parents=True, exist_ok=True)
stats_df.to_csv(OUT_ENRICHED, index=False)
print(f"[SAVE] Enriched stats with adj_dist_to_closure: {OUT_ENRICHED}")


# ======================================================
# PLOT
# ======================================================

plt.figure(figsize=(10, 7))

# Color per closure_id
closure_ids = sorted(stats_df["closure_id"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, max(len(closure_ids), 1)))

for color, cid in zip(colors, closure_ids):
    sub = stats_df[stats_df["closure_id"] == cid]
    plt.scatter(
        sub["adj_dist_to_closure"],
        sub["rel_change"],
        s=20,
        alpha=0.7,
        label=cid,
        color=color,
    )

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Adjacency-based distance to closed lanes (0 = on/very close, 1 ≈ far)")
plt.ylabel("Relative speed change during closure")
plt.title(f"Speed impact vs proximity to closed lanes (District {DISTRICT_CODE})")
plt.legend(title="closure_id", fontsize=8)
plt.tight_layout()

OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PLOT, dpi=200)
print(f"[SAVE] Plot: {OUT_PLOT}")
plt.close()
