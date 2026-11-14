#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx



# ======================================================
# CONFIG
# ======================================================

ROOT = Path("data")
FINAL_DIR = ROOT / "final"
META_ROOT = ROOT / "meta"

FIGURES_DIR = Path("..") / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OVERVIEW_OUT = FIGURES_DIR / "dataset_overview.csv"

TYSON_DIR = FINAL_DIR / "Tyson"  # change if needed

TYSON_FILES = {
    "speed": TYSON_DIR / "new_speed_5 - order.csv",
    "closure": TYSON_DIR / "new_construction_5 - order.csv",
    "adj": TYSON_DIR / "adj.csv",
    "adj1_miles": TYSON_DIR / "adj_1.csv",
}
# ======================================================
# HELPERS
# ======================================================

def district_code_from_folder(name: str) -> str:
    """
    pems03 -> 03
    pems03sub -> 03
    """
    m = re.match(r"pems(\d{2})(sub)?$", name)
    if not m:
        raise ValueError(f"Invalid folder name: {name}")
    return m.group(1)


def choose_dataset_folders(final_dir: Path):
    """
    If pemsXXsub exists, use it instead of pemsXX.
    """
    folders = [p for p in final_dir.iterdir() if p.is_dir() and re.match(r"pems\d{2}(sub)?$", p.name)]

    by_district = {}

    for folder in folders:
        code = district_code_from_folder(folder.name)
        is_sub = folder.name.endswith("sub")

        if code not in by_district:
            by_district[code] = folder
        else:
            current_is_sub = by_district[code].name.endswith("sub")
            if is_sub and not current_is_sub:
                by_district[code] = folder

    return [by_district[k] for k in sorted(by_district.keys())]


def read_time_sensor_csv(path: Path):
    df = pd.read_csv(path)
    time_col = df.columns[0]

    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")

    df.columns = df.columns.astype(str).str.strip()
    return df


def read_square_csv(path: Path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_latlon_file(code: str):
    meta_dir = META_ROOT / f"meta{code}"

    candidates = [
        meta_dir / f"station_latlon_meta{code}.csv",
        meta_dir / "station_latlon_meta.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(meta_dir.glob("station_latlon*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No station_latlon file found in {meta_dir}")


def load_latlon(code: str):
    path = find_latlon_file(code)
    df = pd.read_csv(path)

    # Flexible column handling
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["station_id", "sensor_id", "id"]:
            col_map[c] = "sensor_id"
        elif cl in ["latitude", "lat"]:
            col_map[c] = "latitude"
        elif cl in ["longitude", "lon", "lng"]:
            col_map[c] = "longitude"

    df = df.rename(columns=col_map)

    required = {"sensor_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df = df[["sensor_id", "latitude", "longitude"]].copy()
    df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["sensor_id", "latitude", "longitude"])
    df = df.drop_duplicates(subset=["sensor_id"])

    return df


def count_edges(adj: pd.DataFrame):
    values = pd.to_numeric(adj.stack(), errors="coerce").unstack().fillna(0)
    return int((values.values != 0).sum())


def make_sensor_map(dataset_name, code, sensors):


    latlon = load_latlon(code)
    latlon = latlon[latlon["sensor_id"].isin(set(sensors))].copy()

    if latlon.empty:
        print(f"[WARN] No coordinates found for {dataset_name}")
        return None, 0

    out_csv = FIGURES_DIR / f"{dataset_name}_sensor_locations.csv"
    latlon.to_csv(out_csv, index=False)

    # Convert to GeoDataFrame: WGS84 lat/lon -> Web Mercator for map tiles
    gdf = gpd.GeoDataFrame(
        latlon,
        geometry=gpd.points_from_xy(latlon["longitude"], latlon["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(9, 7))

    gdf.plot(
        ax=ax,
        markersize=50,        # bigger points
        color="blue",          # strong color
        edgecolor="black",    # outline for visibility
        linewidth=0.5,
        alpha=0.9,
    )

    # Add OpenStreetMap basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom="auto",
    )

    ax.set_title(f"{dataset_name.upper()} Sensor Locations (N={len(gdf)})")
    ax.set_axis_off()

    out_png = FIGURES_DIR / f"{dataset_name}_sensor_map.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return out_png, len(latlon)


def summarize_dataset(folder: Path):
    name = folder.name
    code = district_code_from_folder(name)

    speed_path = folder / "speed_cut.csv"
    closure_path = folder / "closure_cut.csv"

    if not speed_path.exists() or not closure_path.exists():
        print(f"[SKIP] {name}: missing speed or closure file")
        return None

    flow = read_time_sensor_csv(speed_path)
    closure = read_time_sensor_csv(closure_path)

    sensors = [s for s in flow.columns if s in closure.columns]
    flow = flow[sensors]
    closure = closure[sensors]

    n_sensors = len(sensors)
    n_timestamps = len(flow)

    start_date = flow.index.min()
    end_date = flow.index.max()
    duration = end_date - start_date

    # Closure activity
    closure_binary = closure > 0.5

    closure_per_sensor = closure_binary.sum(axis=0)
    closure_sensors = closure_per_sensor[closure_per_sensor > 0].index.tolist()
    non_closure_sensors = closure_per_sensor[closure_per_sensor == 0].index.tolist()

    # 80/20 split for seen/unseen closure sensors
    split_idx = int(len(closure_binary) * 0.80)

    train_closure = closure_binary.iloc[:split_idx]
    test_closure = closure_binary.iloc[split_idx:]

    seen_closure_sensors = set(
        train_closure.columns[train_closure.any(axis=0)]
    )

    test_closure_sensors = set(
        test_closure.columns[test_closure.any(axis=0)]
    )

    unseen_closure_sensors = test_closure_sensors - seen_closure_sensors

    # Timestamp-level closure counts
    total_closure_timestamps = int(closure_binary.any(axis=1).sum())
    total_nonclosure_timestamps = int(len(closure_binary) - total_closure_timestamps)

    seen_closure_timestamps = int(train_closure.any(axis=1).sum())
    unseen_closure_timestamps = int(test_closure[list(unseen_closure_sensors)].any(axis=1).sum()) \
        if unseen_closure_sensors else 0

    ratio_closure_vs_all = total_closure_timestamps / n_timestamps

    # Still make map
    make_sensor_map(name, code, sensors)
    return {
        "dataset": name,
        "district": code,
        "data_type": "flow",

        "number_of_sensors": n_sensors,
        "start_date": start_date,
        "end_date": end_date,
        "duration": str(duration),

        "closure_sensors": len(closure_sensors),
        "non_closure_sensors": len(non_closure_sensors),
        "seen_closure_sensors": len(seen_closure_sensors),
        "unseen_closure_sensors_last_20_percent": len(unseen_closure_sensors),

        "frequency": "5 min",
        "total_number_of_timestamps": n_timestamps,

        "total_number_of_closure_timestamps": total_closure_timestamps,
        "total_number_of_seen_closure_timestamps": seen_closure_timestamps,
        "total_number_of_unseen_closure_timestamps": unseen_closure_timestamps,

        "ratio_closure_vs_all": ratio_closure_vs_all,
    }


def summarize_tyson_dataset():
    name = "tyson"

    speed_path = TYSON_FILES["speed"]
    closure_path = TYSON_FILES["closure"]

    if not speed_path.exists() or not closure_path.exists():
        print("[SKIP] Tyson: missing speed or closure file")
        return None

    speed = read_time_sensor_csv(speed_path)
    closure = read_time_sensor_csv(closure_path)

    sensors = [s for s in speed.columns if s in closure.columns]
    speed = speed[sensors]
    closure = closure[sensors]

    n_sensors = len(sensors)
    n_timestamps = len(speed)

    start_date = speed.index.min()
    end_date = speed.index.max()
    duration = end_date - start_date

    closure_binary = closure > 0.5

    closure_per_sensor = closure_binary.sum(axis=0)
    closure_sensors = closure_per_sensor[closure_per_sensor > 0].index.tolist()
    non_closure_sensors = closure_per_sensor[closure_per_sensor == 0].index.tolist()

    split_idx = int(len(closure_binary) * 0.80)

    train_closure = closure_binary.iloc[:split_idx]
    test_closure = closure_binary.iloc[split_idx:]

    seen_closure_sensors = set(
        train_closure.columns[train_closure.any(axis=0)]
    )

    test_closure_sensors = set(
        test_closure.columns[test_closure.any(axis=0)]
    )

    unseen_closure_sensors = test_closure_sensors - seen_closure_sensors

    total_closure_timestamps = int(closure_binary.any(axis=1).sum())

    seen_closure_timestamps = int(train_closure.any(axis=1).sum())

    unseen_closure_timestamps = (
        int(test_closure[list(unseen_closure_sensors)].any(axis=1).sum())
        if unseen_closure_sensors
        else 0
    )

    ratio_closure_vs_all = total_closure_timestamps / n_timestamps

    return {
        "dataset": name,
        "district": "Tyson",
        "data_type": "speed",
        "number_of_sensors": n_sensors,
        "start_date": start_date,
        "end_date": end_date,
        "duration": str(duration),
        "closure_sensors": len(closure_sensors),
        "non_closure_sensors": len(non_closure_sensors),
        "seen_closure_sensors": len(seen_closure_sensors),
        "unseen_closure_sensors_last_20_percent": len(unseen_closure_sensors),
        "frequency": "5 min",
        "total_number_of_timestamps": n_timestamps,
        "total_number_of_closure_timestamps": total_closure_timestamps,
        "total_number_of_seen_closure_timestamps": seen_closure_timestamps,
        "total_number_of_unseen_closure_timestamps": unseen_closure_timestamps,
        "ratio_closure_vs_all": ratio_closure_vs_all,
    }

# ======================================================
# MAIN
# ======================================================

def main():
    folders = choose_dataset_folders(FINAL_DIR)

    print("[INFO] Selected dataset folders:")
    for f in folders:
        print(f"  - {f}")

    rows = []

    for folder in folders:
        print("=" * 70)
        print(f"[INFO] Processing {folder.name}")
        row = summarize_dataset(folder)
        if row is not None:
            rows.append(row)

    tyson_row = summarize_tyson_dataset()
    if tyson_row is not None:
        rows.append(tyson_row)

    overview = pd.DataFrame(rows)
    overview.to_csv(OVERVIEW_OUT, index=False)

    

    print("=" * 70)
    print(f"[DONE] Saved overview table:")
    print(f"  {OVERVIEW_OUT}")

    print(f"[DONE] Saved maps and coordinate files inside:")
    print(f"  {FIGURES_DIR}")


if __name__ == "__main__":
    main()