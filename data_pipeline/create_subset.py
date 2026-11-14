#!/usr/bin/env python3
from pathlib import Path
import random
import re
import numpy as np
import pandas as pd


# ======================================================
# CONFIG
# ======================================================

DISTRICT = 12
MAX_SENSORS = 130

# Choose: "random", "closure_max", "spaced"
METHOD = "spaced"

SEED = 42

ROOT = Path("data")

DISTRICT_CODE = f"{DISTRICT:02d}"

FINAL_IN_DIR = ROOT / "final" / f"pems{DISTRICT_CODE}"
FINAL_OUT_DIR = ROOT / "final" / f"pems{DISTRICT_CODE}sub"

DIST_DIR = ROOT / "dist" / f"pems{DISTRICT_CODE}"
META_DIR = ROOT / "meta" / f"meta{DISTRICT_CODE}"

SPEED_IN = FINAL_IN_DIR / "speed_cut.csv"
CLOSURE_IN = FINAL_IN_DIR / "closure_cut.csv"

DISTANCES_CSV = DIST_DIR / f"distances_d{DISTRICT_CODE}_route.csv"

# Optional: set manually if auto-detection fails
STATION_META_CSV = None
# STATION_META_CSV = META_DIR / "station_meta_dist3_2001-2024.csv"


# ======================================================
# READ HELPERS
# ======================================================

def read_time_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = df.columns[0]
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.astype(str).str.strip()
    return df


def read_distances_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing distance file: {path}")

    df = pd.read_csv(path)
    required = {"from", "to", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df["from"] = df["from"].astype(str).str.strip()
    df["to"] = df["to"].astype(str).str.strip()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    return df.dropna(subset=["cost"])


def find_station_meta_csv(meta_dir: Path) -> Path:
    if STATION_META_CSV is not None:
        if not STATION_META_CSV.exists():
            raise FileNotFoundError(f"STATION_META_CSV not found: {STATION_META_CSV}")
        return STATION_META_CSV

    candidates = sorted(meta_dir.glob("*.csv"))
    scored = []

    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=5)
            cols = set(head.columns)
            need = {"station_id", "snapshot_date", "Fwy", "Abs PM"}
            if need.issubset(cols):
                bonus = 0
                for extra in ["Type", "HOV", "District"]:
                    if extra in cols:
                        bonus += 1
                scored.append((bonus, str(p), p))
        except Exception:
            continue

    if not scored:
        raise FileNotFoundError(
            f"Could not auto-find station meta CSV with columns "
            f"station_id, snapshot_date, Fwy, Abs PM in {meta_dir}"
        )

    scored.sort(reverse=True)
    chosen = scored[0][2]
    print(f"[INFO] Using station meta: {chosen}")
    return chosen


# ======================================================
# SENSOR SELECTION
# ======================================================

def common_sensors(speed: pd.DataFrame, closure: pd.DataFrame, dist_long: pd.DataFrame) -> list[str]:
    speed_ids = list(speed.columns)
    closure_ids = set(closure.columns)

    dist_ids = set(dist_long["from"]) | set(dist_long["to"])

    common = [sid for sid in speed_ids if sid in closure_ids and sid in dist_ids]

    if not common:
        raise ValueError("No common sensors found across speed, closure, and distance files.")

    return common


def select_random(sensors: list[str], max_sensors: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sensors = list(sensors)
    rng.shuffle(sensors)
    return sensors[:max_sensors]


def select_closure_max(
    sensors: list[str],
    closure: pd.DataFrame,
    max_sensors: int,
    seed: int,
) -> list[str]:
    counts = closure[sensors].sum(axis=0).sort_values(ascending=False)

    active = counts[counts > 0].index.tolist()
    inactive = counts[counts == 0].index.tolist()

    selected = active[:max_sensors]

    if len(selected) < max_sensors:
        rng = random.Random(seed)
        rng.shuffle(inactive)
        selected += inactive[: max_sensors - len(selected)]

    return selected


def build_distance_matrix_from_long(dist_long: pd.DataFrame, sensors: list[str]) -> np.ndarray:
    idx = {sid: i for i, sid in enumerate(sensors)}
    N = len(sensors)

    D = np.full((N, N), np.inf, dtype=float)

    for _, row in dist_long.iterrows():
        f, t = row["from"], row["to"]
        if f in idx and t in idx:
            D[idx[f], idx[t]] = float(row["cost"])

    np.fill_diagonal(D, 0.0)
    return D


def symmetrize_distance_matrix(D: np.ndarray) -> np.ndarray:
    D2 = np.full_like(D, np.inf, dtype=float)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            a = D[i, j]
            b = D[j, i]
            vals = [x for x in [a, b] if np.isfinite(x)]
            D2[i, j] = min(vals) if vals else np.inf

    np.fill_diagonal(D2, 0.0)

    finite = D2[np.isfinite(D2)]
    if finite.size == 0:
        raise ValueError("No finite distances found.")

    large = finite.max() * 10
    D2[~np.isfinite(D2)] = large
    return D2


def select_spaced(
    sensors: list[str],
    dist_long: pd.DataFrame,
    max_sensors: int,
    seed: int,
) -> list[str]:
    D = build_distance_matrix_from_long(dist_long, sensors)
    D = symmetrize_distance_matrix(D)

    rng = random.Random(seed)
    start = rng.randrange(len(sensors))

    selected_idx = [start]
    remaining = set(range(len(sensors))) - set(selected_idx)

    while len(selected_idx) < max_sensors and remaining:
        best_i = None
        best_score = -1.0

        for i in remaining:
            min_dist_to_selected = min(D[i, j] for j in selected_idx)
            if min_dist_to_selected > best_score:
                best_score = min_dist_to_selected
                best_i = i

        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [sensors[i] for i in selected_idx]


def select_sensors(
    method: str,
    sensors: list[str],
    closure: pd.DataFrame,
    dist_long: pd.DataFrame,
    max_sensors: int,
    seed: int,
) -> list[str]:
    if max_sensors >= len(sensors):
        return sensors

    if method == "random":
        return select_random(sensors, max_sensors, seed)

    if method == "closure_max":
        return select_closure_max(sensors, closure, max_sensors, seed)

    if method == "spaced":
        return select_spaced(sensors, dist_long, max_sensors, seed)

    raise ValueError(f"Unknown METHOD: {method}")


# ======================================================
# ADJACENCY 1: DISTANCE-BASED
# ======================================================

def get_adjacency_matrix(dist_mx_array: np.ndarray, normalized_k: float = 0.1) -> np.ndarray:
    finite = ~np.isinf(dist_mx_array)
    distances = dist_mx_array[finite].flatten()

    std = distances.std() if distances.size > 0 else 1.0
    if std == 0:
        std = 1.0

    adj = np.exp(-np.square(dist_mx_array / std))
    adj[adj < normalized_k] = 0
    adj[np.isinf(dist_mx_array)] = 0
    return adj


def build_distance_outputs(
    dist_long: pd.DataFrame,
    selected: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    D = build_distance_matrix_from_long(dist_long, selected)

    adj_weighted = get_adjacency_matrix(D, normalized_k=0.1)
    adj1 = pd.DataFrame(adj_weighted, index=selected, columns=selected)

    miles = D.copy()
    miles[np.isinf(miles)] = 0.0
    miles = miles / 1609.344
    adj1_miles = pd.DataFrame(miles, index=selected, columns=selected)

    return adj1, adj1_miles


# ======================================================
# ADJACENCY 2: TOPOLOGY-BASED, FROM YOUR SCRIPT
# ======================================================

def load_latest_station_meta(meta_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(meta_csv)

    required = {"station_id", "snapshot_date", "Fwy", "Abs PM"}
    missing = required - set(m.columns)
    if missing:
        raise KeyError(f"Station meta missing required columns {missing} in {meta_csv}")

    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"], errors="coerce")
    m = m.dropna(subset=["snapshot_date"]).copy()
    m["station_id"] = m["station_id"].astype(str).str.strip()

    latest = (
        m.sort_values(["station_id", "snapshot_date"])
        .drop_duplicates("station_id", keep="last")
        .reset_index(drop=True)
    )
    return latest


def parse_route_dir(fwy: str):
    if not isinstance(fwy, str):
        return None, None

    fwy = fwy.strip()
    mm = re.match(r"^(.*)-([NSEW])$", fwy)
    if mm:
        return mm.group(1), mm.group(2)

    return fwy, None


def build_adj_from_meta(
    latest_meta: pd.DataFrame,
    sensor_ids: list[str],
    pm_tol: float = 0.02,
) -> pd.DataFrame:
    nodes = list(map(str, sensor_ids))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    m = latest_meta.copy()
    m["station_id"] = m["station_id"].astype(str).str.strip()
    m = m[m["station_id"].isin(node_to_idx)].copy()

    m["Abs PM"] = pd.to_numeric(m["Abs PM"], errors="coerce")
    m = m.dropna(subset=["Abs PM", "Fwy"]).copy()

    rd = m["Fwy"].apply(parse_route_dir)
    m["route"] = rd.apply(lambda x: x[0])
    m["dir"] = rd.apply(lambda x: x[1])
    m = m.dropna(subset=["dir"]).copy()

    m["Type_norm"] = (
        m["Type"].astype(str).str.strip().str.lower()
        if "Type" in m.columns
        else ""
    )
    m["HOV_norm"] = (
        m["HOV"].astype(str).str.strip().str.lower()
        if "HOV" in m.columns
        else "no"
    )

    def is_hov(row) -> bool:
        return row["HOV_norm"] in ["yes", "y", "true", "1"]

    def is_mainline_like(row) -> bool:
        if is_hov(row):
            return True
        return row["Type_norm"] in ["mainline", "ml"]

    def is_on_ramp(row) -> bool:
        return row["Type_norm"] in ["on ramp", "or"]

    adj = np.zeros((N, N), dtype=np.float32)

    group_cols = ["route", "dir"]
    if "District" in m.columns:
        group_cols.append("District")

    for _, g in m.groupby(group_cols):
        ddir = str(g["dir"].iloc[0]).upper()
        ascending = ddir in ["N", "E"]

        g_sorted = g.sort_values("Abs PM", ascending=ascending).reset_index(drop=True)

        mainline = g_sorted[g_sorted.apply(is_mainline_like, axis=1)].copy().reset_index(drop=True)
        onramps = g_sorted[g_sorted.apply(is_on_ramp, axis=1)].copy().reset_index(drop=True)

        main_ids = mainline["station_id"].tolist()

        for u, v in zip(main_ids[:-1], main_ids[1:]):
            ui, vi = node_to_idx.get(u), node_to_idx.get(v)
            if ui is not None and vi is not None:
                adj[ui, vi] = 1.0

        if len(mainline) == 0:
            continue

        main_pm = mainline["Abs PM"].to_numpy(dtype=float)
        main_sid = mainline["station_id"].tolist()

        def attach_mainline_idx(pm_val: float):
            same = np.where(np.abs(main_pm - pm_val) <= pm_tol)[0]
            if same.size > 0:
                return int(same[0])
            return int(np.argmin(np.abs(main_pm - pm_val)))

        for _, r in onramps.iterrows():
            ramp_id = r["station_id"]
            pm_val = float(r["Abs PM"])

            j = attach_mainline_idx(pm_val)
            target_main = main_sid[j]

            ri, ti = node_to_idx.get(ramp_id), node_to_idx.get(target_main)
            if ri is not None and ti is not None:
                adj[ri, ti] = 1.0

    return pd.DataFrame(adj, index=nodes, columns=nodes)


# ======================================================
# SAVE
# ======================================================

def save_outputs(
    speed: pd.DataFrame,
    closure: pd.DataFrame,
    selected: list[str],
    adj1: pd.DataFrame,
    adj1_miles: pd.DataFrame,
    adj2: pd.DataFrame,
):
    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    speed[selected].to_csv(FINAL_OUT_DIR / "speed_cut.csv", index_label="")
    closure[selected].to_csv(FINAL_OUT_DIR / "closure_cut.csv", index_label="")

    # Following your naming:
    # adj_1.csv      = distance-based weighted adjacency from Gaussian kernel
    # adj_1_miles.csv = raw OSRM distances in miles
    # adj.csv        = topology-based adjacency
    adj1.to_csv(FINAL_OUT_DIR / "adj_1.csv")
    adj1_miles.to_csv(FINAL_OUT_DIR / "adj_1_miles.csv")
    adj2.to_csv(FINAL_OUT_DIR / "adj.csv")

    pd.DataFrame({"sensor_id": selected}).to_csv(
        FINAL_OUT_DIR / "selected_sensors.csv",
        index=False,
    )


def main():
    print("=" * 70)
    print(f"District:      {DISTRICT_CODE}")
    print(f"Method:        {METHOD}")
    print(f"Max sensors:   {MAX_SENSORS}")
    print(f"Input folder:  {FINAL_IN_DIR}")
    print(f"Output folder: {FINAL_OUT_DIR}")
    print("=" * 70)

    speed = read_time_sensor_csv(SPEED_IN)
    closure = read_time_sensor_csv(CLOSURE_IN)
    dist_long = read_distances_long(DISTANCES_CSV)

    sensors = common_sensors(speed, closure, dist_long)
    print(f"[INFO] Common sensors: {len(sensors)}")

    selected = select_sensors(
        method=METHOD,
        sensors=sensors,
        closure=closure,
        dist_long=dist_long,
        max_sensors=MAX_SENSORS,
        seed=SEED,
    )

    # Preserve original speed file order
    selected_set = set(selected)
    selected = [sid for sid in sensors if sid in selected_set]

    print(f"[INFO] Selected sensors: {len(selected)}")

    adj1, adj1_miles = build_distance_outputs(dist_long, selected)

    meta_csv = find_station_meta_csv(META_DIR)
    latest_meta = load_latest_station_meta(meta_csv)
    adj2 = build_adj_from_meta(latest_meta, selected, pm_tol=0.02)

    save_outputs(speed, closure, selected, adj1, adj1_miles, adj2)

    closure_counts = closure[selected].sum(axis=0)
    print("[INFO] Closure-active selected sensors:", int((closure_counts > 0).sum()))
    print("[INFO] Total selected closure labels:", float(closure_counts.sum()))

    print("[SAVED]")
    print(f"  {FINAL_OUT_DIR / 'speed_cut.csv'}")
    print(f"  {FINAL_OUT_DIR / 'closure_cut.csv'}")
    print(f"  {FINAL_OUT_DIR / 'adj.csv'}")
    print(f"  {FINAL_OUT_DIR / 'adj_1.csv'}")
    print(f"  {FINAL_OUT_DIR / 'adj_1_miles.csv'}")
    print(f"  {FINAL_OUT_DIR / 'selected_sensors.csv'}")


if __name__ == "__main__":
    main()