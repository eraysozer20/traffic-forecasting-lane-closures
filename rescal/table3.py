import os
import pandas as pd

# =========================
# Settings
# =========================
ROOT = "tables"   # change if d03, d04, ... are somewhere else
INPUT_FILE = "table_2_seen_unseen_results.csv"

DISTRICTS = [
    ("d03", "PEMSD03SUB"),
    ("d04", "PEMSD04SUB"),
    ("d05", "PEMSD05"),
    ("d06", "PEMSD06"),
    ("d08", "PEMSD08SUB"),
    ("d10", "PEMSD10"),
    ("d11", "PEMSD11SUB"),
    ("d12", "PEMSD12SUB"),
    ("dTyson", "Tyson"),
]

MODEL_MAP = {
    "Baseline no": "Base",
    "Baseline": "CF",
    "Baseline + ResCAL": "CF + R",
    "Baseline + ResCAL + no weekly + fold": "CF + R + F",
}

OUT_SEEN = os.path.join(ROOT, "seen_rmse_all_table.csv")
OUT_UNSEEN = os.path.join(ROOT, "unseen_rmse_all_table.csv")


def get_mean(value):
    """
    Converts values like:
        38.764 ± 0.726 -> 38.764
        39.945 -> 39.945
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    if "±" in value:
        value = value.split("±")[0].strip()

    return float(value)


def build_table(metric_col):
    rows = []

    for folder, dataset_name in DISTRICTS:
        path = os.path.join(ROOT, folder, INPUT_FILE)

        if not os.path.exists(path):
            print(f"[WARNING] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        row = {"Dataset": dataset_name}

        for old_name, new_name in MODEL_MAP.items():
            match = df[df["Model"] == old_name]

            if match.empty:
                print(f"[WARNING] Missing model '{old_name}' in {path}")
                row[new_name] = None
                continue

            row[new_name] = get_mean(match.iloc[0][metric_col])

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Create tables
# =========================
seen_rmse_all = build_table("Seen RMSE all")
unseen_rmse_all = build_table("Unseen RMSE all")

seen_rmse_all.to_csv(OUT_SEEN, index=False)
unseen_rmse_all.to_csv(OUT_UNSEEN, index=False)

print(f"Saved: {OUT_SEEN}")
print(f"Saved: {OUT_UNSEEN}")