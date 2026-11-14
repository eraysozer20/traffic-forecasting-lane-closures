import os
import json
import glob
import pandas as pd
import numpy as np

# =========================
# Settings
# =========================
ROOT = "save"
MODEL_VARIANT = "wz"
BASE_MODEL = "lstm"

DATASETS = [
    "PEMSD3SUB",
    "PEMSD4SUB",
    "PEMSD5",
    "PEMSD6",
    "PEMSD8SUB",
    "PEMSD10",
    "PEMSD11SUB",
    "PEMSD12SUB",
    "Tyson",
]

OUTPUT_DIR = "tables/ablation_summary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAE_OUT = os.path.join(OUTPUT_DIR, "ablation_final_mae_table.csv")
RMSE_OUT = os.path.join(OUTPUT_DIR, "ablation_final_rmse_table.csv")


def find_json_file(folder):
    """
    Finds the first JSON file inside the folder or its subfolders.
    """
    json_files = glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    return json_files[0] if json_files else None


def load_json_records(json_path):
    """
    Loads JSON as a list of records.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        return data

    return []


def mean_std(records, metric):
    """
    Computes mean ± std for a metric from JSON records.
    """
    vals = []

    for r in records:
        if r.get("status", "ok") != "ok":
            continue

        v = r.get(metric, None)
        if v is not None:
            try:
                vals.append(float(v))
            except ValueError:
                pass

    if len(vals) == 0:
        return ""

    mean = np.mean(vals)
    std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    return f"{mean:.3f} ± {std:.3f}"


def main():
    mae_rows = []
    rmse_rows = []
    all_columns = set()

    raw_results = {}

    for dataset in DATASETS:
        base_path = os.path.join(ROOT, dataset, MODEL_VARIANT, BASE_MODEL)

        if not os.path.exists(base_path):
            print(f"[SKIP] Missing path: {base_path}")
            continue

        adj_folders = [
            d for d in os.listdir(base_path)
            if d.startswith("adj-") and os.path.isdir(os.path.join(base_path, d))
        ]

        raw_results[dataset] = {}

        for folder_name in sorted(adj_folders):
            folder_path = os.path.join(base_path, folder_name)
            json_path = find_json_file(folder_path)

            if json_path is None:
                print(f"[SKIP] No JSON found in: {folder_path}")
                continue

            try:
                records = load_json_records(json_path)
            except Exception as e:
                print(f"[SKIP] Failed reading {json_path}: {e}")
                continue

            mae_val = mean_std(records, "final_mae")
            rmse_val = mean_std(records, "final_rmse")

            raw_results[dataset][folder_name] = {
                "final_mae": mae_val,
                "final_rmse": rmse_val,
            }

            all_columns.add(folder_name)

    all_columns = sorted(all_columns)

    for dataset in DATASETS:
        mae_row = {"Dataset": dataset}
        rmse_row = {"Dataset": dataset}

        for col in all_columns:
            mae_row[col] = raw_results.get(dataset, {}).get(col, {}).get("final_mae", "")
            rmse_row[col] = raw_results.get(dataset, {}).get(col, {}).get("final_rmse", "")

        mae_rows.append(mae_row)
        rmse_rows.append(rmse_row)

    mae_df = pd.DataFrame(mae_rows, columns=["Dataset"] + all_columns)
    rmse_df = pd.DataFrame(rmse_rows, columns=["Dataset"] + all_columns)

    mae_df.to_csv(MAE_OUT, index=False)
    rmse_df.to_csv(RMSE_OUT, index=False)

    print(f"[OK] Saved MAE table:  {MAE_OUT}")
    print(f"[OK] Saved RMSE table: {RMSE_OUT}")


if __name__ == "__main__":
    main()