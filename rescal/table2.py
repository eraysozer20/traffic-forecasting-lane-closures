import os
import json
import math
import pandas as pd

ROOT_DIR = "save"
OUTPUT_CSV = "all_baseline_results_table.csv"

ROWS = []

ALLOWED_MODELS = ["gwnet", "lstm"]

KEEP_COLUMNS = [
    "dataset",
    "model_variant",
    "base_model",
    "mode",
    "run_idx",
    "status",

    "baseline_best_val_rmse",

    "baseline_mae",
    "baseline_rmse",
    "baseline_mape",

    "final_mae",
    "final_rmse",
    "final_mape",

    "event_final_seen_MAE_all",
    "event_final_seen_RMSE_all",
    "event_final_seen_MAE_local",
    "event_final_seen_RMSE_local",

    "event_final_unseen_MAE_all",
    "event_final_unseen_RMSE_all",
    "event_final_unseen_MAE_local",
    "event_final_unseen_RMSE_local",

    "baseline_train_seconds",
    "baseline_eval_seconds",
    "total_mode_seconds",

    "baseline_path",
]


def safe_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return x


def infer_from_path(json_path):
    parts = json_path.replace("\\", "/").split("/")

    dataset = None
    model_variant = None
    base_model = None

    # Expected:
    # save/DATASET/wz/lstm/ablation_compare_summary.json
    if len(parts) >= 5 and parts[0] == "save":
        dataset = parts[1]
        model_variant = parts[2]
        base_model = parts[3]

    return dataset, model_variant, base_model


for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file == "ablation_compare_summary.json":
            json_path = os.path.join(root, file)
            dataset, model_variant, base_model = infer_from_path(json_path)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for row in data:
                new_row = dict(row)
                new_row["dataset"] = dataset
                new_row["model_variant"] = model_variant
                new_row["base_model"] = base_model
                new_row["json_path"] = json_path

                for k, v in new_row.items():
                    new_row[k] = safe_float(v)

                ROWS.append(new_row)


df = pd.DataFrame(ROWS)

# Keep only gwnet and lstm
if "base_model" in df.columns:
    df = df[df["base_model"].isin(ALLOWED_MODELS)]

# Keep only useful columns that exist
cols = [c for c in KEEP_COLUMNS if c in df.columns]
df = df[cols]

# Sort neatly
sort_cols = [c for c in ["dataset", "base_model", "mode", "run_idx"] if c in df.columns]
df = df.sort_values(sort_cols)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved table to: {OUTPUT_CSV}")
print(df)