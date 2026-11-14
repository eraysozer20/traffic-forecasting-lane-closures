import json
import os
import pandas as pd

# =========================
# CONFIG (EDIT THESE)
# =========================
JSON_PATH = "jsons/ablation_compare_summary_Tyson.json"
DISTRICT = "Tyson"   # e.g. "03", "11"
# =========================


MODE_NAMES = {
    "baseline_no": "Baseline no",
    "baseline": "Baseline",
    "rescal_base_only": "Baseline + ResCAL",
    "rescal_base_plus_fold_no_weekly": "Baseline + ResCAL + no weekly + fold",
    "rescal_base_plus_fold_no_closure": "Baseline + ResCAL + no closure + fold",
    "rescal_base_plus_fold": "Baseline + ResCAL + fold",
}

MODE_ORDER = list(MODE_NAMES.keys())


def fmt_mean_std(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return "-"

    mean = values.mean()

    if len(values) == 1:
        return f"{mean:.3f}"

    std = values.std(ddof=1)
    return f"{mean:.3f} ± {std:.3f}"


def add_running_time(df):
    df = df.copy()

    def compute(row):
        if row["mode"] in ["baseline", "baseline_no"]:
            return row.get("baseline_train_seconds", 0)

        return (
            row.get("baseline_train_seconds", 0)
            + row.get("residual_build_seconds", 0)
            + row.get("rescal_train_seconds", 0)
        )

    df["running_time_seconds"] = df.apply(compute, axis=1)
    return df


def make_summary_table(df, columns, output_names):
    rows = []

    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode]

        if sub.empty:
            continue

        row = {"Model": MODE_NAMES[mode]}

        for col, out_name in zip(columns, output_names):
            row[out_name] = fmt_mean_std(sub[col])

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    district = str(DISTRICT).zfill(2)
    out_dir = os.path.join("tables", f"d{district}")
    os.makedirs(out_dir, exist_ok=True)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df = add_running_time(df)

    # -------------------------
    # Table 1: General results
    # -------------------------
    general_table = make_summary_table(
        df,
        ["final_mae", "final_rmse", "final_mape"],
        ["Test MAE", "Test RMSE", "Test MAPE"],
    )

    # -------------------------
    # Table 2: Seen / Unseen
    # -------------------------
    event_table = make_summary_table(
        df,
        [
            "event_final_seen_MAE_all",
            "event_final_seen_RMSE_all",
            "event_final_seen_MAE_local",
            "event_final_seen_RMSE_local",
            "event_final_unseen_MAE_all",
            "event_final_unseen_RMSE_all",
            "event_final_unseen_MAE_local",
            "event_final_unseen_RMSE_local",
        ],
        [
            "Seen MAE all",
            "Seen RMSE all",
            "Seen MAE local",
            "Seen RMSE local",
            "Unseen MAE all",
            "Unseen RMSE all",
            "Unseen MAE local",
            "Unseen RMSE local",
        ],
    )

    # -------------------------
    # Table 3: Running time
    # -------------------------
    time_table = make_summary_table(
        df,
        ["running_time_seconds"],
        ["Running time (sec)"],
    )

    # -------------------------
    # Save
    # -------------------------
    general_table.to_csv(os.path.join(out_dir, "table_1_general_results.csv"), index=False)
    event_table.to_csv(os.path.join(out_dir, "table_2_seen_unseen_results.csv"), index=False)
    time_table.to_csv(os.path.join(out_dir, "table_3_running_time.csv"), index=False)

    with pd.ExcelWriter(os.path.join(out_dir, "ablation_tables.xlsx")) as writer:
        general_table.to_excel(writer, sheet_name="General Results", index=False)
        event_table.to_excel(writer, sheet_name="Seen Unseen Results", index=False)
        time_table.to_excel(writer, sheet_name="Running Time", index=False)

    # -------------------------
    # Print
    # -------------------------
    print("\nTable 1: General Results")
    print(general_table.to_string(index=False))

    print("\nTable 2: Seen and Unseen Results")
    print(event_table.to_string(index=False))

    print("\nTable 3: Running Time")
    print(time_table.to_string(index=False))

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()