import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Settings
# =========================
#DISTRICTS = ["03", "04", "05", "06", "08", "10", "11", "12", "tyson"]
DISTRICTS = ["04","08","11"]
BASE_DIR = "tables"

OUTPUT_CSV = os.path.join(BASE_DIR, "all_running_time.csv")
OUTPUT_PLOT = os.path.join(BASE_DIR, "all_running_time_plot.png")


def extract_mean(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", value)
    return float(match.group()) if match else None


def process_district(d):
    path = os.path.join(BASE_DIR, f"d{d}", "table_3_running_time.csv")

    if not os.path.exists(path):
        print(f"[SKIP] Missing: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except:
        print(f"[SKIP] Failed reading: {path}")
        return None

    if "Model" not in df.columns or "Running time (sec)" not in df.columns:
        print(f"[SKIP] Bad format: {path}")
        return None

    df["time"] = df["Running time (sec)"].apply(extract_mean)

    mapping = {
        "Base": "Baseline no",
        "+CF": "Baseline",
        "+CF+R": "Baseline + ResCAL",
        "+CF+R+F": "Baseline + ResCAL + no weekly + fold",
    }

    result = {}

    for variant, name in mapping.items():
        row = df[df["Model"].str.strip() == name]

        if row.empty:
            print(f"[SKIP] Missing {name} in d{d}")
            return None

        val = row["time"].iloc[0]
        if val is None:
            print(f"[SKIP] Parse error in d{d}")
            return None

        result[variant] = val

    #base = result["Base"]
    #base = 3 * base
    #cf = 3 * result["+CF"]
    #cfr = 3 * result["+CF"] + (-result["+CF"] + result["+CF+R"])
    #cfrf = 3 * result["+CF"] + (result["+CF+R+F"] - result["+CF"])

    base = result["Base"]
    base = 3 * base
    cf = 3 * result["+CF"]
    cfr = 3 * result["+CF"] + (-result["+CF"] + result["+CF+R"])
    cfrf = 3 * result["+CF"] + (result["+CF+R+F"] - result["+CF"])


    return {
        "Dataset": f"d{d}",
        "Base": base,
        "+CF": cf,
        "+CF+R": cfr,
        "+CF+R+F": cfrf,
        "+CF (%)": (cf - base) / base * 100,
        "+CF+R (%)": (cfr - base) / base * 100,
        "+CF+R+F (%)": (cfrf - base) / base * 100,
    }


def main():
    rows = []

    for d in DISTRICTS:
        res = process_district(d)
        if res:
            rows.append(res)

    if not rows:
        print("No valid data found.")
        return

    df_all = pd.DataFrame(rows)

    os.makedirs(BASE_DIR, exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Saved: {OUTPUT_CSV}")
    print(df_all)

    # =========================
    # Plot (Grouped bars)
    # =========================
    plt.figure(figsize=(10, 5))

    x = range(len(df_all))
    width = 0.2

    plt.bar([i - width for i in x], df_all["Base"], width, label="Base")
    plt.bar(x, df_all["+CF"], width, label="+CF")
    plt.bar([i + width for i in x], df_all["+CF+R"], width, label="+CF+R")
    plt.bar([i + 2*width for i in x], df_all["+CF+R+F"], width, label="+CF+R+F")

    plt.xticks(x, df_all["Dataset"])
    plt.ylabel("Running Time (sec)")
    plt.title("Running Time Comparison Across Datasets")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()

    print(f"[OK] Plot saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()