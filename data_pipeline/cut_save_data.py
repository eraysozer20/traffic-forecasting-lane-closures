import pandas as pd
import os
import sys

# 1. CONFIGURATION
# ==========================================

# Input date from command line:
# example:
# python cut_dataset.py 2016-03-24
if len(sys.argv) < 2:
    raise ValueError("Please provide BEST_START_DATE as input, e.g. 2016-03-24")

BEST_START_DATE = sys.argv[1]

# Validate input date format
try:
    start_dt = pd.Timestamp(BEST_START_DATE)
except Exception:
    raise ValueError(f"Invalid date format: {BEST_START_DATE}. Expected format: YYYY-MM-DD")

# Define the Window Duration
WINDOW_SIZE = pd.DateOffset(years=0, months=6)

# Input Paths
DISTRICT = "03"   # default

_env_d = os.getenv("PEMS_DISTRICT_ID")
if _env_d:
    DISTRICT = f"{int(_env_d):02d}"

base_dir = r'data\despt_inputs\pems' + DISTRICT
closure_path = os.path.join(base_dir, 'closure_labels.h5')
speed_path   = os.path.join(base_dir, 'speed.h5')

# Output Paths
output_closure_path = os.path.join(base_dir, 'closure_cut.h5')
output_speed_path   = os.path.join(base_dir, 'speed_cut.h5')

# 2. CALCULATION
# ==========================================
end_dt = start_dt + WINDOW_SIZE

print(f"--- CUTTING DATASET ---")
print(f"Window Start: {start_dt}")
print(f"Window End:   {end_dt}")
print(f"Duration:     0 Years, 6 Months")
print("-" * 30)

# 3. PROCESS CLOSURE LABELS
# ==========================================
print(f"1. Processing Closure File: {closure_path}")
try:
    df_closure = pd.read_hdf(closure_path, key='closure_labels')

    if not isinstance(df_closure.index, pd.DatetimeIndex):
        df_closure.index = pd.to_datetime(df_closure.index)

    df_closure_cut = df_closure[start_dt:end_dt]

    print(f"   Original Shape: {df_closure.shape}")
    print(f"   Cut Shape:      {df_closure_cut.shape}")

    if df_closure_cut.empty:
        raise ValueError("Error: The sliced closure dataframe is empty. Check dates!")

    df_closure_cut.to_hdf(output_closure_path, key='closure_labels', mode='w')
    print(f"   Saved to: {output_closure_path}")

except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("-" * 30)

# 4. PROCESS SPEED DATA
# ==========================================
print(f"2. Processing Speed File: {speed_path}")
try:
    with pd.HDFStore(speed_path, mode='r') as store:
        key = store.keys()[0]
        df_speed = store.get(key)

    if not isinstance(df_speed.index, pd.DatetimeIndex):
        df_speed.index = pd.to_datetime(df_speed.index)

    df_speed_cut = df_speed[start_dt:end_dt]

    print(f"   Original Shape: {df_speed.shape}")
    print(f"   Cut Shape:      {df_speed_cut.shape}")

    if df_speed_cut.shape[1] != df_closure_cut.shape[1]:
        print("   WARNING: Node count mismatch!")
        print(f"   Closure Nodes: {df_closure_cut.shape[1]}")
        print(f"   Speed Nodes:   {df_speed_cut.shape[1]}")

    df_speed_cut.to_hdf(output_speed_path, key='speed', mode='w')
    print(f"   Saved to: {output_speed_path}")

except FileNotFoundError:
    print(f"   ERROR: Speed file not found at {speed_path}")
except Exception as e:
    print(f"   ERROR processing speed file: {e}")

print("-" * 30)
print("DONE. Ready for training.")