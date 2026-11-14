import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# 1. CONFIGURATION
# ==========================================
file_path_h5 = r'data\despt_inputs\pems08\closure_labels.h5'
file_path_grades = r'data\lane_closures\pems08\pems08_graded_closures_v2.csv'

SEARCH_START_DATE = "2004-01-01"
WINDOW_SIZE = pd.DateOffset(years=0, months=6)
STRIDE = pd.DateOffset(days=7) 
RECENCY_BIAS = 1.5  

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# 2. LOAD DATA
# ==========================================
print("Loading HDF5...")
try:
    df_h5 = pd.read_hdf(file_path_h5, key='closure_labels')
except KeyError:
    with pd.HDFStore(file_path_h5, mode='r') as store:
        df_h5 = store.get(store.keys()[0])
if not isinstance(df_h5.index, pd.DatetimeIndex): df_h5.index = pd.to_datetime(df_h5.index)

print("Loading Grades CSV...")
df_grades = pd.read_csv(file_path_grades)
df_grades['Start Date'] = pd.to_datetime(df_grades['Start Date'])
df_grades['station_id'] = df_grades['station_id'].astype(str)

# Filter: Ignore completely noisy events (Grade < 0.1)
df_grades = df_grades[df_grades['closure_grade'] > 0.05]
print(f"Loaded {len(df_grades)} relevant graded events.")

# 3. HELPER FUNCTIONS
# ==========================================
def get_active_nodes(sub_df):
    if sub_df.empty: return set()
    has_closure = (sub_df > 0.5).any(axis=0)
    return set(has_closure[has_closure].index)

def get_grade_stats(start_date, end_date, target_nodes, grades_df):
    """
    Returns (Mean_Grade, Count) for events in this window/node set.
    """
    if len(target_nodes) == 0: return 0.0, 0
    
    mask_date = (grades_df['Start Date'] >= start_date) & (grades_df['Start Date'] <= end_date)
    mask_node = grades_df['station_id'].isin(list(target_nodes))
    
    subset = grades_df.loc[mask_date & mask_node, 'closure_grade']
    
    if len(subset) == 0:
        return 0.0, 0
    
    return subset.mean(), len(subset)

# 4. SLIDING WINDOW ANALYSIS (QUALITY FOCUSED)
# ==========================================
results = []
dataset_start = df_h5.index.min()
dataset_end = df_h5.index.max()
total_duration = (dataset_end - dataset_start).days

search_start_dt = pd.Timestamp(SEARCH_START_DATE)
current_window_start = max(dataset_start, search_start_dt)

print(f"\nScanning windows based on Closure Grade Quality (Ignoring Unseen status)...")
iteration_count = 0

while current_window_start + WINDOW_SIZE <= dataset_end:
    current_window_end = current_window_start + WINDOW_SIZE
    
    n_days_total = (current_window_end - current_window_start).days
    train_end_date = current_window_start + timedelta(days=int(n_days_total * TRAIN_RATIO))
    val_end_date   = current_window_start + timedelta(days=int(n_days_total * (TRAIN_RATIO + VAL_RATIO)))
    
    window_df = df_h5[current_window_start : current_window_end]
    
    if not window_df.empty:
        train_df = window_df[current_window_start : train_end_date]
        val_df   = window_df[train_end_date : val_end_date]
        test_df  = window_df[val_end_date : current_window_end]
        
        # Check for non-empty splits
        if not train_df.empty and not val_df.empty and not test_df.empty:
            
            # Identify ALL active nodes in each split
            train_nodes = get_active_nodes(train_df)
            val_nodes   = get_active_nodes(val_df)
            test_nodes  = get_active_nodes(test_df)
            
            # --- GET STATS (ALL ACTIVE NODES) ---
            # 1. Train Quality
            train_mean, train_count = get_grade_stats(current_window_start, train_end_date, train_nodes, df_grades)
            
            # 2. Val Quality (All active nodes)
            val_mean, val_count = get_grade_stats(train_end_date, val_end_date, val_nodes, df_grades)
            
            # 3. Test Quality (All active nodes)
            test_mean, test_count = get_grade_stats(val_end_date, current_window_end, test_nodes, df_grades)
            
            # --- SCORING LOGIC ---
            # Ensure there is at least SOME rated activity in all splits
            if (train_count >= 5) and (val_count >= 1) and (test_count >= 1):
                
                # Overall Mean Quality
                overall_quality = (train_mean + val_mean + test_mean) / 3.0
                
                # Recency Factor
                progress = (current_window_start - dataset_start).days / total_duration
                recency_factor = 1.0 + (progress * RECENCY_BIAS)
                
                # Confidence Bonus (More events = better trust in the mean)
                # We cap this so volume doesn't completely override quality
                confidence_bonus = np.log1p(test_count + val_count + train_count) * 0.05
                
                score = (overall_quality + confidence_bonus) * recency_factor
                
                results.append({
                    'Window_Start': current_window_start,
                    'Train_Mean': round(train_mean, 3),
                    'Val_Mean': round(val_mean, 3),
                    'Test_Mean': round(test_mean, 3),
                    'Overall_Mean': round(overall_quality, 3),
                    'Train_Events': train_count,
                    'Val_Events': val_count,
                    'Test_Events': test_count,
                    'Score': score
                })
    
    current_window_start += STRIDE
    iteration_count += 1
    if iteration_count % 50 == 0: print(f"Processed: {current_window_start.date()}", end='\r')

res_df = pd.DataFrame(results)

# 5. RESULTS
# ==========================================
if not res_df.empty:
    res_df = res_df.sort_values(by='Score', ascending=False)
    
    print("\n\n--- TOP 5 CANDIDATES (Best Quality Grades) ---")
    cols = ['Window_Start', 'Overall_Mean', 'Test_Mean', 'Train_Events', 'Val_Events', 'Test_Events', 'Score']
    print(res_df[cols].head(5))

    winner = res_df.iloc[0]
    print(f"\nRECOMMENDATION: Start Date {winner['Window_Start'].date()}")
    print(f"Overall Quality: {winner['Overall_Mean']:.3f}")
    print(f"Event Counts -> Train: {int(winner['Train_Events'])}, Val: {int(winner['Val_Events'])}, Test: {int(winner['Test_Events'])}")
else:
    print("\nNo suitable windows found with rated events in all splits.")