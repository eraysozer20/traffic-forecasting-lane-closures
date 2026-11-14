import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. SETUP
# ==========================================
file_path = r'data\despt_inputs\pems08\closure_labels.h5'
key_name = 'closure_labels'

# 2. LOAD DATA
# ==========================================
print("Loading data... (this might take a moment)")
try:
    df = pd.read_hdf(file_path, key=key_name)
except KeyError:
    # Fallback if the key is different, as seen in previous steps
    with pd.HDFStore(file_path, mode='r') as store:
        df = store.get(store.keys()[0])

# Ensure index is datetime
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

print(f"Data Loaded: {df.shape[0]} timestamps, {df.shape[1]} nodes.")

# 3. PROCESS DATA (Convert 0/1 states to Intervals)
# ==========================================
intervals = []

# Iterate through each node (column) to find closure intervals
# We use a threshold of 0.5 to detect '1's (in case data is float)
print("Processing intervals...")

for node in df.columns:
    # Get the series for this node
    series = df[node]
    
    # Skip nodes that never close to save time/space
    if series.sum() == 0:
        continue

    # Find where value changes from 0 to 1 (start) or 1 to 0 (end)
    # diff() calculates current - previous
    # 1  = Start of closure
    # -1 = End of closure
    diffs = series.diff()
    
    # Get the timestamps where changes happen
    starts = diffs[diffs == 1].index
    ends = diffs[diffs == -1].index
    
    # valid_starts and valid_ends lists
    start_list = starts.tolist()
    end_list = ends.tolist()

    # HANDLING EDGE CASES:
    # Case A: Data starts with a closure (so no +1 diff detected)
    if series.iloc[0] == 1:
        start_list.insert(0, df.index[0])
        
    # Case B: Data ends with a closure (so no -1 diff detected)
    if series.iloc[-1] == 1:
        end_list.append(df.index[-1])

    # Combine into intervals
    # zip truncates to shortest, but our logic above ensures lengths match
    for start, end in zip(start_list, end_list):
        intervals.append({
            'Node': node,
            'Start': start,
            'End': end,
            'Duration': end - start
        })

# Create a DataFrame of intervals for easy plotting
df_intervals = pd.DataFrame(intervals)

if df_intervals.empty:
    print("No closures found in the dataset.")
else:
    print(f"Found {len(df_intervals)} closure events.")

    # 4. PLOTTING (The Gantt Chart)
    # ==========================================
    # Sort by Node to keep the Y-axis organized
    df_intervals.sort_values(by='Node', inplace=True)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(15, 10)) # Width, Height

    # We use hlines (horizontal lines) to draw the intervals
    # X axis = Time, Y axis = Node ID
    # 'linewidth' controls the thickness of the bars
    ax.hlines(y=df_intervals['Node'], 
              xmin=df_intervals['Start'], 
              xmax=df_intervals['End'], 
              linewidth=5, 
              color='#d62728',  # Red color for closures
              alpha=0.8)        # Slight transparency

    # Formatting the Plot
    ax.set_xlabel("Time (Year)", fontsize=14)
    ax.set_ylabel("Node ID", fontsize=14)
    ax.set_title("Node Closures Over Time", fontsize=16)

    # Format X-axis to show Years nicely
    ax.xaxis.set_major_locator(mdates.YearLocator(2)) # Tick every 2 years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # If there are too many nodes on Y-axis, hide the labels to keep it clean
    if len(df_intervals['Node'].unique()) > 50:
        # Show only every nth label
        all_nodes = df_intervals['Node'].unique()
        ax.set_yticks(all_nodes[::5]) 
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    print("Displaying plot...")
    plt.show()