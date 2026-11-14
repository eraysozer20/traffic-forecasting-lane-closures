import pandas as pd
import numpy as np
import os

def get_active_nodes(mask_slice):
    """Returns indices of nodes that have at least one closure in the slice."""
    closures_per_node = mask_slice.sum(axis=0)
    return np.where(closures_per_node > 0)[0]

def main():
    print("--- DETAILED DIAGNOSTICS FOR PEMS08 ---")
    data_path = './data'
    # Ensure this path matches where you uploaded the file on the cluster
    mask_path = os.path.join(data_path, 'despt_inputs/pems08/closure_cut.h5')
    
    if not os.path.exists(mask_path):
        print(f"Error: File not found at {mask_path}")
        return

    # 1. Load Data
    print(f"Loading {mask_path}...")
    mask = pd.read_hdf(mask_path).values # [Time, Nodes]
    T, N = mask.shape
    print(f"Total Time Steps: {T}")
    print(f"Total Nodes:      {N}")
    
    # 2. Define Split Ratios
    t_train = 0.70
    t_val   = 0.20 # Using 20% for val based on your selector script logic
    # t_test is implicit (remainder)
    
    # 3. Define Cutoff Points
    TRAIN_END_IDX = int(T * t_train)
    VAL_END_IDX   = int(T * (t_train + t_val))
    
    print(f"\n--- SPLIT BOUNDARIES ---")
    print(f"Train: 0 to {TRAIN_END_IDX} ({t_train:.0%})")
    print(f"Val:   {TRAIN_END_IDX} to {VAL_END_IDX} ({t_val:.0%})")
    print(f"Test:  {VAL_END_IDX} to {T} (Remainder)")
    
    # 4. Create Slices
    train_mask = mask[:TRAIN_END_IDX, :]
    val_mask   = mask[TRAIN_END_IDX:VAL_END_IDX, :]
    test_mask  = mask[VAL_END_IDX:, :]

    # 5. Define "Unseen" (Based strictly on Training Data)
    # A node is "Seen" if it appears in Train.
    seen_nodes_indices = get_active_nodes(train_mask)
    all_indices = np.arange(N)
    unseen_nodes_indices = np.setdiff1d(all_indices, seen_nodes_indices)

    print(f"\n--- NODE CLASSIFICATION (Based on Training Set) ---")
    print(f"Seen Nodes (Train):   {len(seen_nodes_indices)}")
    print(f"Unseen Nodes (Train): {len(unseen_nodes_indices)}")
    if len(unseen_nodes_indices) > 0:
        print(f"Unseen Node Indices: {unseen_nodes_indices}")
    else:
        print("CRITICAL: All nodes appear in the Training set!")

    # 6. Detailed Analysis per Split
    
    def analyze_split(name, mask_slice):
        active_nodes = get_active_nodes(mask_slice)
        
        # Intersection with Unseen Set
        active_unseen = np.intersect1d(active_nodes, unseen_nodes_indices)
        
        # Calculate TimeSteps (Volume of closure)
        total_timestamps = mask_slice.sum()
        unseen_timestamps = mask_slice[:, unseen_nodes_indices].sum()
        seen_timestamps = total_timestamps - unseen_timestamps
        
        print(f"\n[{name} SET ANALYSIS]")
        
        print(f"  > Active Nodes (Total):      {len(active_nodes)}")
        print(f"  > Active Unseen Nodes:       {len(active_unseen)}")
        if len(active_unseen) > 0:
            print(f"    Indices: {active_unseen}")
        
        print(f"  > Total Closure Timestamps:  {total_timestamps}")
        print(f"  > Seen Node Timestamps:      {seen_timestamps}")
        print(f"  > Unseen Node Timestamps:    {unseen_timestamps}")
        
        return unseen_timestamps

    # Run Analysis
    analyze_split("TRAIN", train_mask)
    val_unseen_ts = analyze_split("VAL", val_mask)
    test_unseen_ts = analyze_split("TEST", test_mask)
    
    # 7. Final Verdict
    print(f"\n--- FINAL VERDICT ---")
    if test_unseen_ts > 0:
        print("✅ SUCCESS: The TEST set contains closures on nodes not seen in Train.")
        print(f"   We have {test_unseen_ts} timestamps of 'Unseen Closure' data to evaluate.")
    elif val_unseen_ts > 0:
        print("⚠️ WARNING: Unseen nodes break in VAL, but NOT in TEST.")
        print("   Consider changing t_val=0.0 to capture these events in the test set.")
    else:
        print("❌ FAILURE: The Unseen nodes never break in Val or Test.")
        print("   Try a different start date window.")

if __name__ == "__main__":
    main()