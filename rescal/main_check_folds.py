# main_check_folds.py
import os.path as osp
import numpy as np
import pandas as pd

from utils.cli import build_args
from utils.configs import apply_dataset_cfg
from utils.data_loading import (
    loadData_hyp2,
    loadData_hyp2_fold,
    build_closure_sensor_folds_from_args,
)

def main():
    args = build_args()
    apply_dataset_cfg(args)

    print("CONFIG:", args.dataset_choice, args.model_variant)
    print("Checking fold construction only\n")

    from utils.data_loading import summarize_sensor_closure_open_counts

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)

    num_steps = df_con.shape[0]
    len_train = int(round(0.7 * num_steps))
    df_con_A = df_con.iloc[:len_train]

    sensor_summary = summarize_sensor_closure_open_counts(df_con_A)

    print("\n=== Sensor closure/open summary in A/train ===")
    sensor_summary_nonzero = sensor_summary[sensor_summary["closure_steps"] > 0]
    print(sensor_summary_nonzero.to_string(index=False))
    print("=============================================\n")

    # 1) Build closure sensor folds
    folds, sensor_burden = build_closure_sensor_folds_from_args(args, k=5)

    print("\n=== Fold Summary ===")
    for f in folds:
        print(f"Fold {f['fold_id']}: {len(f['heldout_sensors'])} sensors -> {f['heldout_sensors']}")
    print("====================\n")

    # 2) Load original normal split for comparison
    A_train_full, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    print(f"[Original] Train samples: {len(A_train_full)}")

    # 3) Check each fold
    for f in folds:
        print(f"\n\n########## CHECKING FOLD {f['fold_id']} ##########")
        heldout = f["heldout_sensors"]

        A_train_fold, B_val_fold, C_test_fold, C_time_fold, A_adj_fold, mean_fold, std_fold, \
        node_ids_fold, timestamp_list_fold, fold_meta = loadData_hyp2_fold(
            args,
            heldout_sensors=heldout
        )

        print(f"Held-out sensors: {heldout}")
        print(f"Train before : {fold_meta['num_train_before']}")
        print(f"Train after  : {fold_meta['num_train_after']}")
        print(f"Removed      : {fold_meta['num_train_removed']}")

        if fold_meta['num_train_after'] >= fold_meta['num_train_before']:
            print("[WARNING] No training samples were removed.")
        else:
            print("[OK] Fold removed some closure-related training samples.")

    print("\nDone. Fold construction check completed.")

if __name__ == "__main__":
    main()