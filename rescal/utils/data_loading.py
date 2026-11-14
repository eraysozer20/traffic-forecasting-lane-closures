# utils/data_loading.py
import os.path as osp
import numpy as np
import pandas as pd
import torch

from dataset import mydataset
from utils.features import set_time, set_dis, Construction_Layer, normalize
from utils.windowing import data_generate

def loadData_hyp2(args, *, train_ratio=0.7, val_ratio=0.1):
    print("loading the input file(s) [HYP2 - Basic Sequential A->B->C]")

    df_dis_for_adj = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    if getattr(args, "adj_type", "binary") == "weighted":
        node_ids = df_dis_for_adj.index
        A_adj = build_weighted_adj_from_distance(
            df_dis_for_adj,
            threshold=getattr(args, "weighted_adj_threshold", 0.1),
        )
    else:
        A_adj_df = pd.read_csv(osp.join(args.root, args.adj), index_col=0)
        node_ids = A_adj_df.index
        A_adj = np.array(A_adj_df.astype(np.float32))

    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    all_times = df.index.values

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    time = set_time(df.index.values, df)
    if getattr(args, "closure_representation", "weighted") == "binary":
        con = torch.from_numpy(np.array(df_con).astype(np.float32))
    else:
        dis = set_dis(df_dis, args.lambda_)
        con = Construction_Layer(
            torch.from_numpy(np.array(df_con)),
            torch.from_numpy(np.array(dis))
        ).float()

    dataframe = np.stack([np.array(df), con.numpy(), time.numpy()])

    mean, std = df.values.mean(), df.values.std()
    X = normalize(dataframe, mean, std)

    num_steps = df.shape[0]
    len_train = int(round(train_ratio * num_steps))
    len_val = int(round(val_ratio * num_steps))
    len_test = num_steps - len_train - len_val

    idx_1 = len_train
    idx_2 = idx_1 + len_val

    print("\n=== Hyp2 Sequential Split (A->B->C) ===")
    print(f"Total: {num_steps}")
    print(f"A (Train): 0      -> {idx_1}   ({len_train} steps)")
    print(f"B (Val):   {idx_1} -> {idx_2}   ({len_val} steps)")
    print(f"C (Test):  {idx_2} -> {num_steps}   ({len_test} steps)")
    print("======================================\n")

    A_raw, df_A, t_A = X[:, :idx_1, :], df.iloc[:idx_1], all_times[:idx_1]
    B_raw, df_B, t_B = X[:, idx_1:idx_2, :], df.iloc[idx_1:idx_2], all_times[idx_1:idx_2]
    C_raw, df_C, t_C = X[:, idx_2:, :], df.iloc[idx_2:], all_times[idx_2:]

    A_pack = data_generate(A_raw, args, df_A, "hyp2_A.pt")
    B_pack = data_generate(B_raw, args, df_B, "hyp2_B.pt")
    C_pack = data_generate(C_raw, args, df_C, "hyp2_C.pt")

    A_ds = mydataset(A_pack[0], A_pack[1])
    B_ds = mydataset(B_pack[0], B_pack[1])
    C_ds = mydataset(C_pack[0], C_pack[1])
    C_time = mydataset(C_pack[2], C_pack[3])

    return A_ds, B_ds, C_ds, C_time, A_adj, mean, std, node_ids, [t_A, t_B, t_C]

def load_y_t_list_cached(args, split_file: str):
    cache_path = osp.join("data", "processed", args.dataset_choice, args.model_variant, split_file)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(cached, dict):
        for k in ["y_t_list", "y_t", "times_y", "y_times"]:
            if k in cached:
                return cached[k]
        raise KeyError(f"Can't find y timestamps in cached dict keys: {cached.keys()}")
    else:
        *_, y_t_list = cached
        return y_t_list


def get_closure_sensor_ids(df_con: pd.DataFrame):
    """
    Returns column indices of sensors that have at least one closure timestamp.
    Assumes df_con shape is (T, N), with columns aligned to sensor order.
    """
    con_arr = df_con.to_numpy()
    closure_mask = (con_arr == 1).any(axis=0)   # (N,)
    closure_sensor_ids = np.where(closure_mask)[0].tolist()
    return closure_sensor_ids


def make_sensor_folds_hybrid(df_con_train, sensor_ids, k=5):
    """
    Build folds that are close in sensor count and more balanced in closure burden.

    Parameters
    ----------
    df_con_train : pd.DataFrame
        Closure dataframe for A/train split only, shape (T_train, N)
    sensor_ids : list[int]
        Sensors that have at least one closure
    k : int
        Number of folds

    Returns
    -------
    folds : list[dict]
        Each dict has:
          - fold_id
          - heldout_sensors
          - total_closure_steps
    """
    con_arr = df_con_train.to_numpy()

    # closure burden per sensor = number of train timesteps with closure
    sensor_burden = {}
    for s in sensor_ids:
        sensor_burden[s] = int((con_arr[:, s] > 0).sum())

    # sort sensors from heaviest to lightest
    sorted_sensors = sorted(sensor_ids, key=lambda s: sensor_burden[s], reverse=True)

    target_size = len(sensor_ids) // k
    remainder = len(sensor_ids) % k

    # example for 50 sensors, 5 folds -> capacities = [10,10,10,10,10]
    capacities = [target_size + (1 if i < remainder else 0) for i in range(k)]

    folds = []
    for i in range(k):
        folds.append({
            "fold_id": i,
            "heldout_sensors": [],
            "total_closure_steps": 0
        })

    for s in sorted_sensors:
        # Prefer folds that still have room
        candidate_folds = [
            f for f in folds
            if len(f["heldout_sensors"]) < capacities[f["fold_id"]]
        ]

        # Among those, choose the one with smallest closure burden so far
        best_fold = min(candidate_folds, key=lambda f: f["total_closure_steps"])

        best_fold["heldout_sensors"].append(s)
        best_fold["total_closure_steps"] += sensor_burden[s]

    return folds, sensor_burden


def build_closure_sensor_folds_from_args(args, k=5):
    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)

    # same train split as loadData_hyp2
    num_steps = df_con.shape[0]
    len_train = int(round(0.7 * num_steps))
    df_con_A = df_con.iloc[:len_train]

    closure_sensor_ids = get_closure_sensor_ids(df_con_A)

    folds, sensor_burden = make_sensor_folds_hybrid(df_con_A, closure_sensor_ids, k=k)

    print(f"Total closure sensors in A/train: {len(closure_sensor_ids)}")
    for f in folds:
        print(
            f"Fold {f['fold_id']}: "
            f"{len(f['heldout_sensors'])} sensors | "
            f"closure_steps={f['total_closure_steps']} | "
            f"sensors={f['heldout_sensors']}"
        )

    return folds, sensor_burden

def build_train_keep_mask_from_heldout_closures_using_times(
    df_con_train: pd.DataFrame,
    x_time_list,
    y_time_list,
    heldout_sensors
):
    """
    Build one boolean per actual training sample, aligned with A_pack_full.

    Remove a sample if ANY held-out sensor has closure at ANY timestamp
    in that sample's input or output window.

    Parameters
    ----------
    df_con_train : pd.DataFrame
        Closure dataframe for A/train split, indexed by timestamps, shape (T, N)
    x_time_list : list
        A_pack_full[2], list of arrays of input timestamps
    y_time_list : list
        A_pack_full[3], list of arrays of output timestamps
    heldout_sensors : list[int]
        Held-out closure sensor indices for this fold
    """
    heldout_sensors = list(heldout_sensors)

    # make sure the closure df index is datetime
    con_df = df_con_train.copy()
    con_df.index = pd.to_datetime(con_df.index)

    keep_mask = []

    for x_times, y_times in zip(x_time_list, y_time_list):
        # combine all timestamps in this sample window
        sample_times = list(x_times) + list(y_times)
        sample_times = pd.to_datetime(sample_times)

        # select closure rows for this exact sample window
        window_con = con_df.loc[sample_times]

        heldout_has_closure = (window_con.iloc[:, heldout_sensors].to_numpy() > 0).any()

        keep_mask.append(not heldout_has_closure)

    return np.array(keep_mask, dtype=bool)


def apply_sample_mask_to_pack(pack, keep_mask):
    """
    Apply sample mask to all parts of a pack returned by data_generate:
      pack[0] = x_set
      pack[1] = y_set
      pack[2] = x_time_set
      pack[3] = y_time_set
    """
    new_pack = []
    for part in pack:
        if len(part) != len(keep_mask):
            raise ValueError(f"Mask length {len(keep_mask)} != part length {len(part)}")
        part_f = [p for p, keep in zip(part, keep_mask) if keep]
        new_pack.append(part_f)
    return tuple(new_pack)

def summarize_sensor_closure_open_counts(df_con_train):
    """
    df_con_train: closure dataframe for A/train, shape (T, N)

    Returns a dataframe with one row per sensor:
      - sensor_idx
      - closure_steps
      - open_steps
      - total_steps
      - closure_ratio
    """
    import pandas as pd
    import numpy as np

    con_arr = df_con_train.to_numpy()
    total_steps = con_arr.shape[0]

    closure_steps = (con_arr > 0).sum(axis=0)
    open_steps = total_steps - closure_steps
    closure_ratio = closure_steps / total_steps

    summary = pd.DataFrame({
        "sensor_idx": np.arange(con_arr.shape[1]),
        "closure_steps": closure_steps,
        "open_steps": open_steps,
        "total_steps": total_steps,
        "closure_ratio": closure_ratio,
    })

    return summary.sort_values(
        by=["closure_steps", "sensor_idx"],
        ascending=[False, True]
    ).reset_index(drop=True)

def build_weighted_adj_from_distance(df_dis: pd.DataFrame, threshold: float = 0.1):
    """
    Build routing-enhanced weighted adjacency:

        A_ij = exp(-(d_ij^2 / sigma^2)) if value >= threshold else 0

    where sigma is the std of valid positive distances.
    """
    D = df_dis.astype(np.float32).to_numpy()

    # Ignore invalid/self distances when computing sigma
    valid = D[np.isfinite(D) & (D > 0)]

    if valid.size == 0:
        raise ValueError("No valid positive distances found in df_dis.")

    sigma = valid.std()

    if sigma <= 0:
        raise ValueError(f"Invalid sigma={sigma}. Check distance matrix.")

    A = np.exp(-((D ** 2) / (sigma ** 2))).astype(np.float32)

    # remove invalid values
    A[~np.isfinite(A)] = 0.0

    # threshold small weights
    A[A < threshold] = 0.0

    # optional: remove self-loops here; A_wave may add/normalize later
    np.fill_diagonal(A, 0.0)

    return A

def loadData_hyp2_fold(args, heldout_sensors, *, train_ratio=0.7, val_ratio=0.1):
    print("loading the input file(s) [HYP2-FOLD - Sequential A->B->C with held-out closure sensors]")

    df_dis_for_adj = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    if getattr(args, "adj_type", "binary") == "weighted":
        node_ids = df_dis_for_adj.index
        A_adj = build_weighted_adj_from_distance(
            df_dis_for_adj,
            threshold=getattr(args, "weighted_adj_threshold", 0.1),
        )
    else:
        A_adj_df = pd.read_csv(osp.join(args.root, args.adj), index_col=0)
        node_ids = A_adj_df.index
        A_adj = np.array(A_adj_df.astype(np.float32))

    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    all_times = df.index.values

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    time = set_time(df.index.values, df)
    if getattr(args, "closure_representation", "weighted") == "binary":
        con = torch.from_numpy(np.array(df_con).astype(np.float32))
    else:
        dis = set_dis(df_dis, args.lambda_)
        con = Construction_Layer(
            torch.from_numpy(np.array(df_con)),
            torch.from_numpy(np.array(dis))
        ).float()

    dataframe = np.stack([np.array(df), con.numpy(), time.numpy()])

    mean, std = df.values.mean(), df.values.std()
    X = normalize(dataframe, mean, std)

    num_steps = df.shape[0]
    len_train = int(round(train_ratio * num_steps))
    len_val = int(round(val_ratio * num_steps))
    len_test = num_steps - len_train - len_val

    idx_1 = len_train
    idx_2 = idx_1 + len_val

    print("\n=== Hyp2 Sequential Split (A->B->C) ===")
    print(f"Total: {num_steps}")
    print(f"A (Train): 0      -> {idx_1}   ({len_train} steps)")
    print(f"B (Val):   {idx_1} -> {idx_2}   ({len_val} steps)")
    print(f"C (Test):  {idx_2} -> {num_steps}   ({len_test} steps)")
    print("======================================\n")

    A_raw, df_A, t_A = X[:, :idx_1, :], df.iloc[:idx_1], all_times[:idx_1]
    B_raw, df_B, t_B = X[:, idx_1:idx_2, :], df.iloc[idx_1:idx_2], all_times[idx_1:idx_2]
    C_raw, df_C, t_C = X[:, idx_2:, :], df.iloc[idx_2:], all_times[idx_2:]

    df_con_A = df_con.iloc[:idx_1]

    A_pack_full = data_generate(A_raw, args, df_A, "hyp2_A.pt")
    B_pack = data_generate(B_raw, args, df_B, "hyp2_B.pt")
    C_pack = data_generate(C_raw, args, df_C, "hyp2_C.pt")

    # A_pack_full = (x_set, y_set, x_time_set, y_time_set)
    x_time_A = A_pack_full[2]
    y_time_A = A_pack_full[3]

    keep_mask = build_train_keep_mask_from_heldout_closures_using_times(
        df_con_A,
        x_time_A,
        y_time_A,
        heldout_sensors
    )

    print(f"len(A_pack_full[0]) = {len(A_pack_full[0])}")
    print(f"len(x_time_A)       = {len(x_time_A)}")
    print(f"len(y_time_A)       = {len(y_time_A)}")
    print(f"len(keep_mask)      = {len(keep_mask)}")

    A_pack = apply_sample_mask_to_pack(A_pack_full, keep_mask)
    
    print(f"[Fold train filtering]")
    print(f"heldout_sensors: {heldout_sensors}")
    print(f"train samples before: {len(A_pack_full[0])}")
    print(f"train samples after : {len(A_pack[0])}")
    print(f"removed samples     : {len(A_pack_full[0]) - len(A_pack[0])}")

    A_ds = mydataset(A_pack[0], A_pack[1])
    B_ds = mydataset(B_pack[0], B_pack[1])
    C_ds = mydataset(C_pack[0], C_pack[1])
    C_time = mydataset(C_pack[2], C_pack[3])

    fold_meta = {
        "heldout_sensors": list(heldout_sensors),
        "num_train_before": len(A_pack_full[0]),
        "num_train_after": len(A_pack[0]),
        "num_train_removed": len(A_pack_full[0]) - len(A_pack[0]),
    }

    return A_ds, B_ds, C_ds, C_time, A_adj, mean, std, node_ids, [t_A, t_B, t_C], fold_meta



