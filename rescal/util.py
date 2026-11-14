import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os.path as osp
import torch
from scipy.sparse.linalg import eigs
import torch.nn.functional as F
from dataset import mydataset
from tqdm import tqdm
import math

def split_len(n: int, ratios):
    """Convert ratios to integer lengths that sum to n (sequential split)."""
    lengths = [int(r * n) for r in ratios]
    lengths[-1] = n - sum(lengths[:-1])
    return lengths

# In util.py

def loadData_hyp1(args, *, a_ratio=0.8, b_ratio=0.5):
    print("loading the input file(s) [HYP1 - Strict Sequential A->B->C->D->E]")
    
    # 1. Load Data
    A = pd.read_csv(osp.join(args.root, args.adj), index_col=0)
    node_ids = A.index 
    A = np.array(A.astype(np.float32))

    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    
    # Grab all timestamps sequentially
    all_times = df.index.values

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    # 2. Process Features (Time, Construction, Norm)
    time = set_time(df.index.values, df)
    dis = set_dis(df_dis, args.lambda_)
    con = Construction_Layer(torch.from_numpy(np.array(df_con)), torch.from_numpy(np.array(dis)))
    dataframe = np.stack([np.array(df), con.numpy(), time.numpy()])  # (3, T, N)

    mean, std = df.values.mean(), df.values.std()
    X = normalize(dataframe, mean, std)

    # 3. CALCULATE SPLIT INDICES (Strict Sequential)
    num_samples = df.shape[0]
    
    # Original chunk sizes (to maintain the same amount of data as before)
    # Train Chunk (70%) -> split into A (80%) and C (20%)
    len_train_chunk = round(0.7 * num_samples)
    len_A = int(a_ratio * len_train_chunk)
    len_C = len_train_chunk - len_A  # The remaining part of the 70%

    # Val Chunk (10%) -> split into B (50%) and D (50%)
    len_val_chunk = round(0.1 * num_samples) # Actually calculated as remainder in original, let's be precise
    # Replicating original logic to ensure E is exactly the last 20%
    len_test_chunk = round(0.2 * num_samples)
    len_val_chunk  = num_samples - len_train_chunk - len_test_chunk # The gap
    
    len_B = int(b_ratio * len_val_chunk)
    len_D = len_val_chunk - len_B

    # Define strict boundaries
    idx_1 = len_A                   # End of A
    idx_2 = idx_1 + len_B           # End of B (A -> B)
    idx_3 = idx_2 + len_C           # End of C (A -> B -> C)
    idx_4 = idx_3 + len_D           # End of D (A -> B -> C -> D)
    idx_5 = num_samples             # End of E (Rest is E)

    print("\n=== Hyp1 Sequential Split (A->B->C->D->E) ===")
    print(f"Total: {num_samples}")
    print(f"A (Base Train): 0     -> {idx_1}   ({len_A} steps)")
    print(f"B (Base Val):   {idx_1} -> {idx_2}   ({len_B} steps)")
    print(f"C (Cal Train):  {idx_2} -> {idx_3}   ({len_C} steps)")
    print(f"D (Cal Val):    {idx_3} -> {idx_4}   ({len_D} steps)")
    print(f"E (Test):       {idx_4} -> {idx_5}   ({num_samples - idx_4} steps)")
    print("============================================\n")

    # 4. SLICE DATA
    # A
    A_raw = X[:, :idx_1, :]
    df_A  = df.iloc[:idx_1]
    t_A   = all_times[:idx_1]

    # B
    B_raw = X[:, idx_1:idx_2, :]
    df_B  = df.iloc[idx_1:idx_2]
    t_B   = all_times[idx_1:idx_2]

    # C
    C_raw = X[:, idx_2:idx_3, :]
    df_C  = df.iloc[idx_2:idx_3]
    t_C   = all_times[idx_2:idx_3]

    # D
    D_raw = X[:, idx_3:idx_4, :]
    df_D  = df.iloc[idx_3:idx_4]
    t_D   = all_times[idx_3:idx_4]

    # E
    E_raw = X[:, idx_4:, :]
    df_E  = df.iloc[idx_4:]
    t_E   = all_times[idx_4:]

    # 5. GENERATE DATASETS (Cache files)
    A_pack = data_generate(A_raw, args, df_A, 'hyp1_A.pt')
    B_pack = data_generate(B_raw, args, df_B, 'hyp1_B.pt')
    C_pack = data_generate(C_raw, args, df_C, 'hyp1_C.pt')
    D_pack = data_generate(D_raw, args, df_D, 'hyp1_D.pt')
    E_pack = data_generate(E_raw, args, df_E, 'hyp1_E.pt')

    A_ds = mydataset(A_pack[0], A_pack[1])
    B_ds = mydataset(B_pack[0], B_pack[1])
    C_ds = mydataset(C_pack[0], C_pack[1])
    D_ds = mydataset(D_pack[0], D_pack[1])
    E_ds = mydataset(E_pack[0], E_pack[1])
    E_time = mydataset(E_pack[2], E_pack[3])

    # 6. RETURN (including timestamps)
    # The list is explicitly ordered A, B, C, D, E
    timestamp_list = [t_A, t_B, t_C, t_D, t_E]

    return A_ds, B_ds, C_ds, D_ds, E_ds, E_time, A, mean, std, node_ids, timestamp_list

def loadData_hyp2(args, *, train_ratio=0.7, val_ratio=0.1):
    """
    HYP2 (basic):
      - Baseline and ResCAL share the SAME splits.
      - Strict sequential split: A(train) -> B(val) -> C(test)

    Returns:
      A_ds, B_ds, C_ds, C_time, A_adj, mean, std, node_ids, timestamp_list
      where timestamp_list = [t_A, t_B, t_C]
    """
    print("loading the input file(s) [HYP2 - Basic Sequential A->B->C]")

    import pandas as pd
    import numpy as np
    import os.path as osp
    import torch

    # 1) Load adjacency + node ids
    A_adj_df = pd.read_csv(osp.join(args.root, args.adj), index_col=0)
    node_ids = A_adj_df.index
    A_adj = np.array(A_adj_df.astype(np.float32))

    # 2) Load data + timestamps
    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    all_times = df.index.values

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    # 3) Build features: [speed, construction_layer, time_feature], then normalize
    time = set_time(df.index.values, df)
    dis = set_dis(df_dis, args.lambda_)
    con = Construction_Layer(
        torch.from_numpy(np.array(df_con)),
        torch.from_numpy(np.array(dis))
    )
    dataframe = np.stack([np.array(df), con.numpy(), time.numpy()])  # (3, T, N)

    mean, std = df.values.mean(), df.values.std()
    X = normalize(dataframe, mean, std)

    # 4) Strict sequential split indices: A (train), B (val), C (test)
    num_steps = df.shape[0]

    len_train = int(round(train_ratio * num_steps))
    len_val   = int(round(val_ratio * num_steps))
    # ensure the last chunk is exactly the remainder (stable even with rounding)
    len_test  = num_steps - len_train - len_val

    idx_1 = len_train
    idx_2 = idx_1 + len_val
    idx_3 = num_steps

    print("\n=== Hyp2 Sequential Split (A->B->C) ===")
    print(f"Total: {num_steps}")
    print(f"A (Train): 0      -> {idx_1}   ({len_train} steps)")
    print(f"B (Val):   {idx_1} -> {idx_2}   ({len_val} steps)")
    print(f"C (Test):  {idx_2} -> {idx_3}   ({len_test} steps)")
    print("======================================\n")

    # 5) Slice raw arrays + timestamps
    A_raw = X[:, :idx_1, :]
    df_A  = df.iloc[:idx_1]
    t_A   = all_times[:idx_1]

    B_raw = X[:, idx_1:idx_2, :]
    df_B  = df.iloc[idx_1:idx_2]
    t_B   = all_times[idx_1:idx_2]

    C_raw = X[:, idx_2:, :]
    df_C  = df.iloc[idx_2:]
    t_C   = all_times[idx_2:]

    # 6) Generate windowed datasets (cache files)
    A_pack = data_generate(A_raw, args, df_A, "hyp2_A.pt")
    B_pack = data_generate(B_raw, args, df_B, "hyp2_B.pt")
    C_pack = data_generate(C_raw, args, df_C, "hyp2_C.pt")

    A_ds = mydataset(A_pack[0], A_pack[1])
    B_ds = mydataset(B_pack[0], B_pack[1])
    C_ds = mydataset(C_pack[0], C_pack[1])

    # keep same convention as hyp1: a "*_time" dataset for the final split if you use it elsewhere
    C_time = mydataset(C_pack[2], C_pack[3])

    timestamp_list = [t_A, t_B, t_C]

    return A_ds, B_ds, C_ds, C_time, A_adj, mean, std, node_ids, timestamp_list


def loadData(args):

    print("loading the input file(s)")
    # pdb.set_trace()
    A = pd.read_csv(osp.join(args.root, args.adj),index_col=0)
    index = A.index
    A = A.astype(np.float32)
    A = np.array(A)

    #A = torch.from_numpy(np.array(A))

    df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col = 0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col = 0)
    num_input = df.shape[0]
    train_input = round(0.7 * num_input)
    test_input = round(0.2 * num_input)
    val_input = round(0.1 * num_input)
    time = set_time(df.index.values,df)

    dis = set_dis(df_dis, args.lambda_)
    con = Construction_Layer(torch.from_numpy(np.array(df_con)),torch.from_numpy(np.array(dis)))
    dataframe = np.stack([np.array(df),con.numpy(),time.numpy()])

    #all = dataframe[0,:,:]
    mean, std = df.values.mean(), df.values.std()
    X = normalize(dataframe, mean, std)
    train = X[:,:train_input,:]
    val = X[:,train_input: train_input + val_input,:]
    test = X[:,train_input + val_input:, :]


    train_ = data_generate(train, args, df[:train_input], 'train.pt')
    val_  = data_generate(val, args, df[train_input: train_input + val_input], 'val.pt')
    test_ = data_generate(test, args, df[train_input + val_input:], 'test.pt')

    train = mydataset(train_[0],train_[1])
    val = mydataset(val_[0], val_[1])
    test = mydataset(test_[0], test_[1])
    test_time = mydataset(test_[2], test_[3])
    return train,val,test, test_time,  A, mean, std , index

def data_generate(dataset, args, df, save_file):

    cache_dir = osp.join('data', 'processed', args.dataset_choice, args.model_variant)
    os.makedirs(cache_dir, exist_ok=True)
    path = osp.join(cache_dir, save_file)

    if osp.exists(path) is not True:
        time_interval = 300
        dataset = dataset.transpose(1,2,0)
        num_input, num_node, dims = dataset.shape
        num_sample = num_input - args.T - args.P + 1
        index = pd.DataFrame(df.index.values, columns=['time'])
        index.time = pd.to_datetime(index.time)

        n = 0
        x_set = []
        y_set = []
        x_time_set = []
        y_time_set = []
        for i in tqdm(range(num_sample)):
            flag = 0
            x = np.zeros(shape=(args.T, num_node, dims))
            y = np.zeros(shape=(args.P, num_node, dims))
            x_time = np.empty(shape=(args.T), dtype=object)
            y_time = np.empty(shape=(args.P), dtype=object)
            x[0] = dataset[i, :, :]
            x_time[0] = index['time'].iloc[i]
            for j in range(1, args.T + args.P):
                if j <= (args.T - 1):
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == time_interval:
                        x[j] = dataset[i + j, :, :]
                        x_time[j] = index['time'].iloc[i+j]
                    else:
                        flag = 1
                        break
                else:
                    if (index['time'].iloc[i + j] - index['time'].iloc[i + j - 1]).seconds == time_interval:
                        y[j - args.T] = dataset[i + j, :, :]
                        y_time[j - args.T] = index['time'].iloc[i+j]
                    else:
                        flag = 1
                        break

            if flag == 0:
                n = n + 1
                x = x.astype(np.float32)
                y = y.astype(np.float32)
                x_set.append(x)
                y_set.append(y)
                x_time_set.append(x_time)
                y_time_set.append(y_time)

        l = (len(x_set) // args.batch_size) * args.batch_size
        input = x_set[0:l]
        output = y_set[0:l]
        x_t = x_time_set[0:l]
        y_t = y_time_set[0:l]

        torch.save((input, output, x_t, y_t), path)
        data = torch.load(path, weights_only=False)
    else:
        data = torch.load(path, weights_only=False)

    return data



def set_dis(df, lambda_):
    x = 1 - (df / lambda_)**2
    x[x == 1] = 0
    x[x < 0] = 0
    x.values[tuple([np.arange(x.shape[0])[:, np.newaxis]]) * 2] = 1
    return x

def Construction_Layer(x, adj):
    lens, num_of_vertices = x.shape
    set = torch.zeros(size = (lens, num_of_vertices), dtype=torch.float64)
    for i in range(lens):
        a = (x[i] == 1).nonzero()
        for j in range(len(a)):
            e = adj[a[j]]
            set[i] = torch.max(set[i],e)
    return set

def set_time(x,df):
    time_pd = pd.DataFrame(x, columns = ['Times'])
    time_pd['stamp'] = [pd.Timestamp(time) for time in time_pd['Times'].values]
    time_pd['year'] = time_pd['stamp'].apply(lambda x: x.year)
    time_pd['month'] = time_pd['stamp'].apply(lambda x: x.month)
    time_pd['day'] = time_pd['stamp'].apply(lambda x: x.day)
    time_pd['DayOfWeek'] = time_pd['stamp'].apply(lambda d: d.dayofweek)
    time_pd['DayOfYear'] = time_pd['stamp'].apply(lambda d: d.dayofyear)
    time_pd['WeekOfYear'] = time_pd['stamp'].apply(lambda d: d.weekofyear)
    time_pd['Quarter'] = time_pd['stamp'].apply(lambda d: d.quarter)
    time_pd['Hour'] = time_pd['stamp'].apply(lambda d: d.hour)
    time_pd['Minute'] = time_pd['stamp'].apply(lambda d: d.minute)
    cut_hour = [-1, 5, 11, 14, 17, 21,  23]
    cut_labels = ['last night', 'morning', 'Noon', 'afternoon','evening', 'Night']
    time_pd['Hour_cut'] = pd.cut(time_pd['Hour'], bins=cut_hour, labels=cut_labels)
    time = LabelEncoder().fit_transform(time_pd['Hour_cut'])
    set = torch.zeros(size = df.shape, dtype=torch.long)
    for i in range(len(time)):
        set[i] = time[i]
    return set

def normalize(x,mean,std):
    x[0,:,:] = (x[0, :, :] - mean) / std
    return x

def re_normalization(x, mean, std):
    x = x * std + mean
    return x

def RMSE(input, target, min, max, m_factor):
    rmse = torch.sqrt(F.mse_loss(input, target)) * (max - min) / 2. * m_factor
    return rmse

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        # mask = np.not_equal(label, 0)
        # mask = mask.astype(np.float32)
        # mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse)
        result = np.mean(rmse, axis=0)
        b = np.sqrt(np.mean(result, axis = 0))
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape)
        mape = np.mean(mape)

    return mae, rmse, mape, b

def scaled_Laplacian_1(A):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert A.shape[0] == A.shape[1]

    D = np.diag(np.sum(A, axis=1))

    return D, A, np.identity(A.shape[0])

def A_wave(args, A):
    assert A.shape[0] == A.shape[1]
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return torch.from_numpy(A_wave).type(torch.FloatTensor).to(args.device)

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def get_split_indices(num_input: int, train_ratio=0.7, val_ratio=0.1):
    train_end = round(train_ratio * num_input)
    val_len = round(val_ratio * num_input)
    val_end = train_end + val_len
    test_start = val_end
    return train_end, val_end, test_start


def build_seen_unseen_sets(df_con: pd.DataFrame, test_start: int):
    """
    df_con: DataFrame (time x N) with 1 meaning closure on that sensor at that timestamp
    test_start: integer row index where test split begins
    Returns: (seen_sensors_set, unseen_sensors_set)
    """
    # boolean per sensor
    trainval_has = (df_con.iloc[:test_start] == 1).any(axis=0).to_numpy()
    test_has = (df_con.iloc[test_start:] == 1).any(axis=0).to_numpy()

    seen_mask = trainval_has
    unseen_mask = np.logical_and(test_has, np.logical_not(trainval_has))

    seen = set(np.where(seen_mask)[0].tolist())
    unseen = set(np.where(unseen_mask)[0].tolist())
    return seen, unseen


def build_neighbors_from_distance(df_dis: pd.DataFrame, radius_miles: float):
    """
    Returns list neighbors[s] = np.array of sensor indices within radius (excluding itself)
    """
    dis_mat = df_dis.to_numpy(dtype=float)
    N = dis_mat.shape[0]
    neighbors = []
    for s in range(N):
        idx = np.where((dis_mat[s] <= radius_miles) & (np.arange(N) != s))[0]
        neighbors.append(idx)
    return neighbors


def load_cached_times(args, split_file: str):
    cache_path = osp.join('data', 'processed', args.dataset_choice, args.model_variant, split_file)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)

    # Accept either dict or tuple/list
    if isinstance(cached, dict):
        if "y_t_list" in cached:
            return cached["y_t_list"]
        # fallback common keys
        for k in ["y_times", "y_t", "times_y"]:
            if k in cached:
                return cached[k]
        raise KeyError(f"Cached dict missing y_t_list-like key: {cached.keys()}")
    else:
        # assume last item is y_t_list
        *_, y_t_list = cached
        return y_t_list



def event_closure_metrics(
    *,
    net,
    test_loader,
    args,
    mean,
    std,
    df_con: pd.DataFrame,
    df_dis: pd.DataFrame,
    test_start: int,          # <-- NEW (must match your real split boundary)
    radius_miles: float = 3.0,
    y_t_list=None,
):
    if not isinstance(df_con.index, pd.DatetimeIndex):
        df_con = df_con.copy()
        df_con.index = pd.to_datetime(df_con.index)

    seen_sensors, unseen_sensors = build_seen_unseen_sets(df_con, test_start)
    neighbors = build_neighbors_from_distance(df_dis, radius_miles)

    if y_t_list is None:
        y_t_list = load_cached_times(args, "test.pt")

    # alignment check
    assert len(y_t_list) == len(test_loader.dataset), \
        f"y_t_list length ({len(y_t_list)}) != test dataset length ({len(test_loader.dataset)})"

    acc = {
        "seen_all_mae": [], "seen_all_mse": [],
        "seen_loc_mae": [], "seen_loc_mse": [],
        "unseen_all_mae": [], "unseen_all_mse": [],
        "unseen_loc_mae": [], "unseen_loc_mse": [],
    }

    def add_event(prefix: str, err_vec: np.ndarray, neigh_idx: np.ndarray):
        mae_all = float(np.mean(err_vec))
        mse_all = float(np.mean(np.square(err_vec)))
        acc[f"{prefix}_all_mae"].append(mae_all)
        acc[f"{prefix}_all_mse"].append(mse_all)

        if neigh_idx is not None and len(neigh_idx) > 0:
            mae_loc = float(np.mean(err_vec[neigh_idx]))
            mse_loc = float(np.mean(np.square(err_vec[neigh_idx])))
            acc[f"{prefix}_loc_mae"].append(mae_loc)
            acc[f"{prefix}_loc_mse"].append(mse_loc)

    net.eval()
    sample_offset = 0

    with torch.no_grad():
        for (x_test, y_test) in test_loader:
            B = x_test.shape[0]
            x = x_test.to(args.device)
            y = y_test.to(args.device)

            out = net(x)                     # (B, P, N)
            y_norm = y[:, :, :, 0]           # (B, P, N)

            pred  = re_normalization(out.detach().cpu().numpy(), mean, std)
            label = re_normalization(y_norm.detach().cpu().numpy(), mean, std)
            abs_err = np.abs(pred - label)   # (B, P, N)

            for bi in range(B):
                y_times = y_t_list[sample_offset + bi]  # length P
                for p in range(args.P):
                    ts = pd.to_datetime(y_times[p])
                    if ts not in df_con.index:
                        continue
                    closed_vec = (df_con.loc[ts].to_numpy() == 1)
                    closed_sensors = np.where(closed_vec)[0]
                    if closed_sensors.size == 0:
                        continue

                    for s in closed_sensors:
                        if s in unseen_sensors:
                            add_event("unseen", abs_err[bi, p, :], neighbors[s])
                        elif s in seen_sensors:
                            add_event("seen", abs_err[bi, p, :], neighbors[s])

            sample_offset += B

    def summarize(prefix: str):
        all_mae = acc[f"{prefix}_all_mae"]
        all_mse = acc[f"{prefix}_all_mse"]
        loc_mae = acc[f"{prefix}_loc_mae"]
        loc_mse = acc[f"{prefix}_loc_mse"]

        return {
            "events_all": len(all_mae),
            "MAE_all": float(np.mean(all_mae)) if all_mae else float("nan"),
            "RMSE_all": float(math.sqrt(np.mean(all_mse))) if all_mse else float("nan"),
            "events_local": len(loc_mae),
            "MAE_local": float(np.mean(loc_mae)) if loc_mae else float("nan"),
            "RMSE_local": float(math.sqrt(np.mean(loc_mse))) if loc_mse else float("nan"),
        }

    return {
        "radius_miles": radius_miles,
        "seen_sensors_count": len(seen_sensors),
        "unseen_sensors_count": len(unseen_sensors),
        "seen": summarize("seen"),
        "unseen": summarize("unseen"),
    }
