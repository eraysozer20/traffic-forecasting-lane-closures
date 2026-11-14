import os
import os.path as osp
import argparse
from time import process_time
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import TensorDataset

from ResCAL import ResCAL
import util
from STGCN_WZ import model as model_wz
from STGCN_WZ_NO import model as model_no
from util import build_seen_unseen_sets, build_neighbors_from_distance


# =========================
# Args
# =========================
def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--root', type=str, default='data/', help='data root path')
    parser.add_argument('--model_variant', type=str, default='wz', choices=['wz', 'no'])

    parser.add_argument('--dataset_choice', type=str, default='Tyson',
                        choices=['Tyson', 'PEMSD8', 'SIM_GRID'])

    parser.add_argument('--adj', type=str, default=None)
    parser.add_argument('--adjdata', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--con_data', type=str, default=None)
    parser.add_argument('--dis_data', type=str, default=None)
    parser.add_argument('--num_nodes', type=int, default=None)
    parser.add_argument('--lambda_', type=float, default=None)

    parser.add_argument('--T', type=int, default=12)
    parser.add_argument('--P', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--feature', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=1)

    # Hyp1 split knobs (optional, for later tuning)
    parser.add_argument('--hyp1_a_ratio', type=float, default=0.8, help="A portion of train chunk")
    parser.add_argument('--hyp1_b_ratio', type=float, default=0.5, help="B portion of val chunk")

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args



DATASET_CFG = {
    "Tyson": dict(
        adj="Tyson/adj.csv",
        data="Tyson/new_speed_5 - order.csv",
        con_data="Tyson/new_construction_5 - order.csv",
        dis_data="Tyson/adj_1.csv",
        num_nodes=131,
        lambda_=3.0,
    ),
    "PEMSD8": dict(
        adj="PEMS08/adj.csv",
        data="PEMS08/speed_cut.csv",
        con_data="PEMS08/closure_cut.csv",
        dis_data="PEMS08/adj_1_miles.csv",
        num_nodes=212,
        lambda_=18.0,
    ),
    "SIM_GRID": dict(
        adj="SIM/GRID/adj.csv",
        data="SIM/GRID/sensor_readings.csv",
        con_data="SIM/GRID/closure_labels.csv",
        dis_data="SIM/GRID/adj_1_miles.csv",
        num_nodes=120,
        lambda_=1.1,
    ),
}


def apply_dataset_cfg(args):
    cfg = DATASET_CFG[args.dataset_choice]
    for k, v in cfg.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


# =========================
# Train / Val helpers
# =========================
def train_epoch(net, loader, optimizer, criterion, mean, std, args, epoch, log_every=50):
    net.train()
    losses = []

    train_pred = []
    train_label = []

    for i, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()

        out = net(x)                      # (B, P, N)
        loss = criterion(out, y[:, :, :, 0])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # accumulate for metrics
        pred = util.re_normalization(out.detach().cpu().numpy(), mean, std)
        label = util.re_normalization(
            y[:, :, :, 0].detach().cpu().numpy(), mean, std
        )

        train_pred.append(pred)
        train_label.append(label)

        # ---- per-batch logging ----
        if (i + 1) % log_every == 0:
            p = np.concatenate(train_pred, axis=0)
            l = np.concatenate(train_label, axis=0)
            mae, rmse, _, _ = util.metric(p, l)
            print(
                f"[Epoch {epoch+1:03d}] "
                f"[{i+1:04d}/{len(loader)}] "
                f"loss={loss.item():.4f} "
                f"mae={mae:.4f} "
                f"rmse={rmse:.4f}"
            )

    return float(np.mean(losses))



@torch.no_grad()
def eval_rmse(net, loader, mean, std, args):
    net.eval()
    preds, labels = [], []

    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)

        out = net(x)
        pred = util.re_normalization(out.cpu().numpy(), mean, std)
        label = util.re_normalization(y[:, :, :, 0].cpu().numpy(), mean, std)

        preds.append(pred)
        labels.append(label)

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    _, rmse, _, _ = util.metric(preds, labels)
    return rmse

def generate_time_features(timestamps, num_nodes, T):
    """
    Generates TimeOfDay and DayOfWeek features for the sequence X.
    timestamps: List of timestamp strings for the 'current' time (t) of each sample.
    Returns: Tensor of shape (Samples, T, N, 2) -> [TimeOfDay, DayOfWeek]
    """
    import pandas as pd
    
    # Convert to datetime objects
    dt = pd.to_datetime(timestamps)
    
    # 1. Time of Day (Normalized 0-1)
    # Minute of day / 1440
    minutes = dt.hour * 60 + dt.minute
    tod = minutes / 1440.0
    
    # 2. Day of Week (Normalized 0-1)
    # Monday=0, Sunday=6 -> /6.0
    dow = dt.dayofweek / 6.0
    
    # These are for the 'current' step t. 
    # But X is a sequence of length T [t-11, ..., t]. 
    # We need to construct the history.
    
    num_samples = len(timestamps)
    
    # Create placeholders
    # Shape: (Samples, T, 2)
    time_feature_seq = np.zeros((num_samples, T, 2), dtype=np.float32)
    
    # We can perform a broadcast subtraction to handle the window lag
    # "Current time" is at index T-1.
    # Previous time is current - 5 minutes.
    # We'll approximate this by just subtracting normalized time steps (5 mins = 5/1440)
    # This avoids complex date math for every step.
    
    step_size = 5.0 / 1440.0  # Assumes 5-minute intervals
    
    tod_np = tod.values # (Samples,)
    dow_np = dow.values # (Samples,)
    
    for i in range(T):
        # Time at step i (relative to end of window)
        # i=0 is (T-1) steps ago. i=T-1 is Now.
        lag = (T - 1) - i
        
        # Subtract lag from Time of Day
        # Handle wrapping (negative values) using modulo 1.0
        time_feature_seq[:, i, 0] = (tod_np - (lag * step_size)) % 1.0
        
        # For Day of Week, we simplify and assume it's constant for the 1-hour window
        # (It only changes rarely).
        time_feature_seq[:, i, 1] = dow_np

    # Now expand to Nodes
    # (Samples, T, 2) -> (Samples, T, 1, 2) -> (Samples, T, N, 2)
    time_feature_seq = torch.from_numpy(time_feature_seq).unsqueeze(2).expand(-1, -1, num_nodes, -1)
    
    return time_feature_seq

@torch.no_grad()
def build_residual_dataset(
    base_model,
    source_loader,
    mean, std,
    args,
    timestamps=None,
    cache_path=None,
):
    if cache_path is not None and osp.exists(cache_path):
        pack = torch.load(cache_path, map_location="cpu", weights_only=False)
        # require these to exist; otherwise rebuild
        need = ["X", "U", "R", "BASE", "Y"]
        if all(k in pack for k in need):
            return TensorDataset(pack["X"], pack["U"], pack["R"], pack["BASE"], pack["Y"])
        else:
            print("[BuildResid] Cache exists but missing BASE/Y -> rebuilding:", cache_path)


    base_model.eval()
    X_list, R_list = [], []
    BASE_list, Y_list = [], []



    # 1. Compute Raw Targets (Future Residuals)
    for x, y in source_loader:
        x = x.to(args.device)
        y = y.to(args.device)

        y_norm = y[:, :, :, 0]
        base_out = base_model(x)                 # (B, P, N) normalized
        r_norm = y_norm - base_out               # (B, P, N) normalized residual target

        X_list.append(x.detach().cpu())
        R_list.append(r_norm.detach().cpu())
        BASE_list.append(base_out.detach().cpu())  # (B,P,N) normalized
        Y_list.append(y_norm.detach().cpu())      # (B,P,N) normalized



    X = torch.cat(X_list, dim=0)        # (Samples, Tin, N, F)
    R = torch.cat(R_list, dim=0)        # (Samples, Tout, N)
    R = R.contiguous() 
    BASE = torch.cat(BASE_list, dim=0).contiguous()   # (S, P, N) normalized
    Y    = torch.cat(Y_list, dim=0).contiguous()     # (S,P,N)


    # 2. INJECT TIME FEATURES (Fix X)
    # ------------------------------------------------
    if timestamps is not None:
        # Generate [TimeOfDay, DayOfWeek]
        # Ensure we take the same number of timestamps as samples (some loaders drop last batch)
        valid_timestamps = timestamps[:X.shape[0]]
        
        print(f"[BuildResid] Generating time features for {len(valid_timestamps)} samples...")
        time_feats = generate_time_features(valid_timestamps, args.num_nodes, args.T)
        
        speed = X[:, :, :, 0:1] # (Samples, T, N, 1)
        
        # Concatenate Speed + New Time Features
        X = torch.cat([speed, time_feats], dim=-1) # (Samples, T, N, 3)
        
        print("[BuildResid] X updated with generated TimeOfDay and DayOfWeek.")

    X = X.permute(0, 3, 2, 1).contiguous()

    # 3. Construct Input Sequence U (Method A)
    # ------------------------------------------------
    num_samples = X.shape[0]
    Tin = args.T
    P = args.P
    N = args.num_nodes
    
    pad_len = Tin + P
    R_padded = torch.cat([torch.zeros(pad_len, P, N), R], dim=0)

    U_list = []

    for i in range(num_samples):
        current_sample_idx_padded = i + pad_len
        sequence_u = []
        for t in range(Tin):
            global_t_idx = current_sample_idx_padded - (Tin - 1) + t
            step_u = []
            for k in range(P):
                val = R_padded[global_t_idx - (k + 1), k, :]
                step_u.append(val)
            seq_step = torch.stack(step_u) 
            sequence_u.append(seq_step)
        
        U_sample = torch.stack(sequence_u)
        U_sample = U_sample.permute(0, 2, 1) 
        U_list.append(U_sample)

    U = torch.stack(U_list)
    U = U.permute(0, 3, 2, 1).contiguous()

    print(f"[ResidualDataset] X (ResCAL): {tuple(X.shape)}  expected (S, 3, N, Tin)")
    print(f"[ResidualDataset] U (ResCAL): {tuple(U.shape)}  expected (S, P, N, Tin) where Tu=P")
    print(f"[ResidualDataset] R (Target): {tuple(R.shape)}  expected (S, P, N)")


    if cache_path is not None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save({"X": X, "U": U, "R": R, "BASE": BASE, "Y": Y}, cache_path)

    return TensorDataset(X, U, R, BASE, Y)



def inspect_rescal_inputs(dataset, timestamps=None, num_samples=5, args=None):
    """
    Visualizes the PRE-BUILT Residual Dataset (X, U, R) after reshaping for ResCAL.

    Expected shapes per sample:
      X_sample: (Cx=3, N, Tin)   -> [Speed, TimeOfDay, DayOfWeek]
      U_sample: (Tu=P, N, Tin)   -> past-error channels (Method A)
      R_sample: (P, N)           -> target residuals
    """
    import numpy as np

    print(f"\n{'='*20} INSPECTING DATASET CONTENT (NODE 0) {'='*20}")

    Tin = args.T
    P = args.P

    for i in range(min(num_samples, len(dataset))):
        x_sample, u_sample, r_sample, base_sample, y_sample = dataset[i]

        # Timestamp
        t_str = "N/A"
        if timestamps is not None and i < len(timestamps):
            t_str = timestamps[i]

        # ---- Node 0 extraction ----
        # X: (3, N, Tin) -> node0 => (3, Tin)
        x_node0 = x_sample[:, 0, :].numpy()
        speed = x_node0[0]   # (Tin,)
        tod   = x_node0[1]
        dow   = x_node0[2]

        # U: (P, N, Tin) -> node0 => (P, Tin)
        u_node0 = u_sample[:, 0, :].numpy()

        # R: (P, N) -> node0 => (P,)
        r_node0 = r_sample[:, 0].numpy()

        base_node0 = base_sample[:, 0].numpy()   # (P,)

        y_node0    = y_sample[:, 0].numpy()     # (P,)






        # ---- Printing ----
        print(f"\n[Sample {i}] Timestamp: {t_str}")
        print(f"  X speed      : {np.round(speed, 4)}")
        print(f"  X timeofday  : {np.round(tod, 4)}")
        print(f"  X dayofweek  : {np.round(dow, 4)}")
        print(f"  BASE pred (node0): {np.round(base_node0, 4)}")
        print(f"  Y gt     (node0): {np.round(y_node0, 4)}")
        print(f"  Check Y-BASE == R: {np.round(y_node0 - base_node0, 4)}")
        print(f"  Target Resid (R) node0: {np.round(r_node0, 4)}")

        print(f"  --- Input Sequence (Last 3 Steps of Tin={Tin}) ---")
        for step in range(Tin - 3, Tin):
            u_vec = u_node0[:, step]  # (P,)

            print(f"    Step idx={step}  (time in window)")
            print(f"      X[Speed]     : {float(speed[step]):.4f}")
            print(f"      X[TimeOfDay] : {float(tod[step]):.4f}")
            print(f"      X[DayOfWeek] : {float(dow[step]):.4f}")
            print(f"      U[PastErr] (len P={P}): {np.round(u_vec, 4)}")

            concat = np.concatenate([[speed[step], tod[step], dow[step]], u_vec])
            print(f"      Concat In (3+P): {np.round(concat, 4)}")


def train_rescal_epoch(rescal, loader, support, optimizer, criterion, args, epoch, log_every=50):
    rescal.train()
    losses = []

    for i, (X, U, R, BASE, Y) in enumerate(loader):
        X = X.to(args.device)   # (B, 3, N, Tin)
        U = U.to(args.device)   # (B, P, N, Tin)
        R = R.to(args.device)   # (B, P, N)

        optimizer.zero_grad()
        R_hat, _ = rescal(X, U, support)          # (B, P, N)
        loss = criterion(R_hat, R)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % log_every == 0:
            rmse = torch.sqrt(torch.mean((R_hat - R) ** 2)).item()
            print(f"[ResCAL][Epoch {epoch+1:03d}][{i+1:04d}/{len(loader)}] loss={loss.item():.4f} rmse={rmse:.4f}")

    return float(np.mean(losses))


@torch.no_grad()
def eval_rescal_rmse(rescal, loader, support, args):
    rescal.eval()
    se_sum = 0.0
    n_sum = 0

    for X, U, R, BASE, Y in loader:
        X = X.to(args.device)
        U = U.to(args.device)
        R = R.to(args.device)

        R_hat, _ = rescal(X, U, support)
        se_sum += torch.sum((R_hat - R) ** 2).item()
        n_sum += R.numel()

    return float(np.sqrt(se_sum / max(n_sum, 1)))


@torch.no_grad()
def test_rescal_on_E(rescal, loader_E_resid, support, mean, std, args):
    """
    loader yields: X, U, R, BASE, Y
      BASE and Y are normalized (same scale as baseline output)
    """
    rescal.eval()

    preds_final, preds_base, labels = [], [], []

    for X, U, R, BASE, Y in loader_E_resid:
        X = X.to(args.device)       # (B,3,N,Tin)
        U = U.to(args.device)       # (B,P,N,Tin)
        BASE = BASE.to(args.device) # (B,P,N)
        Y = Y.to(args.device)       # (B,P,N)

        R_hat, _ = rescal(X, U, support)          # (B,P,N)
        Y_hat = BASE + R_hat                      # (B,P,N) normalized

        # denormalize to real scale for metrics
        pred_final = util.re_normalization(Y_hat.detach().cpu().numpy(), mean, std)
        pred_base  = util.re_normalization(BASE.detach().cpu().numpy(), mean, std)
        label      = util.re_normalization(Y.detach().cpu().numpy(), mean, std)

        preds_final.append(pred_final)
        preds_base.append(pred_base)
        labels.append(label)

    preds_final = np.concatenate(preds_final, axis=0)
    preds_base  = np.concatenate(preds_base, axis=0)
    labels      = np.concatenate(labels, axis=0)

    mae_f, rmse_f, mape_f, _ = util.metric(preds_final, labels)
    mae_b, rmse_b, mape_b, _ = util.metric(preds_base, labels)

    return {
        "final": (mae_f, rmse_f, mape_f),
        "base":  (mae_b, rmse_b, mape_b),
    }

import math
import pandas as pd
import numpy as np

@torch.no_grad()
def collect_preds_on_E_resid(rescal, loader_E_resid, support, mean, std, args):
    """Return denorm arrays: pred_base, pred_final, label, each (S,P,N)."""
    preds_base, preds_final, labels = [], [], []
    rescal.eval()

    for X, U, R, BASE, Y in loader_E_resid:
        X = X.to(args.device)
        U = U.to(args.device)
        BASE = BASE.to(args.device)
        Y = Y.to(args.device)

        R_hat, _ = rescal(X, U, support)   # (B,P,N)
        Y_hat = BASE + R_hat               # (B,P,N) normalized

        preds_base.append(util.re_normalization(BASE.cpu().numpy(), mean, std))
        preds_final.append(util.re_normalization(Y_hat.cpu().numpy(), mean, std))
        labels.append(util.re_normalization(Y.cpu().numpy(), mean, std))

    return (
        np.concatenate(preds_base, axis=0),
        np.concatenate(preds_final, axis=0),
        np.concatenate(labels, axis=0),
    )


def event_closure_metrics_from_arrays(
    *,
    preds: np.ndarray,     # (S,P,N) denorm
    labels: np.ndarray,    # (S,P,N) denorm
    df_con: pd.DataFrame,
    df_dis: pd.DataFrame,
    test_start: int,
    radius_miles: float,
    y_t_list,              # list length S, each length P
    P: int,
):
    # datetime index
    if not isinstance(df_con.index, pd.DatetimeIndex):
        df_con = df_con.copy()
        df_con.index = pd.to_datetime(df_con.index)

    seen_sensors, unseen_sensors = build_seen_unseen_sets(df_con, test_start)
    neighbors = build_neighbors_from_distance(df_dis, radius_miles)

    assert len(y_t_list) == preds.shape[0], f"y_t_list ({len(y_t_list)}) != num samples ({preds.shape[0]})"

    abs_err = np.abs(preds - labels)  # (S,P,N)

    acc = {
        "seen_all_mae": [], "seen_all_mse": [],
        "seen_loc_mae": [], "seen_loc_mse": [],
        "unseen_all_mae": [], "unseen_all_mse": [],
        "unseen_loc_mae": [], "unseen_loc_mse": [],
    }

    def add_event(prefix: str, err_vec: np.ndarray, neigh_idx: np.ndarray):
        acc[f"{prefix}_all_mae"].append(float(np.mean(err_vec)))
        acc[f"{prefix}_all_mse"].append(float(np.mean(np.square(err_vec))))
        if neigh_idx is not None and len(neigh_idx) > 0:
            acc[f"{prefix}_loc_mae"].append(float(np.mean(err_vec[neigh_idx])))
            acc[f"{prefix}_loc_mse"].append(float(np.mean(np.square(err_vec[neigh_idx]))))

    for i in range(preds.shape[0]):
        y_times = y_t_list[i]
        for p in range(P):
            ts = pd.to_datetime(y_times[p])
            if ts not in df_con.index:
                continue

            closed_vec = (df_con.loc[ts].to_numpy() == 1)
            closed_sensors = np.where(closed_vec)[0]
            if closed_sensors.size == 0:
                continue

            for s in closed_sensors:
                if s in unseen_sensors:
                    add_event("unseen", abs_err[i, p, :], neighbors[s])
                elif s in seen_sensors:
                    add_event("seen", abs_err[i, p, :], neighbors[s])

    def summarize(prefix: str):
        all_mae = acc[f"{prefix}_all_mae"]; all_mse = acc[f"{prefix}_all_mse"]
        loc_mae = acc[f"{prefix}_loc_mae"]; loc_mse = acc[f"{prefix}_loc_mse"]
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

def load_hyp1_y_t_list(args, split_file: str):
    cache_path = osp.join("data", "processed", args.dataset_choice, args.model_variant, split_file)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)

    # data_generate commonly returns (x, y, x_t, y_t)
    if isinstance(cached, dict):
        # if you ever saved as dict
        for k in ["y_t_list", "y_t", "times_y"]:
            if k in cached:
                return cached[k]
        raise KeyError(f"Can't find y timestamps in cached keys: {cached.keys()}")
    else:
        *_, y_t_list = cached
        return y_t_list

# =========================
# Main
# =========================
def main():
    args = build_args()
    apply_dataset_cfg(args)

    print("CONFIG:", args.dataset_choice, args.model_variant)
    print("Device:", args.device)

    # =========================
    # Hyp1 data
    # =========================
    # UPDATED UNPACKING: Accepts 'node_ids' and 'timestamp_list' at the end
    A_train, B_val, C_cal, D_cal, E_test, E_time, A, mean, std, node_ids, timestamp_list = \
        util.loadData_hyp1(
            args,
            a_ratio=args.hyp1_a_ratio,
            b_ratio=args.hyp1_b_ratio,
        )

    train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
    val_loader_B   = DataLoader(B_val,   batch_size=args.batch_size, shuffle=False)

    cal_train_loader_C = DataLoader(dataset=C_cal, batch_size=args.batch_size, shuffle=False)
    cal_val_loader_D   = DataLoader(dataset=D_cal,   batch_size=args.batch_size, shuffle=False)


    # =========================
    # Model
    # =========================
    A_wave = util.A_wave(args, A)
    build_model = model_wz if args.model_variant == 'wz' else model_no
    net = build_model(args, A_wave).to(args.device)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = nn.MSELoss().to(args.device)

    # =========================
    # Training loop (EARLY STOP = 8)
    # =========================
    save_dir = osp.join("save", args.dataset_choice, args.model_variant)
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "baseline_params.pt")

    # Resume logic
    initepoch = 0
    best_rmse = np.inf
    stop = 0

    if osp.exists(save_path):
        print("[INFO] Loading the baseline checkpoint...")
        checkpoint = torch.load(save_path, map_location=args.device, weights_only=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        initepoch = int(checkpoint.get("epoch", 0))
        best_rmse = float(checkpoint.get("best_rmse", np.inf))
    else:
        print("[INFO] No baseline checkpoint found. Training from scratch.")


    for epoch in range(initepoch, args.epochs):
        scheduler.step()

        train_loss = train_epoch(
            net,
            train_loader_A,
            optimizer,
            criterion,
            mean,
            std,
            args,
            epoch,
            log_every=50,
        )

        val_rmse = eval_rmse(
            net, val_loader_B, mean, std, args
        )

        print(f"[Epoch {epoch+1:03d}] train_loss={train_loss:.4f} | val_RMSE={val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            stop = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                },
                save_path
            )
            print(f"  New best baseline RMSE: {best_rmse:.4f}")
        else:
            stop += 1
            if stop >= 8:
                print("Early stopping triggered (patience = 8)")
                break

    print("\n[BASELINE DONE]")
    print("Best val RMSE:", best_rmse)
    print("Saved to:", save_path)

    # =========================
    # Step 3: Load best baseline + freeze
    # =========================
    ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    print("[Step3] Baseline loaded + frozen from:", save_path)

    # =========================
    # Step 3: Build residual datasets for C and D
    # =========================
    res_cache_dir = osp.join("data", "processed", args.dataset_choice, args.model_variant, "hyp1_residuals")
    os.makedirs(res_cache_dir, exist_ok=True)
    resC_cache = osp.join(res_cache_dir, "residual_C.pt")
    resD_cache = osp.join(res_cache_dir, "residual_D.pt")

    # 1. BUILD C (and inject time features inside build_residual_dataset)
    timestamps_C = timestamp_list[2]
    
    residual_ds_C = build_residual_dataset(
        base_model=net,
        source_loader=cal_train_loader_C,
        mean=mean, std=std,
        args=args,
        timestamps=timestamps_C, # <--- Pass timestamps here to fix X
        cache_path=resC_cache,
    )

    # 2. INSPECT C (using the dataset we just built)
    print("First 3 Timestamps in C:", timestamps_C[:3])
    
    inspect_rescal_inputs(
        dataset=residual_ds_C,  # <--- Pass the dataset object
        timestamps=timestamps_C, 
        num_samples=10, 
        args=args
    )

    # 3. BUILD D
    timestamps_D = timestamp_list[3]
    residual_ds_D = build_residual_dataset(
        base_model=net,
        source_loader=cal_val_loader_D,
        mean=mean, std=std,
        args=args,
        timestamps=timestamps_D,
        cache_path=resD_cache,
    )

    # 4. Create Loaders
    resid_loader_C = DataLoader(residual_ds_C, batch_size=args.batch_size, shuffle=True)
    resid_loader_D = DataLoader(residual_ds_D, batch_size=args.batch_size, shuffle=False)

    # sanity check
    x_r, u_r, r_r, base_r, y_r = next(iter(resid_loader_C))

    
    print(f"[Sanity] Residual loader batch")
    print(f"  X (Speed Input):     {tuple(x_r.shape)}")
    print(f"  U (ResCal Input):    {tuple(u_r.shape)}")
    print(f"  R (ResCal Target):   {tuple(r_r.shape)}")
    print(f"  BASE (Baseline Pred): {tuple(base_r.shape)}")
    print(f"  Y (Ground Truth):     {tuple(y_r.shape)}")

    A_wave = A_wave.to(args.device).float()
    support = [A_wave]



    rescal = ResCAL(
        num_nodes=args.num_nodes,
        Tu=args.P,          # in your setup Tu = P (because U has P channels)
        out_dim=args.P,     # predict P residual horizons
        layers=4,           # <-- k=4
        dc=32,
        nc=16,
        de=16,
        tau=1.0,
        dropout=0.0,
        support_len=1,      # because support=[A_wave]
        gcn_order=1,
    ).to(args.device)

    print("[ResCAL] params:", sum(p.numel() for p in rescal.parameters() if p.requires_grad))

    rescal_opt = optim.Adam(rescal.parameters(), lr=args.learning_rate)
    rescal_sched = lr_scheduler.StepLR(rescal_opt, step_size=40, gamma=0.8)
    rescal_crit = nn.MSELoss().to(args.device)

    save_dir = osp.join("save", args.dataset_choice, args.model_variant)
    os.makedirs(save_dir, exist_ok=True)
    rescal_path = osp.join(save_dir, "rescal_params.pt")

    best_val = np.inf
    stop = 0

    print("\n[STEP4] Training ResCAL on residuals (C=train, D=val)")
    for epoch in range(args.epochs):
        rescal_sched.step()

        tr_loss = train_rescal_epoch(
            rescal, resid_loader_C, support, rescal_opt, rescal_crit, args, epoch, log_every=50
        )
        val_rmse = eval_rescal_rmse(rescal, resid_loader_D, support, args)

        print(f"[ResCAL][Epoch {epoch+1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            stop = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rescal.state_dict(),
                    "opt_state_dict": rescal_opt.state_dict(),
                    "best_val_rmse": best_val,
                },
                rescal_path,
            )
            print(f"  New best ResCAL val RMSE: {best_val:.4f}")
        else:
            stop += 1
            if stop >= 8:
                print("  ResCAL early stopping (patience=8)")
                break

    print("[STEP4 DONE] Best ResCAL val RMSE:", best_val)
    print("Saved:", rescal_path)


    # =========================
    # Step 5: TEST on E
    # =========================
    print("\n[STEP5] Building E residual dataset + testing...")

    # load best rescal
    ck = torch.load(rescal_path, map_location=args.device, weights_only=False)
    rescal.load_state_dict(ck["model_state_dict"])
    rescal.eval()

    # Build E residual dataset (needs timestamps for X time features)
    timestamps_E = timestamp_list[4]
    resE_cache = osp.join(res_cache_dir, "residual_E.pt")

    E_loader_raw = DataLoader(dataset=E_test, batch_size=args.batch_size, shuffle=False)

    residual_ds_E = build_residual_dataset(
        base_model=net,
        source_loader=E_loader_raw,
        mean=mean, std=std,
        args=args,
        timestamps=timestamps_E,
        cache_path=resE_cache,
    )

    E_loader_resid = DataLoader(residual_ds_E, batch_size=args.batch_size, shuffle=False)

    # run test
    test_out = test_rescal_on_E(rescal, E_loader_resid, support, mean, std, args)

    mae_b, rmse_b, mape_b = test_out["base"]
    mae_f, rmse_f, mape_f = test_out["final"]

    print("\n========== TEST RESULTS (E) ==========")
    print(f"[BASELINE ] MAE={mae_b:.4f} RMSE={rmse_b:.4f} MAPE={mape_b:.4f}")
    print(f"[BASE+Res ] MAE={mae_f:.4f} RMSE={rmse_f:.4f} MAPE={mape_f:.4f}")
    print("=====================================\n")

        # =========================
    # Step 6: EVENT METRICS on E (seen/unseen) for BASE and BASE+ResCAL
    # =========================
    print("\n[STEP6] Event-based metrics on E (seen/unseen)...")

    import pandas as pd

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)
    if not isinstance(df_con.index, pd.DatetimeIndex):
        df_con.index = pd.to_datetime(df_con.index)

    # hyp1 boundary (start row of E in the original dataframe timeline)
    idx_4 = len(timestamp_list[0]) + len(timestamp_list[1]) + len(timestamp_list[2]) + len(timestamp_list[3])
    test_start = idx_4

    # collect denorm preds from residual loader (BASE and FINAL)
    pred_base_E, pred_final_E, label_E = collect_preds_on_E_resid(
        rescal, E_loader_resid, support, mean, std, args
    )

    # Load per-sample future timestamps for E from cached hyp1_E.pt
    y_t_E_full = load_hyp1_y_t_list(args, "hyp1_E.pt")
    y_t_E = y_t_E_full[:pred_base_E.shape[0]]  # align AFTER preds exist


    # baseline event metrics (using arrays)
    event_base = event_closure_metrics_from_arrays(
        preds=pred_base_E,
        labels=label_E,
        df_con=df_con,
        df_dis=df_dis,
        test_start=test_start,
        radius_miles=3.0,
        y_t_list=y_t_E,
        P=args.P,
    )

    # final event metrics (using arrays)
    event_final = event_closure_metrics_from_arrays(
        preds=pred_final_E,
        labels=label_E,
        df_con=df_con,
        df_dis=df_dis,
        test_start=test_start,
        radius_miles=3.0,
        y_t_list=y_t_E,
        P=args.P,
    )

    def _print_event(title, event_res):
        print(f"\n========== {title} ==========")
        print(f"[INFO] seen sensors: {event_res['seen_sensors_count']}, unseen sensors: {event_res['unseen_sensors_count']}, radius={event_res['radius_miles']} miles")
        print(f"[SEEN]   events(all)={event_res['seen']['events_all']}, MAE_all={event_res['seen']['MAE_all']:.4f}, RMSE_all={event_res['seen']['RMSE_all']:.4f}")
        print(f"[SEEN]   events(loc)={event_res['seen']['events_local']}, MAE_loc={event_res['seen']['MAE_local']:.4f}, RMSE_loc={event_res['seen']['RMSE_local']:.4f}")
        print(f"[UNSEEN] events(all)={event_res['unseen']['events_all']}, MAE_all={event_res['unseen']['MAE_all']:.4f}, RMSE_all={event_res['unseen']['RMSE_all']:.4f}")
        print(f"[UNSEEN] events(loc)={event_res['unseen']['events_local']}, MAE_loc={event_res['unseen']['MAE_local']:.4f}, RMSE_loc={event_res['unseen']['RMSE_local']:.4f}")
        print("=====================================\n")

    _print_event("EVENT METRICS (BASELINE on E)", event_base)
    _print_event("EVENT METRICS (BASE + ResCAL on E)", event_final)








if __name__ == "__main__":
    main() 