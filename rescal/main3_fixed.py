import os
import os.path as osp
import json
import random
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from ResCAL import ResCAL
from STGCN_WZ import model as model_wz
from STGCN_WZ_NO import model as model_no

from utils.cli import build_args
from utils.configs import apply_dataset_cfg
from utils.data_loading import (
    loadData_hyp2,
    loadData_hyp2_fold,
    load_y_t_list_cached,
    build_closure_sensor_folds_from_args,
)
from utils.features import A_wave
from utils.training import (
    train_epoch,
    eval_rmse,
    train_rescal_epoch,
    eval_rescal_rmse,
)
from utils.residuals import (
    build_residual_dataset,
    find_heldout_closure_events,
    expand_and_merge_events,
    build_sample_mask_from_expanded_idx_intervals,
)
from utils.metrics import (
    collect_preds_on_E_resid,
    event_closure_metrics_from_arrays,
    metric,
)
from utils.printing import (
    print_test_results_block,
    print_event_metrics_block,
)
from utils.local_rescal_prep import (
    compute_weekly_mu_train_nonclosure_simple,
    weekly_mu_for_timestamps,
)


# ============================================================
# Ablation configs
# ============================================================
ABLATION_CONFIGS = {
    "baseline_no": {
        "run_rescal": False,
        "use_extra_folds": False,
        "use_closure_channel": False,
        "use_weekly_residual": True,
    },
    "baseline": {
        "run_rescal": False,
        "use_extra_folds": False,
        "use_closure_channel": True,
        "use_weekly_residual": True,
    },
    "rescal_base_only": {
        "run_rescal": True,
        "use_extra_folds": False,
        "use_closure_channel": True,
        "use_weekly_residual": True,
    },
    "rescal_base_plus_fold": {
        "run_rescal": True,
        "use_extra_folds": True,
        "use_closure_channel": True,
        "use_weekly_residual": True,
    },
    "rescal_base_plus_fold_no_closure": {
        "run_rescal": True,
        "use_extra_folds": True,
        "use_closure_channel": False,
        "use_weekly_residual": True,
    },
    "rescal_base_plus_fold_no_weekly": {
        "run_rescal": True,
        "use_extra_folds": True,
        "use_closure_channel": True,
        "use_weekly_residual": False,
    },
}


# ============================================================
# Generic helpers
# ============================================================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_ablation_args(args):
    if not hasattr(args, "ablation_modes"):
        args.ablation_modes = [
            "baseline",
            "baseline_no",
            "rescal_base_only",
            "rescal_base_plus_fold",
            "rescal_base_plus_fold_no_closure",
            "rescal_base_plus_fold_no_weekly",
        ]
    if not hasattr(args, "num_runs"):
        args.num_runs = 5
    if not hasattr(args, "base_seed"):
        args.base_seed = 42
    if not hasattr(args, "ablation_csv"):
        args.ablation_csv = "ablation_results.csv"

    # Existing
    if not hasattr(args, "baseline_clean_start"):
        args.baseline_clean_start = 0
    if not hasattr(args, "rescal_clean_start"):
        args.rescal_clean_start = 0
    if not hasattr(args, "reuse_residual_cache"):
        args.reuse_residual_cache = 1

    # New knobs to help avoid OOM in ResCAL modes
    if not hasattr(args, "rescal_batch_size"):
        args.rescal_batch_size = args.batch_size
    if not hasattr(args, "rescal_eval_batch_size"):
        args.rescal_eval_batch_size = args.batch_size

    return args


def append_result_to_csv(csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    if osp.exists(csv_path):
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(csv_path, mode="w", header=True, index=False)


def flatten_numeric_dict(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_prefix = f"{prefix}_{k}" if prefix else str(k)
            out.update(flatten_numeric_dict(v, new_prefix))
    elif isinstance(d, (int, float, np.integer, np.floating)):
        out[prefix] = float(d)
    return out


def get_global_shared_dirs(args):
    shared_save_dir = osp.join(
        "save",
        args.dataset_choice,
        args.model_variant,
        "ablation",
        "shared",
    )
    shared_cache_dir = osp.join(
        "data",
        "processed",
        args.dataset_choice,
        args.model_variant,
        "ablation",
        "shared",
    )

    os.makedirs(shared_save_dir, exist_ok=True)
    os.makedirs(shared_cache_dir, exist_ok=True)
    return shared_save_dir, shared_cache_dir


def get_mode_dirs(args, run_idx, mode):
    run_name = f"run_{run_idx:02d}"

    save_dir = osp.join(
        "save",
        args.dataset_choice,
        args.model_variant,
        "ablation",
        run_name,
        mode,
    )
    cache_dir = osp.join(
        "data",
        "processed",
        args.dataset_choice,
        args.model_variant,
        "ablation",
        run_name,
        mode,
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    return save_dir, cache_dir


# ============================================================
# Dataset wrappers
# ============================================================
class ResidualTensorDataset(Dataset):
    def __init__(self, X, U, R, BASE, Y):
        self.X = X.float()
        self.U = U.float()
        self.R = R.float()
        self.BASE = BASE.float()
        self.Y = Y.float()

    @classmethod
    def from_pt(cls, pt_path):
        pack = torch.load(pt_path, map_location="cpu", weights_only=False)
        return cls(
            X=pack["X"],
            U=pack["U"],
            R=pack["R"],
            BASE=pack["BASE"],
            Y=pack["Y"],
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.U[idx],
            self.R[idx],
            self.BASE[idx],
            self.Y[idx],
        )


# ============================================================
# Baseline helpers
# ============================================================
def get_baseline_kind_for_mode(mode):
    if mode == "baseline_no":
        return "wz_no"
    return "wz"


def build_baseline_model(args, A_adj, baseline_kind="wz"):
    A_wave_val = A_wave(args, A_adj)

    if baseline_kind == "wz":
        build_model = model_wz
    elif baseline_kind == "wz_no":
        build_model = model_no
    else:
        raise ValueError(f"Unknown baseline_kind: {baseline_kind}")

    net = build_model(args, A_wave_val).to(args.device)
    return net, A_wave_val


def fit_baseline(net, train_loader, val_loader, mean, std, args, save_dir):
    best_path = osp.join(save_dir, "baseline_best.pt")
    last_path = osp.join(save_dir, "baseline_last.pt")
    legacy_path = osp.join(save_dir, "baseline_params.pt")

    if getattr(args, "baseline_clean_start", 0):
        for p in [best_path, last_path, legacy_path]:
            if osp.exists(p):
                print(f"[INFO] baseline_clean_start=1, removing: {p}")
                os.remove(p)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = nn.MSELoss().to(args.device)

    initepoch = 0
    best_rmse = np.inf
    stop = 0

    if osp.exists(last_path):
        print("[INFO] Loading last baseline checkpoint:", last_path)
        ckpt = torch.load(last_path, map_location=args.device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        initepoch = int(ckpt.get("epoch", 0))
        best_rmse = float(ckpt.get("best_rmse", np.inf))
        stop = int(ckpt.get("stop", 0))
        print(f"[INFO] Resuming baseline from epoch {initepoch}, best_val_rmse={best_rmse:.4f}")

    elif osp.exists(legacy_path):
        print("[INFO] Found legacy baseline checkpoint:", legacy_path)
        ckpt = torch.load(legacy_path, map_location=args.device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        initepoch = int(ckpt.get("epoch", 0))
        best_rmse = float(ckpt.get("best_rmse", np.inf))
        stop = 0

        torch.save(
            {
                "epoch": initepoch,
                "model_state_dict": net.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rmse": best_rmse,
                "stop": stop,
            },
            last_path,
        )
        if not osp.exists(best_path):
            torch.save(
                {
                    "epoch": initepoch,
                    "model_state_dict": net.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_rmse": best_rmse,
                    "stop": stop,
                },
                best_path,
            )

    else:
        print("[INFO] No baseline checkpoint found. Training from scratch.")

    if initepoch >= args.epochs:
        print(f"[INFO] Baseline already trained to epoch {initepoch}, target epochs={args.epochs}.")
    else:
        for epoch in range(initepoch, args.epochs):
            tr_loss = train_epoch(
                net,
                train_loader,
                optimizer,
                criterion,
                mean,
                std,
                args,
                epoch,
                log_every=50,
            )
            val_rmse = eval_rmse(net, val_loader, mean, std, args)

            print(f"[Epoch {epoch + 1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
            scheduler.step()

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                stop = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": net.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_rmse": best_rmse,
                        "stop": stop,
                    },
                    best_path,
                )
                print(f"  New best baseline RMSE: {best_rmse:.4f}")
            else:
                stop += 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_rmse": best_rmse,
                    "stop": stop,
                },
                last_path,
            )

            if stop >= 8:
                print("Early stopping baseline (patience=8)")
                break

    final_path = best_path if osp.exists(best_path) else last_path
    print("[INFO] Loading best baseline for evaluation/use:", final_path)

    ckpt = torch.load(final_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    print("[Baseline] Loaded + frozen:", final_path)
    return net, float(ckpt.get("best_rmse", best_rmse)), final_path


@torch.no_grad()
def collect_baseline_preds(net, loader, mean, std, args):
    preds = []
    labels = []

    net.eval()
    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)

        out = net(x)
        pred = out.detach().cpu().numpy() * std + mean
        label = y[:, :, :, 0].detach().cpu().numpy() * std + mean

        preds.append(pred)
        labels.append(label)

    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


# ============================================================
# ResCAL helpers
# ============================================================
def fit_rescal(rescal, train_loader, val_loader, support, args, save_path):
    opt = optim.Adam(rescal.parameters(), lr=args.learning_rate)
    sched = lr_scheduler.StepLR(opt, step_size=40, gamma=0.8)
    crit = nn.MSELoss().to(args.device)

    initepoch = 0
    best_val = np.inf
    stop = 0

    if getattr(args, "rescal_clean_start", 0):
        if osp.exists(save_path):
            print(f"[INFO] rescal_clean_start=1, removing existing ResCAL checkpoint: {save_path}")
            os.remove(save_path)

    if osp.exists(save_path):
        print(f"[INFO] Loading existing ResCAL checkpoint: {save_path}")
        ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
        rescal.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        if "scheduler_state_dict" in ckpt:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        initepoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val_rmse", np.inf))
        stop = int(ckpt.get("stop", 0))
        print(f"[INFO] Resuming ResCAL from epoch {initepoch}, best_val_rmse={best_val:.4f}")
    else:
        print("[INFO] No ResCAL checkpoint found. Training from scratch.")

    if initepoch >= args.epochs:
        print(f"[INFO] ResCAL already trained to epoch {initepoch}, target epochs={args.epochs}.")
    else:
        for epoch in range(initepoch, args.epochs):
            tr_loss = train_rescal_epoch(
                rescal, train_loader, support, opt, crit, args, epoch, log_every=50
            )
            val_rmse = eval_rescal_rmse(rescal, val_loader, support, args)

            print(f"[ResCAL][Epoch {epoch + 1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
            sched.step()

            if val_rmse < best_val:
                best_val = val_rmse
                stop = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": rescal.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "best_val_rmse": best_val,
                        "stop": stop,
                    },
                    save_path,
                )
                print(f"  New best ResCAL val RMSE: {best_val:.4f}")
            else:
                stop += 1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": rescal.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "best_val_rmse": best_val,
                        "stop": stop,
                    },
                    save_path,
                )

                if stop >= 8:
                    print("ResCAL early stopping (patience=8)")
                    break

    ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
    rescal.load_state_dict(ckpt["model_state_dict"])
    rescal.eval()
    return rescal, float(ckpt.get("best_val_rmse", best_val))


# ============================================================
# Cache / timestamp helpers
# ============================================================
def load_cached_split_pack(args, split_tag):
    cache_path = osp.join("data", "processed", args.dataset_choice, args.model_variant, f"hyp2_{split_tag}.pt")
    if not osp.exists(cache_path):
        raise FileNotFoundError(f"Missing cached split pack: {cache_path}")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def load_full_A_cached_pack(args):
    return load_cached_split_pack(args, "A")


# ============================================================
# Weekly residual helpers
# ============================================================
def compute_fullgraph_weekly_mu(args, node_ids, timestamp_list, interval_minutes=5):
    t_A = timestamp_list[0]

    speed_df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)

    speed_df.columns = speed_df.columns.astype(str)
    con_df.columns = con_df.columns.astype(str)
    node_ids = [str(x) for x in node_ids]

    speed_df = speed_df.loc[:, node_ids]
    con_df = con_df.loc[:, node_ids]

    speed_df_A = speed_df.iloc[:len(t_A)].copy()
    con_df_A = con_df.iloc[:len(t_A)].copy()

    speed_df_A.index = pd.to_datetime(speed_df_A.index)
    con_df_A.index = pd.to_datetime(con_df_A.index)

    mu = compute_weekly_mu_train_nonclosure_simple(
        speed_df_A=speed_df_A,
        closure_df_A=con_df_A,
        interval_minutes=interval_minutes,
    )
    return mu


def overwrite_speed_channel_with_weekly_residuals(pack, x_time_list, mu, interval_minutes=5):
    X = pack["X"].clone().float()
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    all_nodes = list(range(num_nodes))

    for s in range(num_samples):
        x_times = pd.to_datetime(x_time_list[s])
        weekly_base = weekly_mu_for_timestamps(
            mu=mu,
            timestamps=x_times,
            node_indices=all_nodes,
            interval_minutes=interval_minutes,
        )
        weekly_base = torch.as_tensor(weekly_base, dtype=X.dtype)
        X[s, 0] = X[s, 0] - weekly_base

    pack["X"] = X
    return pack


def save_weekly_residual_pack(raw_cache_path, weekly_cache_path, x_time_list, mu, interval_minutes=5):
    pack = torch.load(raw_cache_path, map_location="cpu", weights_only=False)
    pack = overwrite_speed_channel_with_weekly_residuals(
        pack=pack,
        x_time_list=x_time_list,
        mu=mu,
        interval_minutes=interval_minutes,
    )
    torch.save(pack, weekly_cache_path)
    return weekly_cache_path


def build_residual_dataset_for_split(
    base_model,
    source_loader,
    mean,
    std,
    args,
    raw_cache_path,
    weekly_cache_path,
    x_time_list,
    anchor_timestamps,
    use_existing_closure_channel,
    use_weekly_residual,
    mu=None,
):
    if osp.exists(raw_cache_path):
        print(f"[INFO] Removing existing residual raw cache: {raw_cache_path}")
        os.remove(raw_cache_path)
    if weekly_cache_path is not None and osp.exists(weekly_cache_path):
        print(f"[INFO] Removing existing residual weekly cache: {weekly_cache_path}")
        os.remove(weekly_cache_path)

    build_residual_dataset(
        base_model=base_model,
        source_loader=source_loader,
        mean=mean,
        std=std,
        args=args,
        anchor_timestamps=anchor_timestamps,
        cache_path=raw_cache_path,
        use_existing_closure_channel=use_existing_closure_channel,
    )

    if use_weekly_residual:
        save_weekly_residual_pack(
            raw_cache_path=raw_cache_path,
            weekly_cache_path=weekly_cache_path,
            x_time_list=x_time_list,
            mu=mu,
            interval_minutes=5,
        )
        ds = ResidualTensorDataset.from_pt(weekly_cache_path)

        if osp.exists(raw_cache_path):
            os.remove(raw_cache_path)
        if weekly_cache_path is not None and osp.exists(weekly_cache_path):
            os.remove(weekly_cache_path)

        return ds

    ds = ResidualTensorDataset.from_pt(raw_cache_path)

    if osp.exists(raw_cache_path):
        os.remove(raw_cache_path)

    return ds


# ============================================================
# Fold unseen residual helpers
# ============================================================
def train_or_load_fold_baseline(args, fold, mean, std, fold_save_dir):
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    print("\n" + "=" * 70)
    print(f"TRAIN / LOAD FOLD BASELINE {fold_id}")
    print(f"Held-out sensors: {heldout}")
    print("=" * 70)

    (
        A_train_fold,
        B_val,
        _C_test,
        _C_time,
        A_adj_fold,
        _mean_fold,
        _std_fold,
        _node_ids_fold,
        _timestamp_list_fold,
        _fold_meta,
    ) = loadData_hyp2_fold(args, heldout_sensors=heldout)

    train_loader_A_fold = DataLoader(A_train_fold, batch_size=args.batch_size, shuffle=True)
    val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)

    os.makedirs(fold_save_dir, exist_ok=True)
    net_fold, _ = build_baseline_model(args, A_adj_fold, baseline_kind="wz")

    t0 = time.perf_counter()
    net_fold, best_rmse, baseline_path = fit_baseline(
        net_fold,
        train_loader_A_fold,
        val_loader_B,
        mean,
        std,
        args,
        fold_save_dir,
    )
    fold_baseline_seconds = time.perf_counter() - t0
    print(f"[TIME] Fold {fold_id} baseline: {fold_baseline_seconds:.2f} sec")

    print(f"[Fold {fold_id}] best_val_rmse={best_rmse:.4f}")
    cleanup_memory()
    return net_fold, baseline_path, fold_baseline_seconds


def build_unseen_fold_residual_dataset(
    args,
    fold,
    full_A_dataset,
    mean,
    std,
    mu,
    use_closure_channel,
    use_weekly_residual,
    fold_baseline_save_dir,
    mode_fold_cache_dir,
):
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    print("\n" + "-" * 70)
    print(f"BUILD UNSEEN FOLD RESIDUALS FOR RESCAL | fold={fold_id}")
    print(f"Held-out sensors: {heldout}")
    print("-" * 70)

    net_fold, baseline_path, fold_baseline_seconds = train_or_load_fold_baseline(
        args=args,
        fold=fold,
        mean=mean,
        std=std,
        fold_save_dir=fold_baseline_save_dir,
    )

    A_pack_full = load_full_A_cached_pack(args)
    x_time_A = A_pack_full[2]
    y_time_A = A_pack_full[3]

    _, _, _, _, _, _, _, node_ids, timestamp_list = loadData_hyp2(args)
    t_A = timestamp_list[0]

    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    con_df.columns = con_df.columns.astype(str)
    node_ids = [str(x) for x in node_ids]
    con_df = con_df.loc[:, node_ids]
    con_df_A = con_df.iloc[:len(t_A)].copy()
    con_df_A.index = pd.to_datetime(con_df_A.index)

    events = find_heldout_closure_events(
        con_df_A,
        heldout_sensors=heldout,
        center_threshold=0.999,
    )
    print(f"[Fold {fold_id}] raw held-out A events: {len(events)}")

    merged = expand_and_merge_events(
        events,
        total_len=len(con_df_A),
        pre_steps=args.P,
        post_steps=args.T,
    )
    print(f"[Fold {fold_id}] merged expanded intervals: {len(merged)}")

    keep_mask = build_sample_mask_from_expanded_idx_intervals(
        x_time_A,
        y_time_A,
        split_index=con_df_A.index,
        merged_intervals=merged,
    )

    selected_indices = [i for i, keep in enumerate(keep_mask) if keep]
    print(f"[Fold {fold_id}] selected unseen A samples: {len(selected_indices)}")

    if len(selected_indices) == 0:
        cleanup_memory()
        return None, 0.0, 0.0

    A_subset = Subset(full_A_dataset, selected_indices)
    A_loader_sel = DataLoader(A_subset, batch_size=args.rescal_eval_batch_size, shuffle=False)

    x_time_sel = [x_time_A[i] for i in selected_indices]
    anchor_timestamps = [y_time_A[i][0] for i in selected_indices]

    os.makedirs(mode_fold_cache_dir, exist_ok=True)
    raw_cache = osp.join(mode_fold_cache_dir, "residual_unseen_fold_A_raw.pt")
    weekly_cache = osp.join(mode_fold_cache_dir, "residual_unseen_fold_A_weekly.pt") if use_weekly_residual else None

    residual_t0 = time.perf_counter()
    ds = build_residual_dataset_for_split(
        base_model=net_fold,
        source_loader=A_loader_sel,
        mean=mean,
        std=std,
        args=args,
        raw_cache_path=raw_cache,
        weekly_cache_path=weekly_cache,
        x_time_list=x_time_sel,
        anchor_timestamps=anchor_timestamps,
        use_existing_closure_channel=use_closure_channel,
        use_weekly_residual=use_weekly_residual,
        mu=mu,
    )
    fold_residual_seconds = time.perf_counter() - residual_t0

    meta = {
        "fold_id": int(fold_id),
        "heldout_sensors": heldout,
        "baseline_path": baseline_path,
        "num_selected_samples": len(selected_indices),
        "raw_cache": "",
        "weekly_cache": "",
        "use_closure_channel": bool(use_closure_channel),
        "use_weekly_residual": bool(use_weekly_residual),
        "fold_baseline_seconds": float(fold_baseline_seconds),
        "fold_residual_seconds": float(fold_residual_seconds),
    }
    meta_path = osp.join(mode_fold_cache_dir, "residual_unseen_fold_A_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Fold {fold_id}] saved unseen residual dataset -> {meta_path}")
    cleanup_memory()
    return ds, fold_baseline_seconds, fold_residual_seconds


# ============================================================
# Evaluation / ablation runners
# ============================================================
def evaluate_predictions(args, timestamp_list, pred_base_C, pred_final_C, label_C):
    test_start = len(timestamp_list[0]) + len(timestamp_list[1])

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    y_t_C_full = load_y_t_list_cached(args, "hyp2_C.pt")
    y_t_C = y_t_C_full[:pred_base_C.shape[0]]

    event_base = event_closure_metrics_from_arrays(
        preds=pred_base_C,
        labels=label_C,
        df_con=df_con,
        df_dis=df_dis,
        test_start=test_start,
        radius_miles=3.0,
        y_t_list=y_t_C,
        P=args.P,
    )
    event_final = event_closure_metrics_from_arrays(
        preds=pred_final_C,
        labels=label_C,
        df_con=df_con,
        df_dis=df_dis,
        test_start=test_start,
        radius_miles=3.0,
        y_t_list=y_t_C,
        P=args.P,
    )

    mae_b, rmse_b, mape_b, _ = metric(pred_base_C, label_C)
    mae_f, rmse_f, mape_f, _ = metric(pred_final_C, label_C)

    return {
        "overall": {
            "baseline_mae": float(mae_b),
            "baseline_rmse": float(rmse_b),
            "baseline_mape": float(mape_b),
            "final_mae": float(mae_f),
            "final_rmse": float(rmse_f),
            "final_mape": float(mape_f),
        },
        "event_base": event_base,
        "event_final": event_final,
    }


def run_baseline_mode(
    mode,
    mode_save_dir,
    csv_path,
    args,
    run_idx,
    seed,
    best_base,
    baseline_path,
    net,
    test_loader_C,
    mean,
    std,
    timestamp_list,
    cfg,
    baseline_train_seconds,
    total_fold_baseline_seconds,
    weekly_mu_seconds,
):
    eval_t0 = time.perf_counter()

    pred_base_C, label_C = collect_baseline_preds(net, test_loader_C, mean, std, args)
    metrics_bundle = evaluate_predictions(
        args=args,
        timestamp_list=timestamp_list,
        pred_base_C=pred_base_C,
        pred_final_C=pred_base_C,
        label_C=label_C,
    )

    baseline_eval_seconds = time.perf_counter() - eval_t0
    total_mode_seconds = baseline_eval_seconds

    event_base = metrics_bundle["event_base"]
    event_final = metrics_bundle["event_final"]
    overall = metrics_bundle["overall"]

    print_event_metrics_block(f"EVENT METRICS ({mode} | BASELINE on C)", event_base)
    print_event_metrics_block(f"EVENT METRICS ({mode} | FINAL on C)", event_final)
    print_test_results_block(
        overall["baseline_mae"],
        overall["baseline_rmse"],
        overall["baseline_mape"],
        overall["final_mae"],
        overall["final_rmse"],
        overall["final_mape"],
        split_name=f"C | {mode}",
    )

    row = {
        "mode": mode,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "use_extra_folds": bool(cfg["use_extra_folds"]),
        "use_closure_channel": bool(cfg["use_closure_channel"]),
        "use_weekly_residual": bool(cfg["use_weekly_residual"]),
        "baseline_best_val_rmse": float(best_base),
        "rescal_best_val_rmse": np.nan,
        **overall,
        **flatten_numeric_dict(event_base, prefix="event_base"),
        **flatten_numeric_dict(event_final, prefix="event_final"),
        "baseline_train_seconds": float(baseline_train_seconds),
        "total_fold_baseline_seconds": float(total_fold_baseline_seconds),
        "weekly_mu_seconds": float(weekly_mu_seconds),
        "residual_build_seconds": 0.0,
        "fold_extra_seconds": 0.0,
        "fold_residual_only_seconds": 0.0,
        "rescal_train_seconds": 0.0,
        "baseline_eval_seconds": float(baseline_eval_seconds),
        "evaluation_seconds": 0.0,
        "total_mode_seconds": float(total_mode_seconds),
        "baseline_path": baseline_path,
        "rescal_path": "",
    }
    append_result_to_csv(csv_path, row)

    summary_path = osp.join(mode_save_dir, f"summary_{mode}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(f"[INFO] Saved summary: {summary_path}")
    cleanup_memory()
    return row


def run_one_rescal_experiment(
    mode,
    cfg,
    dsA,
    dsB,
    dsC,
    extra_fold_datasets,
    support,
    mode_save_dir,
    csv_path,
    args,
    run_idx,
    seed,
    best_base,
    baseline_path,
    mean,
    std,
    timestamp_list,
    baseline_train_seconds,
    total_fold_baseline_seconds,
    weekly_mu_seconds,
    residual_build_seconds,
    fold_extra_seconds,
    fold_residual_only_seconds,
):
    mode_t0 = time.perf_counter()

    if len(extra_fold_datasets) > 0:
        train_rescal_ds = ConcatDataset([dsA] + extra_fold_datasets)
    else:
        train_rescal_ds = dsA

    resid_loader_A = DataLoader(train_rescal_ds, batch_size=args.rescal_batch_size, shuffle=True)
    resid_loader_B = DataLoader(dsB, batch_size=args.rescal_eval_batch_size, shuffle=False)
    resid_loader_C = DataLoader(dsC, batch_size=args.rescal_eval_batch_size, shuffle=False)

    extra_count = sum(len(ds) for ds in extra_fold_datasets)

    sample_X, _, _, _, _ = train_rescal_ds[0]
    cx = int(sample_X.shape[0])

    print("\n" + "=" * 70)
    print(f"RUNNING RESCAL MODE: {mode}")
    print("=" * 70)
    print(f"Base A residual samples     : {len(dsA)}")
    print(f"Extra fold residual samples : {extra_count}")
    print(f"Total ResCAL train samples  : {len(train_rescal_ds)}")
    print(f"ResCAL val samples (B)      : {len(dsB)}")
    print(f"ResCAL input channels       : {cx}")
    print(f"ResCAL train batch size     : {args.rescal_batch_size}")
    print(f"ResCAL eval batch size      : {args.rescal_eval_batch_size}")
    print("===================================\n")

    rescal = ResCAL(
        num_nodes=args.num_nodes,
        Tu=args.P,
        out_dim=args.P,
        Cx=cx,
        layers=4,
        dc=32,
        nc=16,
        de=16,
        tau=1.0,
        dropout=0.0,
        support_len=1,
        gcn_order=1,
    ).to(args.device)

    rescal_path = osp.join(mode_save_dir, f"rescal_{mode}.pt")

    train_t0 = time.perf_counter()
    rescal, best_res = fit_rescal(rescal, resid_loader_A, resid_loader_B, support, args, rescal_path)
    rescal_train_seconds = time.perf_counter() - train_t0
    print(f"[TIME] ResCAL training ({mode}): {rescal_train_seconds:.2f} sec")

    eval_t0 = time.perf_counter()
    pred_base_C, pred_final_C, label_C = collect_preds_on_E_resid(
        rescal, resid_loader_C, support, mean, std, args
    )

    metrics_bundle = evaluate_predictions(
        args=args,
        timestamp_list=timestamp_list,
        pred_base_C=pred_base_C,
        pred_final_C=pred_final_C,
        label_C=label_C,
    )
    evaluation_seconds = time.perf_counter() - eval_t0

    total_mode_seconds = time.perf_counter() - mode_t0

    event_base = metrics_bundle["event_base"]
    event_final = metrics_bundle["event_final"]
    overall = metrics_bundle["overall"]

    print_event_metrics_block(f"EVENT METRICS ({mode} | BASELINE on C)", event_base)
    print_event_metrics_block(f"EVENT METRICS ({mode} | BASE + ResCAL on C)", event_final)
    print_test_results_block(
        overall["baseline_mae"],
        overall["baseline_rmse"],
        overall["baseline_mape"],
        overall["final_mae"],
        overall["final_rmse"],
        overall["final_mape"],
        split_name=f"C | {mode}",
    )

    row = {
        "mode": mode,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "use_extra_folds": bool(cfg["use_extra_folds"]),
        "use_closure_channel": bool(cfg["use_closure_channel"]),
        "use_weekly_residual": bool(cfg["use_weekly_residual"]),
        "baseline_best_val_rmse": float(best_base),
        "rescal_best_val_rmse": float(best_res),
        "rescal_train_base_samples": int(len(dsA)),
        "rescal_train_extra_samples": int(extra_count),
        "rescal_train_total_samples": int(len(train_rescal_ds)),
        **overall,
        **flatten_numeric_dict(event_base, prefix="event_base"),
        **flatten_numeric_dict(event_final, prefix="event_final"),
        "baseline_train_seconds": float(baseline_train_seconds),
        "total_fold_baseline_seconds": float(total_fold_baseline_seconds),
        "weekly_mu_seconds": float(weekly_mu_seconds),
        "residual_build_seconds": float(residual_build_seconds),
        "fold_extra_seconds": float(fold_extra_seconds),
        "fold_residual_only_seconds": float(fold_residual_only_seconds),
        "rescal_train_seconds": float(rescal_train_seconds),
        "baseline_eval_seconds": 0.0,
        "evaluation_seconds": float(evaluation_seconds),
        "total_mode_seconds": float(total_mode_seconds),
        "baseline_path": baseline_path,
        "rescal_path": rescal_path,
    }

    append_result_to_csv(csv_path, row)

    summary_path = osp.join(mode_save_dir, f"summary_{mode}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(f"[INFO] Saved summary: {summary_path}")
    cleanup_memory()
    return row


# ============================================================
# Main
# ============================================================
def main():
    args = build_args()
    apply_dataset_cfg(args)
    ensure_ablation_args(args)

    print("CONFIG:", args.dataset_choice, args.model_variant)
    print("Device:", args.device)
    print("Ablation modes:", args.ablation_modes)
    print("Num runs:", args.num_runs)
    print("Base seed:", args.base_seed)
    print("baseline_clean_start:", args.baseline_clean_start)
    print("rescal_clean_start:", args.rescal_clean_start)
    print("reuse_residual_cache:", args.reuse_residual_cache)
    print("rescal_batch_size:", args.rescal_batch_size)
    print("rescal_eval_batch_size:", args.rescal_eval_batch_size)

    csv_path = osp.join("save", args.dataset_choice, args.model_variant, args.ablation_csv)

    for mode in args.ablation_modes:
        if mode not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation mode: {mode}")

    all_run_summaries = []

    # --------------------------------------------------------
    # GLOBAL SHARED BASELINES (train once total)
    # --------------------------------------------------------
    shared_save_dir, _shared_cache_dir = get_global_shared_dirs(args)

    baseline_save_dir_wz = osp.join(shared_save_dir, "full_baseline_wz")
    baseline_save_dir_no = osp.join(shared_save_dir, "full_baseline_wz_no")
    folds_save_root = osp.join(shared_save_dir, "fold_baselines")

    os.makedirs(baseline_save_dir_wz, exist_ok=True)
    os.makedirs(baseline_save_dir_no, exist_ok=True)
    os.makedirs(folds_save_root, exist_ok=True)

    set_all_seeds(args.base_seed)

    A_train, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
    val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)
    test_loader_C = DataLoader(C_test, batch_size=args.batch_size, shuffle=False)

    # -------- Full baseline: STGCN_WZ --------
    net_wz, A_wave_val_wz = build_baseline_model(args, A_adj, baseline_kind="wz")
    t0 = time.perf_counter()
    net_wz, best_base_wz, baseline_path_wz = fit_baseline(
        net_wz,
        train_loader_A,
        val_loader_B,
        mean,
        std,
        args,
        baseline_save_dir_wz,
    )
    baseline_train_seconds_wz = time.perf_counter() - t0
    print(f"[TIME] Full baseline WZ training: {baseline_train_seconds_wz:.2f} sec")

    # -------- Full baseline: STGCN_WZ_NO --------
    net_no, A_wave_val_no = build_baseline_model(args, A_adj, baseline_kind="wz_no")
    t0 = time.perf_counter()
    net_no, best_base_no, baseline_path_no = fit_baseline(
        net_no,
        train_loader_A,
        val_loader_B,
        mean,
        std,
        args,
        baseline_save_dir_no,
    )
    baseline_train_seconds_no = time.perf_counter() - t0
    print(f"[TIME] Full baseline WZ_NO training: {baseline_train_seconds_no:.2f} sec")

    support = [A_wave_val_wz.to(args.device).float()]

    folds, _sensor_burden = build_closure_sensor_folds_from_args(args, k=5)

    print("\n=== Fold Summary ===")
    for fold in folds:
        print(
            f"Fold {fold['fold_id']}: "
            f"{len(fold['heldout_sensors'])} sensors | "
            f"closure_steps={fold['total_closure_steps']} | "
            f"sensors={fold['heldout_sensors']}"
        )
    print("====================\n")

    total_fold_baseline_seconds = 0.0
    for fold in folds:
        fold_baseline_save_dir = osp.join(folds_save_root, f"fold_{fold['fold_id']}")
        os.makedirs(fold_baseline_save_dir, exist_ok=True)

        _, _, fold_baseline_seconds = train_or_load_fold_baseline(
            args=args,
            fold=fold,
            mean=mean,
            std=std,
            fold_save_dir=fold_baseline_save_dir,
        )
        total_fold_baseline_seconds += fold_baseline_seconds

    weekly_t0 = time.perf_counter()
    mu = compute_fullgraph_weekly_mu(
        args=args,
        node_ids=node_ids,
        timestamp_list=timestamp_list,
        interval_minutes=5,
    )
    weekly_mu_seconds = time.perf_counter() - weekly_t0
    print(f"[TIME] Weekly mean computation: {weekly_mu_seconds:.2f} sec")

    A_pack = load_cached_split_pack(args, "A")
    B_pack = load_cached_split_pack(args, "B")
    C_pack = load_cached_split_pack(args, "C")

    x_time_A, y_time_A = A_pack[2], A_pack[3]
    x_time_B, y_time_B = B_pack[2], B_pack[3]
    x_time_C, y_time_C = C_pack[2], C_pack[3]

    anchor_timestamps_A = [yt[0] for yt in y_time_A]
    anchor_timestamps_B = [yt[0] for yt in y_time_B]
    anchor_timestamps_C = [yt[0] for yt in y_time_C]

    baseline_modes_already_saved = set()

    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        set_all_seeds(seed)

        print("\n" + "#" * 90)
        print(f"RUN {run_idx:02d} | SEED={seed}")
        print("#" * 90)

        train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
        val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)
        test_loader_C = DataLoader(C_test, batch_size=args.batch_size, shuffle=False)

        for mode in args.ablation_modes:
            cfg = ABLATION_CONFIGS[mode]
            mode_save_dir, mode_cache_dir = get_mode_dirs(args, run_idx, mode)
            baseline_kind = get_baseline_kind_for_mode(mode)

            if baseline_kind == "wz":
                net_mode = net_wz
                best_base_mode = best_base_wz
                baseline_path_mode = baseline_path_wz
                baseline_train_seconds_mode = baseline_train_seconds_wz
            else:
                net_mode = net_no
                best_base_mode = best_base_no
                baseline_path_mode = baseline_path_no
                baseline_train_seconds_mode = baseline_train_seconds_no

            print("\n" + "*" * 90)
            print(f"MODE: {mode}")
            print(json.dumps(cfg, indent=2))
            print("*" * 90)

            mu_mode = mu if cfg["use_weekly_residual"] else None

            if not cfg["run_rescal"]:
                if mode in baseline_modes_already_saved:
                    print(f"[INFO] Baseline mode '{mode}' already saved once. Skipping run {run_idx:02d}.")
                    continue

                summary = run_baseline_mode(
                    mode=mode,
                    mode_save_dir=mode_save_dir,
                    csv_path=csv_path,
                    args=args,
                    run_idx=run_idx,
                    seed=seed,
                    best_base=best_base_mode,
                    baseline_path=baseline_path_mode,
                    net=net_mode,
                    test_loader_C=test_loader_C,
                    mean=mean,
                    std=std,
                    timestamp_list=timestamp_list,
                    cfg=cfg,
                    baseline_train_seconds=baseline_train_seconds_mode,
                    total_fold_baseline_seconds=total_fold_baseline_seconds,
                    weekly_mu_seconds=weekly_mu_seconds,
                )
                all_run_summaries.append(summary)
                baseline_modes_already_saved.add(mode)
                continue

            residual_t0 = time.perf_counter()
            dsA = build_residual_dataset_for_split(
                base_model=net_mode,
                source_loader=DataLoader(A_train, batch_size=args.rescal_eval_batch_size, shuffle=False),
                mean=mean,
                std=std,
                args=args,
                raw_cache_path=osp.join(mode_cache_dir, "residual_A_raw.pt"),
                weekly_cache_path=osp.join(mode_cache_dir, "residual_A_weekly.pt") if cfg["use_weekly_residual"] else None,
                x_time_list=x_time_A,
                anchor_timestamps=anchor_timestamps_A,
                use_existing_closure_channel=cfg["use_closure_channel"],
                use_weekly_residual=cfg["use_weekly_residual"],
                mu=mu_mode,
            )

            dsB = build_residual_dataset_for_split(
                base_model=net_mode,
                source_loader=DataLoader(B_val, batch_size=args.rescal_eval_batch_size, shuffle=False),
                mean=mean,
                std=std,
                args=args,
                raw_cache_path=osp.join(mode_cache_dir, "residual_B_raw.pt"),
                weekly_cache_path=osp.join(mode_cache_dir, "residual_B_weekly.pt") if cfg["use_weekly_residual"] else None,
                x_time_list=x_time_B,
                anchor_timestamps=anchor_timestamps_B,
                use_existing_closure_channel=cfg["use_closure_channel"],
                use_weekly_residual=cfg["use_weekly_residual"],
                mu=mu_mode,
            )

            dsC = build_residual_dataset_for_split(
                base_model=net_mode,
                source_loader=DataLoader(C_test, batch_size=args.rescal_eval_batch_size, shuffle=False),
                mean=mean,
                std=std,
                args=args,
                raw_cache_path=osp.join(mode_cache_dir, "residual_C_raw.pt"),
                weekly_cache_path=osp.join(mode_cache_dir, "residual_C_weekly.pt") if cfg["use_weekly_residual"] else None,
                x_time_list=x_time_C,
                anchor_timestamps=anchor_timestamps_C,
                use_existing_closure_channel=cfg["use_closure_channel"],
                use_weekly_residual=cfg["use_weekly_residual"],
                mu=mu_mode,
            )
            residual_build_seconds = time.perf_counter() - residual_t0
            print(f"[TIME] Residual build ({mode}): {residual_build_seconds:.2f} sec")

            fold_extra_t0 = time.perf_counter()
            extra_fold_datasets = []
            fold_residual_only_seconds = 0.0

            if cfg["use_extra_folds"]:
                for fold in folds:
                    fold_baseline_save_dir = osp.join(folds_save_root, f"fold_{fold['fold_id']}")
                    mode_fold_cache_dir = osp.join(mode_cache_dir, f"fold_{fold['fold_id']}")

                    ds_extra, _fold_baseline_seconds, fold_residual_seconds = build_unseen_fold_residual_dataset(
                        args=args,
                        fold=fold,
                        full_A_dataset=A_train,
                        mean=mean,
                        std=std,
                        mu=mu_mode,
                        use_closure_channel=cfg["use_closure_channel"],
                        use_weekly_residual=cfg["use_weekly_residual"],
                        fold_baseline_save_dir=fold_baseline_save_dir,
                        mode_fold_cache_dir=mode_fold_cache_dir,
                    )
                    if ds_extra is not None:
                        extra_fold_datasets.append(ds_extra)
                    fold_residual_only_seconds += fold_residual_seconds
                    cleanup_memory()

            fold_extra_seconds = time.perf_counter() - fold_extra_t0
            print(f"[TIME] Fold extra residual build ({mode}): {fold_extra_seconds:.2f} sec")

            summary = run_one_rescal_experiment(
                mode=mode,
                cfg=cfg,
                dsA=dsA,
                dsB=dsB,
                dsC=dsC,
                extra_fold_datasets=extra_fold_datasets,
                support=support,
                mode_save_dir=mode_save_dir,
                csv_path=csv_path,
                args=args,
                run_idx=run_idx,
                seed=seed,
                best_base=best_base_mode,
                baseline_path=baseline_path_mode,
                mean=mean,
                std=std,
                timestamp_list=timestamp_list,
                baseline_train_seconds=baseline_train_seconds_mode,
                total_fold_baseline_seconds=total_fold_baseline_seconds,
                weekly_mu_seconds=weekly_mu_seconds,
                residual_build_seconds=residual_build_seconds,
                fold_extra_seconds=fold_extra_seconds,
                fold_residual_only_seconds=fold_residual_only_seconds,
            )

            all_run_summaries.append(summary)

            del dsA, dsB, dsC, extra_fold_datasets
            cleanup_memory()

    compare_path = osp.join("save", args.dataset_choice, args.model_variant, "ablation_compare_summary.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(all_run_summaries, f, indent=2)

    print(f"[INFO] Saved full ablation summary: {compare_path}")


if __name__ == "__main__":
    main()
