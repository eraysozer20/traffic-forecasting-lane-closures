# main_fold_baseline.py
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader

from STGCN_WZ import model as model_wz
from STGCN_WZ_NO import model as model_no

from utils.cli import build_args
from utils.configs import apply_dataset_cfg
from utils.data_loading import (
    build_closure_sensor_folds_from_args,
    loadData_hyp2_fold,
    loadData_hyp2,
)
from utils.residuals import (
    find_heldout_closure_events,
    expand_and_merge_events,
    build_sample_mask_from_expanded_idx_intervals,
    subset_pack,
    build_residual_dataset,
)
from utils.features import A_wave
from utils.training import train_epoch, eval_rmse
from utils.metrics import metric
from utils.printing import print_test_results_block
from utils.local_rescal_prep import (
    build_local_rescal_dataset_for_fold,
    compute_weekly_mu_train_nonclosure_simple,
    weekly_mu_for_timestamps,
    get_local_nodes_from_distance,
    slice_local_adjacency,
    expand_events_without_merging,
    choose_center_for_sample,
)

import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, ConcatDataset, random_split
from MyResCAL import ResCAL   


def build_baseline_model(args, A_adj):
    A_wave_val = A_wave(args, A_adj)
    build_model = model_wz if args.model_variant == "wz" else model_no
    net = build_model(args, A_wave_val).to(args.device)
    return net, A_wave_val

def load_full_A_cached_pack(args):
    cache_path = osp.join("data", "processed", args.dataset_choice, args.model_variant, "hyp2_A.pt")
    if not osp.exists(cache_path):
        raise FileNotFoundError(f"Missing cached A pack: {cache_path}")
    return torch.load(cache_path, map_location="cpu", weights_only=False)

def fit_baseline(net, train_loader, val_loader, mean, std, args, save_dir):
    best_path = osp.join(save_dir, "baseline_best.pt")
    last_path = osp.join(save_dir, "baseline_last.pt")

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = nn.MSELoss().to(args.device)

    initepoch = 0
    best_rmse = np.inf
    stop = 0

    # Resume from LAST checkpoint if it exists
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

        if initepoch >= args.epochs:
            print(f"[INFO] Last checkpoint already trained to epoch {initepoch}, target epochs={args.epochs}.")
        else:
            print(f"[INFO] Resuming from epoch {initepoch}, best_val_rmse={best_rmse:.4f}")
    else:
        print("[INFO] No last checkpoint found. Training from scratch.")

    for epoch in range(initepoch, args.epochs):
        tr_loss = train_epoch(
            net, train_loader, optimizer, criterion, mean, std, args, epoch, log_every=50
        )
        val_rmse = eval_rmse(net, val_loader, mean, std, args)

        print(f"[Epoch {epoch+1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
        scheduler.step()

        improved = val_rmse < best_rmse
        if improved:
            best_rmse = val_rmse
            stop = 0

            # Save BEST checkpoint
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

        # Save LAST checkpoint every epoch
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

    # For evaluation, load BEST checkpoint if available, otherwise LAST
    final_path = best_path if osp.exists(best_path) else last_path
    ckpt = torch.load(final_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    return net, float(ckpt.get("best_rmse", best_rmse))


@torch.no_grad()
def collect_baseline_preds(net, loader, mean, std, args):
    preds = []
    labels = []

    net.eval()
    for batch in loader:
        x, y = batch
        x = x.to(args.device)
        y = y.to(args.device)

        out = net(x)                    # (B, P, N)
        out = out * std + mean

        y = y[:, :, :, 0]              # speed only
        y = y * std + mean             # (B, P, N)

        preds.append(out.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    label = np.concatenate(labels, axis=0)
    return pred, label


def save_fold_metadata(save_dir, fold_id, fold_meta):
    meta_path = osp.join(save_dir, f"fold_{fold_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fold_meta, f, indent=2)
    print(f"[INFO] Saved fold metadata: {meta_path}")


def train_one_fold(args, fold):
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    print("\n" + "=" * 70)
    print(f"TRAINING FOLD {fold_id}")
    print(f"Held-out sensors: {heldout}")
    print("=" * 70)

    (
        A_train,
        B_val,
        C_test,
        C_time,
        A_adj,
        mean,
        std,
        node_ids,
        timestamp_list,
        fold_meta,
    ) = loadData_hyp2_fold(args, heldout_sensors=heldout)

    train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
    val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)
    test_loader_C = DataLoader(C_test, batch_size=args.batch_size, shuffle=False)

    print(f"[Fold {fold_id}] Train samples after filtering: {len(A_train)}")
    print(f"[Fold {fold_id}] Val samples: {len(B_val)}")
    print(f"[Fold {fold_id}] Test samples: {len(C_test)}")

    save_dir = osp.join("save", args.dataset_choice, args.model_variant, f"fold_{fold_id}")
    os.makedirs(save_dir, exist_ok=True)

    save_fold_metadata(save_dir, fold_id, fold_meta)

    net, A_wave_val = build_baseline_model(args, A_adj)
    net, best_val_rmse = fit_baseline(
        net, train_loader_A, val_loader_B, mean, std, args, save_dir
    )

    pred_C, label_C = collect_baseline_preds(net, test_loader_C, mean, std, args)
    print("pred_C shape :", pred_C.shape)
    print("label_C shape:", label_C.shape)

    mae_c, rmse_c, mape_c, _ = metric(pred_C, label_C)

    print_test_results_block(
        mae_c, rmse_c, mape_c,
        mae_c, rmse_c, mape_c,
        split_name=f"C (Fold {fold_id}, Baseline Only)"
    )

    best_path = osp.join(save_dir, "baseline_best.pt")
    last_path = osp.join(save_dir, "baseline_last.pt")

    results = {
        "fold_id": fold_id,
        "heldout_sensors": heldout,
        "num_train_after": len(A_train),
        "num_val": len(B_val),
        "num_test": len(C_test),
        "best_val_rmse": float(best_val_rmse),
        "test_mae": float(mae_c),
        "test_rmse": float(rmse_c),
        "test_mape": float(mape_c),
        "checkpoint_best_path": best_path,
        "checkpoint_last_path": last_path,
    }

    results_path = osp.join(save_dir, "baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved fold results: {results_path}")
    return results

def train_full_baseline(args):
    """
    Train the normal full baseline on A, validate on B, evaluate on C.
    No fold filtering.
    """
    print("\n" + "=" * 70)
    print("TRAINING FULL BASELINE (NO FOLD FILTERING)")
    print("=" * 70)

    A_train, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)

    train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
    val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)
    test_loader_C = DataLoader(C_test, batch_size=args.batch_size, shuffle=False)

    save_dir = osp.join("save", args.dataset_choice, args.model_variant, "full_baseline")
    os.makedirs(save_dir, exist_ok=True)

    net, A_wave_val = build_baseline_model(args, A_adj)
    net, best_val_rmse = fit_baseline(
        net, train_loader_A, val_loader_B, mean, std, args, save_dir
    )

    pred_C, label_C = collect_baseline_preds(net, test_loader_C, mean, std, args)
    mae_c, rmse_c, mape_c, _ = metric(pred_C, label_C)

    print_test_results_block(
        mae_c, rmse_c, mape_c,
        mae_c, rmse_c, mape_c,
        split_name="C (Full Baseline)"
    )

    results = {
        "best_val_rmse": float(best_val_rmse),
        "test_mae": float(mae_c),
        "test_rmse": float(rmse_c),
        "test_mape": float(mape_c),
        "checkpoint_best_path": osp.join(save_dir, "baseline_best.pt"),
        "checkpoint_last_path": osp.join(save_dir, "baseline_last.pt"),
    }

    results_path = osp.join(save_dir, "full_baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved full baseline results: {results_path}")
    return results


def build_unseen_residuals_for_fold(args, fold):
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    print("\n" + "-" * 70)
    print(f"BUILDING UNSEEN RESIDUALS FOR FOLD {fold_id}")
    print(f"Held-out sensors: {heldout}")
    print("-" * 70)

    # full A split and metadata
    A_train_full, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    t_A = timestamp_list[0]

    # full cached A pack
    A_pack_full = load_full_A_cached_pack(args)
    x_time_A = A_pack_full[2]
    y_time_A = A_pack_full[3]

    # closure df for A/train only
    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    len_train = len(t_A)
    df_con_A = df_con.iloc[:len_train].copy()
    df_con_A.index = pd.to_datetime(df_con_A.index)

    # load trained fold baseline
    save_dir = osp.join("save", args.dataset_choice, args.model_variant, f"fold_{fold_id}")
    baseline_path = pick_baseline_checkpoint(save_dir)
    if not osp.exists(baseline_path):
        raise FileNotFoundError(f"Missing fold baseline checkpoint: {baseline_path}")

    net, _ = build_baseline_model(args, A_adj)
    ckpt = torch.load(baseline_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # 1) find held-out closure-center events in A/train
    events = find_heldout_closure_events(
        df_con_A,
        heldout_sensors=heldout,
        center_threshold=0.999
    )
    print(f"[Fold {fold_id}] raw held-out events: {len(events)}")

    # 2) expand intervals: start-P, end+T
    merged = expand_and_merge_events(
        events,
        total_len=len(df_con_A),
        pre_steps=args.P,   # e.g. 6
        post_steps=args.T,  # e.g. 12
    )
    print(f"[Fold {fold_id}] merged expanded intervals: {len(merged)}")

    # 3) build sample mask from x/y time overlap
    keep_mask = build_sample_mask_from_expanded_idx_intervals(
        x_time_A,
        y_time_A,
        split_index=df_con_A.index,
        merged_intervals=merged,
    )

    num_selected = int(keep_mask.sum())
    print(f"[Fold {fold_id}] A full samples           : {len(keep_mask)}")
    print(f"[Fold {fold_id}] selected residual samples : {num_selected}")

    if num_selected == 0:
        print(f"[Fold {fold_id}] No unseen residual samples selected. Skipping residual dataset build.")
        return {
            "fold_id": fold_id,
            "heldout_sensors": heldout,
            "num_raw_events": len(events),
            "num_merged_intervals": len(merged),
            "num_selected_samples": 0,
            "residual_cache": None,
        }

    # 4) subset dataset using the selected indices
    selected_indices = [i for i, keep in enumerate(keep_mask) if keep]
    A_subset = Subset(A_train_full, selected_indices)
    A_loader_sel = DataLoader(A_subset, batch_size=args.batch_size, shuffle=False)

    # 5) first y timestamp per selected sample -> anchor timestamps for time features
    anchor_timestamps = [yt[0] for yt, keep in zip(y_time_A, keep_mask) if keep]

    # 6) build residual dataset using existing closure channel in x
    resid_cache = osp.join(save_dir, "residuals_unseen_fold_A.pt")
    resid_ds = build_residual_dataset(
        base_model=net,
        source_loader=A_loader_sel,
        mean=mean,
        std=std,
        args=args,
        anchor_timestamps=anchor_timestamps,
        cache_path=resid_cache,
        use_existing_closure_channel=True,
    )

    meta = {
        "fold_id": fold_id,
        "heldout_sensors": heldout,
        "num_raw_events": len(events),
        "num_merged_intervals": len(merged),
        "num_selected_samples": num_selected,
        "pre_steps": int(args.P),
        "post_steps": int(args.T),
        "residual_cache": resid_cache,
    }

    meta_path = osp.join(save_dir, "residuals_unseen_fold_A_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Fold {fold_id}] saved residual dataset -> {resid_cache}")
    print(f"[Fold {fold_id}] saved residual metadata -> {meta_path}")
    return meta

def save_all_fold_results(args, all_results):
    summary_dir = osp.join("save", args.dataset_choice, args.model_variant)
    os.makedirs(summary_dir, exist_ok=True)

    summary = {
        "dataset_choice": args.dataset_choice,
        "model_variant": args.model_variant,
        "num_folds": len(all_results),
        "fold_results": all_results,
    }

    if len(all_results) > 0:
        summary["avg_best_val_rmse"] = float(np.mean([r["best_val_rmse"] for r in all_results]))
        summary["avg_test_mae"] = float(np.mean([r["test_mae"] for r in all_results]))
        summary["avg_test_rmse"] = float(np.mean([r["test_rmse"] for r in all_results]))
        summary["avg_test_mape"] = float(np.mean([r["test_mape"] for r in all_results]))

    summary_path = osp.join(summary_dir, "baseline_fold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved fold summary: {summary_path}")

def save_all_residual_results(args, all_residual_meta):
    summary_dir = osp.join("save", args.dataset_choice, args.model_variant)
    os.makedirs(summary_dir, exist_ok=True)

    summary = {
        "dataset_choice": args.dataset_choice,
        "model_variant": args.model_variant,
        "num_folds": len(all_residual_meta),
        "residual_datasets": all_residual_meta,
    }

    if len(all_residual_meta) > 0:
        summary["total_selected_samples"] = int(sum(m["num_selected_samples"] for m in all_residual_meta))
        summary["avg_selected_samples"] = float(np.mean([m["num_selected_samples"] for m in all_residual_meta]))

    summary_path = osp.join(summary_dir, "residual_fold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved residual fold summary: {summary_path}")


class LocalResCALDataset(Dataset):
    def __init__(self, pt_path):
        pack = torch.load(pt_path, map_location="cpu", weights_only=False)

        self.X = pack["X_local"]       # (S, Cx, N_local, T)
        self.U = pack["U_local"]       # (S, P,  N_local, T)
        self.R = pack["R_local"]       # (S, P,  N_local)
        self.BASE = pack["BASE_local"] # (S, P,  N_local)
        self.Y = pack["Y_local"]       # (S, P,  N_local)
        self.A = pack["A_local"]       # (S, N_local, N_local)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx].float(),
            self.U[idx].float(),
            self.R[idx].float(),
            self.BASE[idx].float(),
            self.Y[idx].float(),
            self.A[idx].float(),
        )

def load_all_local_rescal_datasets(args, folds, k_neighbors=5):
    datasets = []

    for fold in folds:
        fold_id = fold["fold_id"]
        save_dir = osp.join("save", args.dataset_choice, args.model_variant, f"fold_{fold_id}")
        pt_path = osp.join(save_dir, f"local_rescal_k{k_neighbors}.pt")

        if not osp.exists(pt_path):
            raise FileNotFoundError(f"Missing local ResCAL dataset: {pt_path}")

        datasets.append(LocalResCALDataset(pt_path))

    return ConcatDataset(datasets)


def build_rescal_model(args, num_nodes_local, cx):
    model = ResCAL(
        num_nodes=num_nodes_local,
        Tu=args.P,
        out_dim=args.P,
        Cx=cx,
        dh=32,
        dilation_channels=32,
        kernel_size=2,
        layers=8,
        dropout=0.0,
        support_len=1,
        gcn_order=1,
        dc=4,
        nc=8,
        de=32,
        tau=1.0,
    ).to(args.device)
    return model


def evaluate_rescal(model, loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X, U, R, BASE, Y, A in loader:
            X = X.to(args.device)
            U = U.to(args.device)
            R = R.to(args.device)
            A = A.to(args.device)

            R_hat, extras = model(X, U, A)
            loss = criterion(R_hat, R)

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def fit_rescal(model, train_loader, val_loader, args, save_dir):
    best_path = osp.join(save_dir, "rescal_best.pt")
    last_path = osp.join(save_dir, "rescal_last.pt")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = nn.MSELoss().to(args.device)

    initepoch = 0
    best_val = np.inf
    stop = 0

    if osp.exists(last_path):
        print("[INFO] Loading last ResCAL checkpoint:", last_path)
        ckpt = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        initepoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val_loss", np.inf))
        stop = int(ckpt.get("stop", 0))

        if initepoch >= args.epochs:
            print(f"[INFO] Last ResCAL checkpoint already trained to epoch {initepoch}, target epochs={args.epochs}.")
        else:
            print(f"[INFO] Resuming ResCAL from epoch {initepoch}, best_val_loss={best_val:.6f}")
    else:
        print("[INFO] No ResCAL checkpoint found. Training from scratch.")

    for epoch in range(initepoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for X, U, R, BASE, Y, A in train_loader:
            X = X.to(args.device)
            U = U.to(args.device)
            R = R.to(args.device)
            A = A.to(args.device)

            optimizer.zero_grad()
            R_hat, extras = model(X, U, A)
            loss = criterion(R_hat, R)
            loss.backward()
            optimizer.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_rescal(model, val_loader, criterion, args)

        print(f"[ResCAL Epoch {epoch+1:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            stop = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val,
                    "stop": stop,
                },
                best_path,
            )
            print(f"  New best ResCAL val loss: {best_val:.6f}")
        else:
            stop += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "stop": stop,
            },
            last_path,
        )

        if stop >= 8:
            print("Early stopping ResCAL (patience=8)")
            break

    final_path = best_path if osp.exists(best_path) else last_path
    ckpt = torch.load(final_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, float(ckpt.get("best_val_loss", best_val))


def pick_baseline_checkpoint(save_dir):
    best_path = osp.join(save_dir, "baseline_best.pt")
    last_path = osp.join(save_dir, "baseline_last.pt")
    legacy_path = osp.join(save_dir, "baseline_params.pt")

    if osp.exists(best_path):
        return best_path
    if osp.exists(last_path):
        return last_path
    if osp.exists(legacy_path):
        return legacy_path

    raise FileNotFoundError(
        f"No baseline checkpoint found in {save_dir}. "
        f"Checked: baseline_best.pt, baseline_last.pt, baseline_params.pt"
    )


def find_real_unseen_test_sensors(args, require_unseen_in_B=True, center_threshold=0.999):
    """
    Real unseen sensors:
      - no closure in A
      - optionally no closure in B
      - has closure in C
    """
    _, _, _, _, _, _, _, node_ids, timestamp_list = loadData_hyp2(args)
    t_A, t_B, t_C = timestamp_list

    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    con_df = con_df.loc[:, node_ids]

    nA = len(t_A)
    nB = len(t_B)
    nC = len(t_C)

    con_A = con_df.iloc[:nA].to_numpy(dtype=np.float32)
    con_B = con_df.iloc[nA:nA+nB].to_numpy(dtype=np.float32)
    con_C = con_df.iloc[nA+nB:nA+nB+nC].to_numpy(dtype=np.float32)

    # center-only closures
    A_has = (con_A >= center_threshold).any(axis=0)
    B_has = (con_B >= center_threshold).any(axis=0)
    C_has = (con_C >= center_threshold).any(axis=0)

    if require_unseen_in_B:
        mask = (~A_has) & (~B_has) & (C_has)
    else:
        mask = (~A_has) & (C_has)

    sensor_indices = np.where(mask)[0].tolist()

    print("\n=== Real Unseen Test Sensors ===")
    print(f"Require unseen in B: {require_unseen_in_B}")
    print(f"Found {len(sensor_indices)} sensors")
    print(f"Indices: {sensor_indices}")
    print("================================\n")

    return sensor_indices

def load_cached_split_pack(args, split_tag: str):
    cache_path = osp.join("data", "processed", args.dataset_choice, args.model_variant, f"hyp2_{split_tag}.pt")
    if not osp.exists(cache_path):
        raise FileNotFoundError(f"Missing cached split pack: {cache_path}")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def build_local_rescal_eval_dataset_for_fold_C(args, fold, k_neighbors=5, interval_minutes=5):
    """
    Build local closure-centered evaluation dataset on C/test for one fold.
    Returns a saved local pack path + meta.
    """
    fold_id = fold["fold_id"]
    heldout = fold["heldout_sensors"]

    print("\n" + "-" * 70)
    print(f"BUILDING LOCAL C EVAL DATA FOR FOLD {fold_id}")
    print(f"Held-out sensors: {heldout}")
    print("-" * 70)

    # Load full splits
    A_train_full, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    t_A, t_B, t_C = timestamp_list

    # Raw cached C timestamps
    C_pack_full = load_cached_split_pack(args, "C")
    x_time_C = C_pack_full[2]
    y_time_C = C_pack_full[3]

    # Raw speed / closure / distance
    speed_df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    dis_df = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    speed_df = speed_df.loc[:, node_ids]
    con_df = con_df.loc[:, node_ids]

    # A split for weekly normalization
    speed_df_A = speed_df.iloc[:len(t_A)].copy()
    con_df_A = con_df.iloc[:len(t_A)].copy()
    speed_df_A.index = pd.to_datetime(speed_df_A.index)
    con_df_A.index = pd.to_datetime(con_df_A.index)

    # C split for event detection
    start_C = len(t_A) + len(t_B)
    end_C = start_C + len(t_C)
    con_df_C = con_df.iloc[start_C:end_C].copy()
    con_df_C.index = pd.to_datetime(con_df_C.index)

    # Weekly baseline from A non-closure only
    mu = compute_weekly_mu_train_nonclosure_simple(
        speed_df_A=speed_df_A,
        closure_df_A=con_df_A,
        interval_minutes=interval_minutes,
    )

    # Load trained fold baseline
    save_dir = osp.join("save", args.dataset_choice, args.model_variant, f"fold_{fold_id}")
    baseline_path = pick_baseline_checkpoint(save_dir)

    net, _ = build_baseline_model(args, A_adj)
    ckpt = torch.load(baseline_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # 1) held-out closure-center events in C
    raw_events = find_heldout_closure_events(
        con_df_C,
        heldout_sensors=heldout,
        center_threshold=0.999,
    )
    print(f"[Fold {fold_id}] raw held-out C events: {len(raw_events)}")

    # 2) expand intervals
    merged = expand_and_merge_events(
        raw_events,
        total_len=len(con_df_C),
        pre_steps=args.P,
        post_steps=args.T,
    )
    print(f"[Fold {fold_id}] merged expanded C intervals: {len(merged)}")

    # 3) sample mask on C
    keep_mask = build_sample_mask_from_expanded_idx_intervals(
        x_time_C,
        y_time_C,
        split_index=con_df_C.index,
        merged_intervals=merged,
    )

    num_selected = int(keep_mask.sum())
    print(f"[Fold {fold_id}] C full samples           : {len(keep_mask)}")
    print(f"[Fold {fold_id}] selected eval samples    : {num_selected}")

    if num_selected == 0:
        return None, {
            "fold_id": fold_id,
            "heldout_sensors": heldout,
            "num_selected_samples": 0,
            "local_eval_cache": None,
        }

    # 4) build residual dataset on selected C samples
    selected_indices = [i for i, keep in enumerate(keep_mask) if keep]
    C_subset = Subset(C_test, selected_indices)
    C_loader_sel = DataLoader(C_subset, batch_size=args.batch_size, shuffle=False)

    anchor_timestamps = [yt[0] for yt, keep in zip(y_time_C, keep_mask) if keep]

    resid_cache = osp.join(save_dir, "residuals_unseen_fold_C.pt")
    build_residual_dataset(
        base_model=net,
        source_loader=C_loader_sel,
        mean=mean,
        std=std,
        args=args,
        anchor_timestamps=anchor_timestamps,
        cache_path=resid_cache,
        use_existing_closure_channel=True,
    )

    resid_pack = torch.load(resid_cache, map_location="cpu", weights_only=False)
    X = resid_pack["X"]        # (S, Cx, N, T)
    U = resid_pack["U"]        # (S, P,  N, T)
    R = resid_pack["R"]        # (S, P,  N)
    BASE = resid_pack["BASE"]  # (S, P,  N)
    Y = resid_pack["Y"]        # (S, P,  N)

    # 5) assign closure center per selected sample
    x_time_sel = [x_time_C[i] for i in selected_indices]
    y_time_sel = [y_time_C[i] for i in selected_indices]

    expanded_events = expand_events_without_merging(
        raw_events,
        total_len=len(con_df_C),
        pre_steps=args.P,
        post_steps=args.T,
    )

    X_local_list = []
    U_local_list = []
    R_local_list = []
    BASE_local_list = []
    Y_local_list = []
    A_local_list = []
    centers_list = []
    local_nodes_list = []

    for s in range(X.shape[0]):
        center = choose_center_for_sample(
            x_time_sel[s],
            y_time_sel[s],
            split_index=con_df_C.index,
            expanded_events=expanded_events,
        )

        if center is None:
            continue

        local_nodes = get_local_nodes_from_distance(
            center_node=center,
            dist_df=dis_df,
            k_neighbors=k_neighbors,
        )

        # Weekly-normalized speed using A weekly baseline
        x_times = pd.to_datetime(x_time_sel[s])
        weekly_base_local = weekly_mu_for_timestamps(
            mu=mu,
            timestamps=x_times,
            node_indices=local_nodes,
            interval_minutes=interval_minutes,
        )  # (N_local, T)

        raw_speed_local = X[s, 0, local_nodes, :]         # (N_local, T)
        speed_dev_local = raw_speed_local - weekly_base_local

        if X.shape[1] == 4:
            closure_local = X[s, 1, local_nodes, :]
            tod_local = X[s, 2, local_nodes, :]
            dow_local = X[s, 3, local_nodes, :]
            X_local = torch.stack(
                [speed_dev_local, closure_local, tod_local, dow_local],
                dim=0
            )
        elif X.shape[1] == 3:
            tod_local = X[s, 1, local_nodes, :]
            dow_local = X[s, 2, local_nodes, :]
            X_local = torch.stack(
                [speed_dev_local, tod_local, dow_local],
                dim=0
            )
        else:
            raise ValueError(f"Unexpected X channel count: {X.shape[1]}")

        U_local = U[s, :, local_nodes, :]
        R_local = R[s, :, local_nodes]
        BASE_local = BASE[s, :, local_nodes]
        Y_local = Y[s, :, local_nodes]
        A_local = slice_local_adjacency(A_adj, local_nodes)

        X_local_list.append(X_local)
        U_local_list.append(U_local)
        R_local_list.append(R_local)
        BASE_local_list.append(BASE_local)
        Y_local_list.append(Y_local)
        A_local_list.append(A_local)
        centers_list.append(int(center))
        local_nodes_list.append(local_nodes)

    if len(X_local_list) == 0:
        return None, {
            "fold_id": fold_id,
            "heldout_sensors": heldout,
            "num_selected_samples": 0,
            "local_eval_cache": None,
        }

    X_local = torch.stack(X_local_list)
    U_local = torch.stack(U_local_list)
    R_local = torch.stack(R_local_list)
    BASE_local = torch.stack(BASE_local_list)
    Y_local = torch.stack(Y_local_list)
    A_local = torch.stack(A_local_list)

    out = {
        "X_local": X_local,
        "U_local": U_local,
        "R_local": R_local,
        "BASE_local": BASE_local,
        "Y_local": Y_local,
        "A_local": A_local,
        "centers": centers_list,
        "local_nodes": local_nodes_list,
        "k_neighbors": int(k_neighbors),
        "fold_id": int(fold_id),
        "heldout_sensors": heldout,
    }

    save_path = osp.join(save_dir, f"local_rescal_eval_C_k{k_neighbors}.pt")
    torch.save(out, save_path)

    meta = {
        "fold_id": int(fold_id),
        "heldout_sensors": heldout,
        "k_neighbors": int(k_neighbors),
        "num_samples": int(X_local.shape[0]),
        "N_local": int(X_local.shape[2]),
        "save_path": save_path,
    }

    meta_path = osp.join(save_dir, f"local_rescal_eval_C_k{k_neighbors}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Fold {fold_id}] local eval C dataset saved -> {save_path}")
    return save_path, meta


def build_local_rescal_eval_dataset_for_sensor_list_C(
    args,
    sensor_list,
    k_neighbors=5,
    interval_minutes=5,
    save_name="local_rescal_eval_C_real_unseen_k5.pt",
):
    """
    Build local eval dataset on C for a given sensor list,
    using the FULL baseline (not fold baselines).
    """
    print("\n" + "-" * 70)
    print("BUILDING LOCAL C EVAL DATA FOR REAL UNSEEN SENSORS")
    print(f"Sensors: {sensor_list}")
    print("-" * 70)

    if len(sensor_list) == 0:
        return None, {"num_samples": 0, "save_path": None}

    # Full splits
    A_train_full, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    t_A, t_B, t_C = timestamp_list

    C_pack_full = load_cached_split_pack(args, "C")
    x_time_C = C_pack_full[2]
    y_time_C = C_pack_full[3]

    speed_df = pd.read_csv(osp.join(args.root, args.data), index_col=0)
    con_df = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    dis_df = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    speed_df = speed_df.loc[:, node_ids]
    con_df = con_df.loc[:, node_ids]

    # A split for weekly normalization
    speed_df_A = speed_df.iloc[:len(t_A)].copy()
    con_df_A = con_df.iloc[:len(t_A)].copy()
    speed_df_A.index = pd.to_datetime(speed_df_A.index)
    con_df_A.index = pd.to_datetime(con_df_A.index)

    # C split for event detection
    start_C = len(t_A) + len(t_B)
    end_C = start_C + len(t_C)
    con_df_C = con_df.iloc[start_C:end_C].copy()
    con_df_C.index = pd.to_datetime(con_df_C.index)

    mu = compute_weekly_mu_train_nonclosure_simple(
        speed_df_A=speed_df_A,
        closure_df_A=con_df_A,
        interval_minutes=interval_minutes,
    )

    # FULL baseline
    full_baseline_dir = osp.join("save", args.dataset_choice, args.model_variant, "full_baseline")
    baseline_path = pick_baseline_checkpoint(full_baseline_dir)

    net, _ = build_baseline_model(args, A_adj)
    ckpt = torch.load(baseline_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Events in C only for chosen sensors
    raw_events = find_heldout_closure_events(
        con_df_C,
        heldout_sensors=sensor_list,
        center_threshold=0.999,
    )
    print(f"[Real Unseen] raw C events: {len(raw_events)}")

    merged = expand_and_merge_events(
        raw_events,
        total_len=len(con_df_C),
        pre_steps=args.P,
        post_steps=args.T,
    )
    print(f"[Real Unseen] merged expanded C intervals: {len(merged)}")

    keep_mask = build_sample_mask_from_expanded_idx_intervals(
        x_time_C,
        y_time_C,
        split_index=con_df_C.index,
        merged_intervals=merged,
    )

    num_selected = int(keep_mask.sum())
    print(f"[Real Unseen] C full samples        : {len(keep_mask)}")
    print(f"[Real Unseen] selected eval samples : {num_selected}")

    if num_selected == 0:
        return None, {"num_samples": 0, "save_path": None}

    selected_indices = [i for i, keep in enumerate(keep_mask) if keep]
    C_subset = Subset(C_test, selected_indices)
    C_loader_sel = DataLoader(C_subset, batch_size=args.batch_size, shuffle=False)

    anchor_timestamps = [yt[0] for yt, keep in zip(y_time_C, keep_mask) if keep]

    save_dir = osp.join("save", args.dataset_choice, args.model_variant, "rescal_local_k5")
    os.makedirs(save_dir, exist_ok=True)

    resid_cache = osp.join(save_dir, "residuals_real_unseen_C.pt")
    build_residual_dataset(
        base_model=net,
        source_loader=C_loader_sel,
        mean=mean,
        std=std,
        args=args,
        anchor_timestamps=anchor_timestamps,
        cache_path=resid_cache,
        use_existing_closure_channel=True,
    )

    resid_pack = torch.load(resid_cache, map_location="cpu", weights_only=False)
    X = resid_pack["X"]
    U = resid_pack["U"]
    R = resid_pack["R"]
    BASE = resid_pack["BASE"]
    Y = resid_pack["Y"]

    x_time_sel = [x_time_C[i] for i in selected_indices]
    y_time_sel = [y_time_C[i] for i in selected_indices]

    expanded_events = expand_events_without_merging(
        raw_events,
        total_len=len(con_df_C),
        pre_steps=args.P,
        post_steps=args.T,
    )

    X_local_list, U_local_list = [], []
    R_local_list, BASE_local_list, Y_local_list = [], [], []
    A_local_list = []
    centers_list, local_nodes_list = [], []

    for s in range(X.shape[0]):
        center = choose_center_for_sample(
            x_time_sel[s],
            y_time_sel[s],
            split_index=con_df_C.index,
            expanded_events=expanded_events,
        )
        if center is None:
            continue

        local_nodes = get_local_nodes_from_distance(
            center_node=center,
            dist_df=dis_df,
            k_neighbors=k_neighbors,
        )

        x_times = pd.to_datetime(x_time_sel[s])
        weekly_base_local = weekly_mu_for_timestamps(
            mu=mu,
            timestamps=x_times,
            node_indices=local_nodes,
            interval_minutes=interval_minutes,
        )

        raw_speed_local = X[s, 0, local_nodes, :]
        speed_dev_local = raw_speed_local - weekly_base_local

        if X.shape[1] == 4:
            closure_local = X[s, 1, local_nodes, :]
            tod_local = X[s, 2, local_nodes, :]
            dow_local = X[s, 3, local_nodes, :]
            X_local = torch.stack([speed_dev_local, closure_local, tod_local, dow_local], dim=0)
        elif X.shape[1] == 3:
            tod_local = X[s, 1, local_nodes, :]
            dow_local = X[s, 2, local_nodes, :]
            X_local = torch.stack([speed_dev_local, tod_local, dow_local], dim=0)
        else:
            raise ValueError(f"Unexpected X channel count: {X.shape[1]}")

        U_local = U[s, :, local_nodes, :]
        R_local = R[s, :, local_nodes]
        BASE_local = BASE[s, :, local_nodes]
        Y_local = Y[s, :, local_nodes]
        A_local = slice_local_adjacency(A_adj, local_nodes)

        X_local_list.append(X_local)
        U_local_list.append(U_local)
        R_local_list.append(R_local)
        BASE_local_list.append(BASE_local)
        Y_local_list.append(Y_local)
        A_local_list.append(A_local)
        centers_list.append(int(center))
        local_nodes_list.append(local_nodes)

    if len(X_local_list) == 0:
        return None, {"num_samples": 0, "save_path": None}

    out = {
        "X_local": torch.stack(X_local_list),
        "U_local": torch.stack(U_local_list),
        "R_local": torch.stack(R_local_list),
        "BASE_local": torch.stack(BASE_local_list),
        "Y_local": torch.stack(Y_local_list),
        "A_local": torch.stack(A_local_list),
        "centers": centers_list,
        "local_nodes": local_nodes_list,
        "k_neighbors": int(k_neighbors),
        "sensor_list": sensor_list,
    }

    save_path = osp.join(save_dir, save_name)
    torch.save(out, save_path)

    meta = {
        "sensor_list": sensor_list,
        "k_neighbors": int(k_neighbors),
        "num_samples": int(out["X_local"].shape[0]),
        "N_local": int(out["X_local"].shape[2]),
        "save_path": save_path,
    }

    meta_path = osp.join(save_dir, save_name.replace(".pt", "_meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Real Unseen] local eval dataset saved -> {save_path}")
    return save_path, meta

@torch.no_grad()
def evaluate_rescal_on_local_pack(model, pt_path, args):
    """
    Compare baseline vs calibrated predictions on one saved local eval pack.
    """
    pack = torch.load(pt_path, map_location="cpu", weights_only=False)

    X = pack["X_local"]
    U = pack["U_local"]
    BASE = pack["BASE_local"]
    Y = pack["Y_local"]
    A = pack["A_local"]

    preds_resid = []
    base_all = []
    y_all = []

    model.eval()

    num_samples = X.shape[0]
    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)

        xb = X[start:end].to(args.device)
        ub = U[start:end].to(args.device)
        ab = A[start:end].to(args.device)

        R_hat, extras = model(xb, ub, ab)

        preds_resid.append(R_hat.detach().cpu().numpy())
        base_all.append(BASE[start:end].numpy())
        y_all.append(Y[start:end].numpy())

    R_hat = np.concatenate(preds_resid, axis=0)   # (S, P, N_local)
    BASE_np = np.concatenate(base_all, axis=0)
    Y_np = np.concatenate(y_all, axis=0)

    CAL_np = BASE_np + R_hat

    base_mae, base_rmse, base_mape, _ = metric(BASE_np, Y_np)
    cal_mae, cal_rmse, cal_mape, _ = metric(CAL_np, Y_np)

    return {
        "baseline_mae": float(base_mae),
        "baseline_rmse": float(base_rmse),
        "baseline_mape": float(base_mape),
        "calibrated_mae": float(cal_mae),
        "calibrated_rmse": float(cal_rmse),
        "calibrated_mape": float(cal_mape),
        "improvement_rmse": float(base_rmse - cal_rmse),
        "improvement_mae": float(base_mae - cal_mae),
        "num_samples": int(X.shape[0]),
    }

def main():
    args = build_args()
    apply_dataset_cfg(args)

    print("CONFIG:", args.dataset_choice, args.model_variant)
    print("Device:", args.device)

    folds, sensor_burden = build_closure_sensor_folds_from_args(args, k=5)

    print("\n=== Fold Summary ===")
    for f in folds:
        print(
            f"Fold {f['fold_id']}: "
            f"{len(f['heldout_sensors'])} sensors | "
            f"closure_steps={f['total_closure_steps']} | "
            f"sensors={f['heldout_sensors']}"
        )
    print("====================\n")

    # -------------------------
    # Stage 1: train all baselines
    # -------------------------
    all_results = []

    for fold in folds:
        try:
            results = train_one_fold(args, fold)
            all_results.append(results)
        except Exception as e:
            print(f"[ERROR] Fold {fold['fold_id']} failed during baseline training/eval: {e}")
            raise

    print("\n=== All Fold Baseline Results ===")
    for r in all_results:
        print(
            f"Fold {r['fold_id']} | "
            f"best_val_rmse={r['best_val_rmse']:.4f} | "
            f"test_mae={r['test_mae']:.4f} | "
            f"test_rmse={r['test_rmse']:.4f} | "
            f"test_mape={r['test_mape']:.4f}"
        )

    if len(all_results) > 0:
        print("\n=== Average Across Folds ===")
        print(f"avg_best_val_rmse = {np.mean([r['best_val_rmse'] for r in all_results]):.4f}")
        print(f"avg_test_mae      = {np.mean([r['test_mae'] for r in all_results]):.4f}")
        print(f"avg_test_rmse     = {np.mean([r['test_rmse'] for r in all_results]):.4f}")
        print(f"avg_test_mape     = {np.mean([r['test_mape'] for r in all_results]):.4f}")

    save_all_fold_results(args, all_results)

    # -------------------------
    # Stage 2: build unseen residual datasets from trained fold baselines
    # -------------------------
    print("\n" + "=" * 70)
    print("BUILDING UNSEEN-CLOSURE RESIDUAL DATASETS")
    print("=" * 70)

    all_residual_meta = []

    for fold in folds:
        try:
            meta = build_unseen_residuals_for_fold(args, fold)
            all_residual_meta.append(meta)
        except Exception as e:
            print(f"[ERROR] Fold {fold['fold_id']} failed during residual dataset build: {e}")
            raise

    print("\n=== Residual Dataset Summary ===")
    for m in all_residual_meta:
        print(
            f"Fold {m['fold_id']} | "
            f"events={m['num_raw_events']} | "
            f"merged_intervals={m['num_merged_intervals']} | "
            f"selected_samples={m['num_selected_samples']}"
        )

    if len(all_residual_meta) > 0:
        print(f"\nTotal selected unseen residual samples = {sum(m['num_selected_samples'] for m in all_residual_meta)}")

    save_all_residual_results(args, all_residual_meta)

    # -------------------------
    # Stage 3: build local closure-centered ResCAL datasets
    # -------------------------
    print("\n" + "=" * 70)
    print("BUILDING LOCAL CLOSURE-CENTERED RESCAL DATASETS")
    print("=" * 70)

    all_local_meta = []

    for fold in folds:
        try:
            save_path, meta = build_local_rescal_dataset_for_fold(
                args=args,
                fold=fold,
                k_neighbors=5,   # later you can change this easily
                interval_minutes=5,
            )
            all_local_meta.append(meta)
            print(f"[Fold {fold['fold_id']}] local dataset saved -> {save_path}")
        except Exception as e:
            print(f"[ERROR] Fold {fold['fold_id']} failed during local ResCAL prep: {e}")
            raise

    print("\n=== Local ResCAL Dataset Summary ===")
    for m in all_local_meta:
        print(
            f"Fold {m['fold_id']} | "
            f"k={m['k_neighbors']} | "
            f"num_samples={m['num_samples']} | "
            f"N_local={m['N_local']}"
        )

    # -------------------------
    # Stage 4: train ResCAL on all local datasets
    # -------------------------
    print("\n" + "=" * 70)
    print("TRAINING LOCAL RESCAL")
    print("=" * 70)

    full_local_ds = load_all_local_rescal_datasets(args, folds, k_neighbors=5)
    total_samples = len(full_local_ds)
    print(f"[INFO] Total local ResCAL samples: {total_samples}")

    val_size = max(1, int(0.1 * total_samples))
    train_size = total_samples - val_size

    train_ds, val_ds = random_split(
        full_local_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader_rescal = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader_rescal = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # inspect one sample to get dimensions
    X0, U0, R0, BASE0, Y0, A0 = full_local_ds[0]
    cx = X0.shape[0]
    n_local = X0.shape[1]

    print(f"[INFO] ResCAL input channels Cx: {cx}")
    print(f"[INFO] ResCAL local node count : {n_local}")
    print(f"[INFO] ResCAL train samples    : {len(train_ds)}")
    print(f"[INFO] ResCAL val samples      : {len(val_ds)}")

    rescal_save_dir = osp.join("save", args.dataset_choice, args.model_variant, "rescal_local_k5")
    os.makedirs(rescal_save_dir, exist_ok=True)

    rescal_model = build_rescal_model(args, num_nodes_local=n_local, cx=cx)
    rescal_model, best_rescal_val = fit_rescal(
        rescal_model,
        train_loader_rescal,
        val_loader_rescal,
        args,
        rescal_save_dir,
    )

    rescal_summary = {
        "dataset_choice": args.dataset_choice,
        "model_variant": args.model_variant,
        "k_neighbors": 5,
        "total_samples": total_samples,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "Cx": int(cx),
        "N_local": int(n_local),
        "best_val_loss": float(best_rescal_val),
        "checkpoint_best_path": osp.join(rescal_save_dir, "rescal_best.pt"),
        "checkpoint_last_path": osp.join(rescal_save_dir, "rescal_last.pt"),
    }

    rescal_summary_path = osp.join(rescal_save_dir, "rescal_train_summary.json")
    with open(rescal_summary_path, "w", encoding="utf-8") as f:
        json.dump(rescal_summary, f, indent=2)

    print(f"[INFO] Saved ResCAL summary: {rescal_summary_path}")

    # -------------------------
    # Stage 5: evaluate baseline vs calibrated on local C datasets
    # -------------------------
    print("\n" + "=" * 70)
    print("EVALUATING BASELINE VS CALIBRATED ON C")
    print("=" * 70)

    all_eval_results = []

    for fold in folds:
        try:
            eval_path, eval_meta = build_local_rescal_eval_dataset_for_fold_C(
                args=args,
                fold=fold,
                k_neighbors=5,
                interval_minutes=5,
            )

            if eval_path is None:
                print(f"[Fold {fold['fold_id']}] No evaluation samples found on C.")
                continue

            metrics_eval = evaluate_rescal_on_local_pack(
                model=rescal_model,
                pt_path=eval_path,
                args=args,
            )

            fold_eval = {
                "fold_id": fold["fold_id"],
                "heldout_sensors": fold["heldout_sensors"],
                "eval_path": eval_path,
                **metrics_eval,
            }
            all_eval_results.append(fold_eval)

            print(
                f"[Fold {fold['fold_id']}] "
                f"baseline_rmse={metrics_eval['baseline_rmse']:.4f} | "
                f"calibrated_rmse={metrics_eval['calibrated_rmse']:.4f} | "
                f"improvement={metrics_eval['improvement_rmse']:.4f}"
            )

        except Exception as e:
            print(f"[ERROR] Fold {fold['fold_id']} failed during C evaluation: {e}")
            raise

    print("\n=== C Evaluation Summary ===")
    for r in all_eval_results:
        print(
            f"Fold {r['fold_id']} | "
            f"base_rmse={r['baseline_rmse']:.4f} | "
            f"cal_rmse={r['calibrated_rmse']:.4f} | "
            f"improv={r['improvement_rmse']:.4f}"
        )

    if len(all_eval_results) > 0:
        avg_base_rmse = np.mean([r["baseline_rmse"] for r in all_eval_results])
        avg_cal_rmse = np.mean([r["calibrated_rmse"] for r in all_eval_results])
        avg_improv = np.mean([r["improvement_rmse"] for r in all_eval_results])

        print("\n=== Average C Evaluation ===")
        print(f"avg baseline RMSE   = {avg_base_rmse:.4f}")
        print(f"avg calibrated RMSE = {avg_cal_rmse:.4f}")
        print(f"avg RMSE improvement= {avg_improv:.4f}")

        eval_save_dir = osp.join("save", args.dataset_choice, args.model_variant, "rescal_local_k5")
        os.makedirs(eval_save_dir, exist_ok=True)

        eval_summary_path = osp.join(eval_save_dir, "rescal_eval_C_summary.json")
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold_results": all_eval_results,
                    "avg_baseline_rmse": float(avg_base_rmse),
                    "avg_calibrated_rmse": float(avg_cal_rmse),
                    "avg_rmse_improvement": float(avg_improv),
                },
                f,
                indent=2,
            )

        print(f"[INFO] Saved C evaluation summary: {eval_summary_path}")

    # -------------------------
    # Stage 5.5: train/load full baseline for real unseen evaluation
    # -------------------------
    print("\n" + "=" * 70)
    print("TRAINING / LOADING FULL BASELINE FOR REAL UNSEEN EVALUATION")
    print("=" * 70)

    full_baseline_results = train_full_baseline(args)

    # -------------------------
    # Stage 6: real unseen sensor evaluation with FULL baseline
    # -------------------------
    print("\n" + "=" * 70)
    print("REAL UNSEEN SENSOR EVALUATION (FULL BASELINE + RESCAL)")
    print("=" * 70)

    # stricter version: no closure in A or B, closure in C
    real_unseen_sensors = find_real_unseen_test_sensors(args, require_unseen_in_B=True, center_threshold=0.999)

    if len(real_unseen_sensors) == 0:
        print("[INFO] No real unseen sensors found under current definition.")
    else:
        eval_path_real, eval_meta_real = build_local_rescal_eval_dataset_for_sensor_list_C(
            args=args,
            sensor_list=real_unseen_sensors,
            k_neighbors=5,
            interval_minutes=5,
            save_name="local_rescal_eval_C_real_unseen_k5.pt",
        )

        if eval_path_real is None:
            print("[INFO] No local evaluation samples built for real unseen sensors.")
        else:
            real_metrics = evaluate_rescal_on_local_pack(
                model=rescal_model,
                pt_path=eval_path_real,
                args=args,
            )

            print("\n=== Real Unseen Sensor Evaluation ===")
            print(f"baseline_rmse   = {real_metrics['baseline_rmse']:.4f}")
            print(f"calibrated_rmse = {real_metrics['calibrated_rmse']:.4f}")
            print(f"rmse_improvement= {real_metrics['improvement_rmse']:.4f}")
            print(f"baseline_mae    = {real_metrics['baseline_mae']:.4f}")
            print(f"calibrated_mae  = {real_metrics['calibrated_mae']:.4f}")
            print(f"mae_improvement = {real_metrics['improvement_mae']:.4f}")
            print(f"num_samples     = {real_metrics['num_samples']}")

            real_save_dir = osp.join("save", args.dataset_choice, args.model_variant, "rescal_local_k5")
            os.makedirs(real_save_dir, exist_ok=True)

            real_summary_path = osp.join(real_save_dir, "rescal_eval_C_real_unseen_summary.json")
            with open(real_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sensor_list": real_unseen_sensors,
                        "eval_path": eval_path_real,
                        **real_metrics,
                    },
                    f,
                    indent=2,
                )

            print(f"[INFO] Saved real unseen evaluation summary: {real_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()