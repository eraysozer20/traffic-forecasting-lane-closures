# main.py
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler

from ResCAL import ResCAL
from STGCN_WZ import model as model_wz
from STGCN_WZ_NO import model as model_no

from utils.cli import build_args
from utils.configs import apply_dataset_cfg
from utils.data_loading import loadData_hyp2, load_y_t_list_cached
from utils.features import A_wave
from utils.training import train_epoch, eval_rmse, train_rescal_epoch, eval_rescal_rmse
from utils.residuals import build_residual_dataset
from utils.metrics import (
    collect_preds_on_E_resid,
    event_closure_metrics_from_arrays,
    metric
)
from utils.printing import (
    print_test_results_block,
    print_event_metrics_block,
)


def build_baseline_model(args, A_adj):
    A_wave_val = A_wave(args, A_adj)
    build_model = model_wz if args.model_variant == "wz" else model_no
    net = build_model(args, A_wave_val).to(args.device)
    return net, A_wave_val 

def fit_baseline(net, train_loader, val_loader, mean, std, args, save_path):
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = nn.MSELoss().to(args.device)

    initepoch = 0
    best_rmse = np.inf
    stop = 0

    if osp.exists(save_path):
        print("[INFO] Loading baseline checkpoint:", save_path)
        ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        initepoch = int(ckpt.get("epoch", 0))
        best_rmse = float(ckpt.get("best_rmse", np.inf))
    else:
        print("[INFO] No baseline checkpoint found. Training from scratch.")

    for epoch in range(initepoch, args.epochs):
        tr_loss = train_epoch(net, train_loader, optimizer, criterion, mean, std, args, epoch, log_every=50)
        val_rmse = eval_rmse(net, val_loader, mean, std, args)

        print(f"[Epoch {epoch+1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
        scheduler.step()

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
                save_path,
            )
            print(f"  New best baseline RMSE: {best_rmse:.4f}")
        else:
            stop += 1
            if stop >= 8:
                print("Early stopping baseline (patience=8)")
                break

    ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print("[Baseline] Loaded + frozen:", save_path)

    return net, best_rmse

def fit_rescal(rescal, train_loader, val_loader, support, args, save_path):
    if osp.exists(save_path):
        ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
        rescal.load_state_dict(ckpt["model_state_dict"])
        rescal.eval()
        best_val = float(ckpt.get("best_val_rmse", float("nan")))
        return rescal, best_val

    opt = optim.Adam(rescal.parameters(), lr=args.learning_rate)
    sched = lr_scheduler.StepLR(opt, step_size=40, gamma=0.8)
    crit = nn.MSELoss().to(args.device)

    best_val = np.inf
    stop = 0

    for epoch in range(args.epochs):
        tr_loss = train_rescal_epoch(rescal, train_loader, support, opt, crit, args, epoch, log_every=50)
        val_rmse = eval_rescal_rmse(rescal, val_loader, support, args)

        print(f"[ResCAL][Epoch {epoch+1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
        sched.step()

        if val_rmse < best_val:
            best_val = val_rmse
            stop = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rescal.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "best_val_rmse": best_val,
                },
                save_path,
            )
            print(f"  New best ResCAL val RMSE: {best_val:.4f}")
        else:
            stop += 1
            if stop >= 8:
                print("ResCAL early stopping (patience=8)")
                break

    ckpt = torch.load(save_path, map_location=args.device, weights_only=False)
    rescal.load_state_dict(ckpt["model_state_dict"])
    rescal.eval()
    return rescal, best_val


def main():
    args = build_args()
    apply_dataset_cfg(args)

    print("CONFIG:", args.dataset_choice, args.model_variant)
    print("Device:", args.device)

    # ---- data ----
    A_train, B_val, C_test, C_time, A_adj, mean, std, node_ids, timestamp_list = loadData_hyp2(args)
    train_loader_A = DataLoader(A_train, batch_size=args.batch_size, shuffle=True)
    val_loader_B = DataLoader(B_val, batch_size=args.batch_size, shuffle=False)
    test_loader_C = DataLoader(C_test, batch_size=args.batch_size, shuffle=False)

    # ---- baseline ----
    save_dir = osp.join("save", args.dataset_choice, args.model_variant)
    os.makedirs(save_dir, exist_ok=True)
    baseline_path = osp.join(save_dir, "baseline_params.pt")

    net, A_wave = build_baseline_model(args, A_adj)
    net, best_base = fit_baseline(net, train_loader_A, val_loader_B, mean, std, args, baseline_path)

    support = [A_wave.to(args.device).float()]

    # ---- residual datasets ----
    res_cache_dir = osp.join("data", "processed", args.dataset_choice, args.model_variant, "hyp2_residuals")
    os.makedirs(res_cache_dir, exist_ok=True)

    dsA = build_residual_dataset(net, train_loader_A, mean, std, args, timestamps=timestamp_list[0],
                                 cache_path=osp.join(res_cache_dir, "residual_A.pt"))
    dsB = build_residual_dataset(net, val_loader_B, mean, std, args, timestamps=timestamp_list[1],
                                 cache_path=osp.join(res_cache_dir, "residual_B.pt"))
    dsC = build_residual_dataset(net, test_loader_C, mean, std, args, timestamps=timestamp_list[2],
                                 cache_path=osp.join(res_cache_dir, "residual_C.pt"))

    resid_loader_A = DataLoader(dsA, batch_size=args.batch_size, shuffle=True)
    resid_loader_B = DataLoader(dsB, batch_size=args.batch_size, shuffle=False)
    resid_loader_C = DataLoader(dsC, batch_size=args.batch_size, shuffle=False)

    # ---- ResCAL ----
    rescal = ResCAL(
        num_nodes=args.num_nodes, Tu=args.P, out_dim=args.P,
        layers=4, dc=32, nc=16, de=16, tau=1.0, dropout=0.0,
        support_len=1, gcn_order=1,
    ).to(args.device)

    rescal_path = osp.join(save_dir, "rescal_params.pt")
    rescal, best_res = fit_rescal(rescal, resid_loader_A, resid_loader_B, support, args, rescal_path)

    # ---- test + event metrics ----
    pred_base_C, pred_final_C, label_C = collect_preds_on_E_resid(rescal, resid_loader_C, support, mean, std, args)

    # test_start boundary: A + B steps in original timeline
    test_start = len(timestamp_list[0]) + len(timestamp_list[1])

    df_con = pd.read_csv(osp.join(args.root, args.con_data), index_col=0)
    df_dis = pd.read_csv(osp.join(args.root, args.dis_data), index_col=0)

    y_t_C_full = load_y_t_list_cached(args, "hyp2_C.pt")
    y_t_C = y_t_C_full[:pred_base_C.shape[0]]

    event_base = event_closure_metrics_from_arrays(
        preds=pred_base_C, labels=label_C,
        df_con=df_con, df_dis=df_dis,
        test_start=test_start, radius_miles=3.0,
        y_t_list=y_t_C, P=args.P,
    )
    event_final = event_closure_metrics_from_arrays(
        preds=pred_final_C, labels=label_C,
        df_con=df_con, df_dis=df_dis,
        test_start=test_start, radius_miles=3.0,
        y_t_list=y_t_C, P=args.P,
    )

    print_event_metrics_block("EVENT METRICS (BASELINE on C)", event_base)
    print_event_metrics_block("EVENT METRICS (BASE + ResCAL on C)", event_final)

    # ---- overall TEST RESULTS (C) ----
    mae_b, rmse_b, mape_b, _ = metric(pred_base_C, label_C)
    mae_f, rmse_f, mape_f, _ = metric(pred_final_C, label_C)

    print_test_results_block(
        mae_b, rmse_b, mape_b,
        mae_f, rmse_f, mape_f,
        split_name="C",
    )


if __name__ == "__main__":
    main()
