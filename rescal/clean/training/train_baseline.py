# training/train_baseline.py

import os
import os.path as osp
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from ..data.data_utils import A_wave
from ..data.data_utils import metric, re_normalization
from ..models.STGCN_WZ import model_wz
from ..models.STGCN_WZ_NO import model_no


def make_baseline_loaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def train_epoch(net, loader, optimizer, criterion, mean, std, args, epoch, log_every=50):
    net.train()
    losses = []
    train_pred, train_label = [], []

    for i, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()
        out = net(x)  # (B,P,N)
        loss = criterion(out, y[:, :, :, 0])
        loss.backward()

        if getattr(args, "grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

        optimizer.step()

        losses.append(loss.item())

        pred = re_normalization(out.detach().cpu().numpy(), mean, std)
        label = re_normalization(y[:, :, :, 0].detach().cpu().numpy(), mean, std)
        train_pred.append(pred)
        train_label.append(label)

        if (i + 1) % log_every == 0:
            p = np.concatenate(train_pred, axis=0)
            l = np.concatenate(train_label, axis=0)
            mae, rmse, _, _ = metric(p, l)
            print(
                f"[Baseline][Epoch {epoch+1:03d}] "
                f"[{i+1:04d}/{len(loader):04d}] "
                f"loss={loss.item():.4f} mae={mae:.4f} rmse={rmse:.4f}"
            )

    return float(np.mean(losses))


@torch.no_grad()
def eval_rmse(net, loader, mean, std, args):
    net.eval()
    pred_list, label_list = [], []

    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)

        out = net(x)  # (B,P,N)

        pred = re_normalization(out.detach().cpu().numpy(), mean, std)
        label = re_normalization(y[:, :, :, 0].detach().cpu().numpy(), mean, std)

        pred_list.append(pred)
        label_list.append(label)

    p = np.concatenate(pred_list, axis=0)
    l = np.concatenate(label_list, axis=0)
    _, rmse, _, _ = metric(p, l)

    return float(rmse)


def build_baseline_model(args, A_adj):
    A_wave_val = A_wave(args, A_adj)
    build_model = model_wz if args.model_variant == "wz" else model_no
    net = build_model(args, A_wave_val).to(args.device)
    return net, A_wave_val


def save_checkpoint(path, epoch, model, optimizer, best_rmse):
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "best_rmse": best_rmse,
        },
        path,
    )


def fit_baseline(net, train_loader, val_loader, mean, std, args, save_path):
    root, ext = osp.splitext(save_path)
    best_path = save_path
    last_path = f"{root}_last{ext}"

    optimizer = optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 40),
        gamma=getattr(args, "lr_gamma", 0.8),
    )
    criterion = nn.MSELoss().to(args.device)

    init_epoch = 0
    best_rmse = np.inf
    stop = 0
    patience = getattr(args, "patience", 8)

    if osp.exists(best_path) and not getattr(args, "force_retrain", False):
        print("[INFO] Loading baseline checkpoint:", best_path)
        ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        best_rmse = float(ckpt.get("best_rmse", np.inf))
        return net, best_rmse, {"best": best_path, "last": last_path}

    print("[INFO] No baseline checkpoint found. Training from scratch.")

    for epoch in range(init_epoch, args.epochs):
        tr_loss = train_epoch(
            net, train_loader, optimizer, criterion, mean, std, args, epoch,
            log_every=getattr(args, "log_every", 50)
        )
        val_rmse = eval_rmse(net, val_loader, mean, std, args)

        print(f"[Baseline][Epoch {epoch+1:03d}] train_loss={tr_loss:.4f} | val_RMSE={val_rmse:.4f}")
        scheduler.step()

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            stop = 0
            save_checkpoint(best_path, epoch + 1, net, optimizer, best_rmse)
            print(f"  New best baseline RMSE: {best_rmse:.4f}")
        else:
            stop += 1
            if stop >= patience:
                print(f"Early stopping baseline (patience={patience})")
                break

    save_checkpoint(last_path, epoch + 1, net, optimizer, best_rmse)
    print("[Baseline] Saved last checkpoint:", last_path)

    ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    print("[Baseline] Loaded + frozen:", best_path)
    return net, best_rmse, {"best": best_path, "last": last_path}