import os
import os.path as osp
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


def make_rescal_loaders(
    train_dataset,
    val_dataset=None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def _prepare_support(support: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """
    If ResCAL expects batched adjacency/support, repeat it across batch.
    support:
        (N, N) -> returns (B, N, N)
        already batched -> returned as-is on device
    """
    support = support.to(device)

    if support.dim() == 2:
        support = support.unsqueeze(0).repeat(batch_size, 1, 1)

    return support


def train_rescal_epoch(
    rescal,
    train_loader,
    support,
    optimizer,
    criterion,
    args,
    epoch: int,
    log_every: int = 50,
) -> float:
    """
    One training epoch.

    Dataset batch format:
        X, U, R, BASE, Y
    We only train against R.
    """
    rescal.train()
    running_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(train_loader):
        X, U, R, BASE, Y = batch

        X = X.to(args.device)       # (B,Cx,N,T)
        U = U.to(args.device)       # (B,P,N,T)
        R = R.to(args.device)       # (B,P,N)

        A = _prepare_support(support, X.size(0), args.device)

        optimizer.zero_grad()

        # Adjust this line if your ResCAL forward signature differs
        R_hat = rescal(X, U, A)     # (B,P,N)

        loss = criterion(R_hat, R)
        loss.backward()

        if getattr(args, "grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(rescal.parameters(), args.grad_clip)

        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if (step + 1) % log_every == 0:
            print(
                f"[ResCAL][Epoch {epoch+1:03d}][Step {step+1:04d}/{len(train_loader):04d}] "
                f"loss={loss.item():.6f}"
            )

    if total_samples == 0:
        return 0.0

    return running_loss / total_samples


@torch.no_grad()
def eval_rescal_rmse(
    rescal,
    val_loader,
    support,
    args,
) -> float:
    """
    Validation metric on calibrated prediction:
        Y_cal = BASE + R_hat

    Returns:
        scalar RMSE over all batches
    """
    rescal.eval()

    sq_error_sum = 0.0
    count = 0

    for batch in val_loader:
        X, U, R, BASE, Y = batch

        X = X.to(args.device)
        U = U.to(args.device)
        BASE = BASE.to(args.device)
        Y = Y.to(args.device)

        A = _prepare_support(support, X.size(0), args.device)

        # Adjust this line if your ResCAL forward signature differs
        R_hat = rescal(X, U, A)         # (B,P,N)
        Y_cal = BASE + R_hat            # (B,P,N)

        sq_error_sum += torch.sum((Y_cal - Y) ** 2).item()
        count += Y.numel()

    if count == 0:
        return float("inf")

    rmse = np.sqrt(sq_error_sum / count)
    return float(rmse)


def save_checkpoint(
    path: str,
    epoch: int,
    model,
    optimizer,
    best_val_rmse: float,
):
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
        },
        path,
    )


def fit_rescal(
    rescal,
    train_loader,
    val_loader,
    support,
    args,
    save_path: str,
):
    """
    Trains ResCAL and saves:
      - best checkpoint
      - last checkpoint

    Returns:
      rescal loaded with BEST checkpoint
      best validation RMSE
      paths dict
    """
    best_path = save_path
    last_path = save_path.replace(".pt", "_last.pt")

    # optional resume/load-best shortcut
    if osp.exists(best_path) and not getattr(args, "force_retrain", False):
        ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
        rescal.load_state_dict(ckpt["model_state_dict"])
        rescal.eval()
        best_val = float(ckpt.get("best_val_rmse", float("nan")))
        return rescal, best_val, {"best": best_path, "last": last_path}

    optimizer = optim.Adam(
        rescal.parameters(),
        lr=args.learning_rate,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 40),
        gamma=getattr(args, "lr_gamma", 0.8),
    )

    criterion = nn.MSELoss().to(args.device)

    best_val = np.inf
    patience = getattr(args, "patience", 8)
    early_stop_counter = 0

    rescal = rescal.to(args.device)

    for epoch in range(args.epochs):
        train_loss = train_rescal_epoch(
            rescal=rescal,
            train_loader=train_loader,
            support=support,
            optimizer=optimizer,
            criterion=criterion,
            args=args,
            epoch=epoch,
            log_every=getattr(args, "log_every", 50),
        )

        val_rmse = eval_rescal_rmse(
            rescal=rescal,
            val_loader=val_loader,
            support=support,
            args=args,
        )

        print(
            f"[ResCAL][Epoch {epoch+1:03d}] "
            f"train_loss={train_loss:.6f} | val_RMSE={val_rmse:.6f}"
        )

        scheduler.step()

        # save BEST
        if val_rmse < best_val:
            best_val = val_rmse
            early_stop_counter = 0

            save_checkpoint(
                path=best_path,
                epoch=epoch + 1,
                model=rescal,
                optimizer=optimizer,
                best_val_rmse=best_val,
            )
            print(f"  New best ResCAL val RMSE: {best_val:.6f}")

        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"ResCAL early stopping (patience={patience})")
                break

    # save LAST
    save_checkpoint(
        path=last_path,
        epoch=epoch + 1,
        model=rescal,
        optimizer=optimizer,
        best_val_rmse=best_val,
    )
    print(f"Saved last checkpoint to: {last_path}")

    # reload BEST for return
    best_ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
    rescal.load_state_dict(best_ckpt["model_state_dict"])
    rescal.eval()

    return rescal, best_val, {"best": best_path, "last": last_path}