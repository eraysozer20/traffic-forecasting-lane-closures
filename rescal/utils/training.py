# utils/training.py
import numpy as np
import torch
from utils.features import re_normalization
from utils.metrics import metric

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
            print(f"[Epoch {epoch+1:03d}] [{i+1:04d}/{len(loader)}] loss={loss.item():.4f} mae={mae:.4f} rmse={rmse:.4f}")

    return float(np.mean(losses))

@torch.no_grad()
def eval_rmse(net, loader, mean, std, args):
    net.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)
        out = net(x)
        preds.append(re_normalization(out.cpu().numpy(), mean, std))
        labels.append(re_normalization(y[:, :, :, 0].cpu().numpy(), mean, std))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    _, rmse, _, _ = metric(preds, labels)
    return rmse

def train_rescal_epoch(rescal, loader, support, optimizer, criterion, args, epoch, log_every=50):
    rescal.train()
    losses = []

    for i, (X, U, R, BASE, Y) in enumerate(loader):
        X, U, R = X.to(args.device), U.to(args.device), R.to(args.device)
        optimizer.zero_grad()
        R_hat, _ = rescal(X, U, support)
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
    se_sum, n_sum = 0.0, 0
    for X, U, R, BASE, Y in loader:
        X, U, R = X.to(args.device), U.to(args.device), R.to(args.device)
        R_hat, _ = rescal(X, U, support)
        se_sum += torch.sum((R_hat - R) ** 2).item()
        n_sum += R.numel()
    return float(np.sqrt(se_sum / max(n_sum, 1)))
