import os
import os.path as osp
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrast_dataset import EventContrastiveDatasetV2


# -----------------------
# Encoder (simple + strong baseline)
# -----------------------
class TemporalConvEncoder(nn.Module):
    """
    Input:  [B, L, C]
    Output: [B, D]
    """
    def __init__(self, in_channels: int, emb_dim: int = 64, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B,L,C] -> [B,C,L]
        x = x.transpose(1, 2)
        h = self.net(x)           # [B,D,L]
        h = self.pool(h).squeeze(-1)  # [B,D]
        h = F.normalize(h, dim=-1)
        return h


# -----------------------
# InfoNCE with explicit negatives
# -----------------------
def info_nce_with_neg(z1, z2, zneg, tau=0.2):
    """
    z1:   [B,D]
    z2:   [B,D] (positive)
    zneg: [B,M,D] (negatives for each sample)
    returns loss: [B]
    """
    # positive logit: [B,1]
    pos = torch.sum(z1 * z2, dim=-1, keepdim=True) / tau  # cosine since z normalized

    # negative logits: [B,M]
    neg = torch.einsum("bd,bmd->bm", z1, zneg) / tau

    logits = torch.cat([pos, neg], dim=1)  # [B, 1+M]
    labels = torch.zeros(z1.shape[0], dtype=torch.long, device=z1.device)  # positive is index 0
    return F.cross_entropy(logits, labels, reduction="none")


# -----------------------
# Training
# -----------------------
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Tyson", choices=["Tyson", "PEMSD8", "SIM_GRID"])
    p.add_argument("--root", type=str, default="data/processed_contrastive")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--history_len", type=int, default=12)
    p.add_argument("--K_neighbors", type=int, default=8)
    p.add_argument("--M_neg", type=int, default=8)

    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.2)

    # augmentation knobs
    p.add_argument("--jitter_sigma", type=float, default=0.03)
    p.add_argument("--dropout_p", type=float, default=0.05)

    # alpha scaling knobs
    p.add_argument("--alpha_mode", type=str, default="raw", choices=["raw", "norm"])
    p.add_argument("--alpha_floor", type=float, default=0.15)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root_dir = osp.join(args.root, args.dataset)
    ds = EventContrastiveDatasetV2(
        root_dir=root_dir,
        history_len=args.history_len,
        K_neighbors=args.K_neighbors,
        M_neg=args.M_neg,
        jitter_sigma=args.jitter_sigma,
        dropout_p=args.dropout_p,
        alpha_mode=args.alpha_mode,
        alpha_floor=args.alpha_floor,
        seed=args.seed,
        device="cpu",  # keep dataset on CPU; we move to device in training
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Infer C
    sample = ds[0]
    C = sample["x1"].shape[-1]
    print(f"Dataset: {args.dataset} | events: {len(ds)} | L={args.history_len} | C={C} | S=1+{args.K_neighbors} | M={args.M_neg}")

    enc = TemporalConvEncoder(in_channels=C, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # simple schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    # training loop
    for ep in range(1, args.epochs + 1):
        enc.train()
        losses = []
        losses_anchor = []
        losses_nbr = []

        for batch in dl:
            # batch tensors:
            # x1:    [B,S,L,C]
            # x2:    [B,S,L,C]
            # neg:   [B,S,M,L,C]
            # alpha: [B,S]
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            neg = batch["neg"].to(device)
            alpha = batch["alpha"].to(device)

            B, S, L, C = x1.shape
            M = neg.shape[2]

            # flatten sensor dimension so encoder sees [B*S, L, C]
            x1f = x1.reshape(B * S, L, C)
            x2f = x2.reshape(B * S, L, C)

            z1 = enc(x1f)  # [B*S, D]
            z2 = enc(x2f)  # [B*S, D]

            # negatives: [B,S,M,L,C] -> [B*S*M, L, C] -> [B*S*M, D] -> [B*S, M, D]
            negf = neg.reshape(B * S * M, L, C)
            zneg = enc(negf).reshape(B * S, M, -1)

            # InfoNCE per (sample,sensor): [B*S]
            loss_bs = info_nce_with_neg(z1, z2, zneg, tau=args.tau)  # [B*S]

            # reshape back to [B,S]
            loss_bs = loss_bs.reshape(B, S)

            # weighted mean over sensors then over batch
            # alpha already has anchor alpha=1, neighbors <=1
            weighted = (alpha * loss_bs).sum(dim=1) / (alpha.sum(dim=1) + 1e-6)  # [B]
            loss = weighted.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()

            losses.append(loss.item())

            # diagnostics: anchor vs neighbors
            anchor_loss = loss_bs[:, 0].mean().item()
            nbr_loss = loss_bs[:, 1:].mean().item() if S > 1 else float("nan")
            losses_anchor.append(anchor_loss)
            losses_nbr.append(nbr_loss)

        sched.step()

        print(
            f"Epoch {ep:03d} | "
            f"loss={np.mean(losses):.4f} | "
            f"anchor={np.mean(losses_anchor):.4f} | "
            f"nbr={np.mean(losses_nbr):.4f} | "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )

    # save checkpoint
    ckpt_path = osp.join(root_dir, "ckpt_contrastive.pt")
    os.makedirs(root_dir, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": enc.state_dict(),
            "args": vars(args),
        },
        ckpt_path
    )
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()