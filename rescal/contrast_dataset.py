import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def aug_jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma

def aug_dropout_timesteps(x: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    if p <= 0:
        return x
    L = x.shape[0]
    keep = (torch.rand(L, device=x.device) > p).float().view(L, 1)
    return x * keep

class EventContrastiveDatasetV2(Dataset):
    def __init__(
        self,
        root_dir: str,
        history_len: int = 12,
        K_neighbors: int = 8,
        M_neg: int = 8,
        min_valid_frac: float = 0.8,
        neg_pool_max: int = 300,
        jitter_sigma: float = 0.03,
        dropout_p: float = 0.05,
        alpha_mode: str = "raw",     # "raw" or "norm"
        alpha_floor: float = 0.15,
        seed: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.root_dir = root_dir
        self.L = int(history_len)
        self.Kn = int(K_neighbors)
        self.M = int(M_neg)
        self.min_valid_frac = float(min_valid_frac)
        self.neg_pool_max = int(neg_pool_max)
        self.jitter_sigma = float(jitter_sigma)
        self.dropout_p = float(dropout_p)
        self.alpha_mode = str(alpha_mode)
        self.alpha_floor = float(alpha_floor)
        self.device = device

        # tensors
        self.B = torch.load(osp.join(root_dir, "behavior.pt"), weights_only=True).float()   # [Tg,N,C]
        self.event_mask = torch.load(osp.join(root_dir, "grid_event_mask.pt"), weights_only=True).bool()  # [Tg,N]
        self.Tg, self.N, self.C = self.B.shape

        # dict/meta
        self.meta = torch.load(osp.join(root_dir, "behavior_meta.pt"), weights_only=False)
        pos_pack = torch.load(osp.join(root_dir, "pos_nodes_weights.pt"), weights_only=False)
        self.pos_nodes = pos_pack["pos_nodes"]
        self.pos_w = pos_pack["pos_w"]

        # time context
        self.grid_timestamps = self.meta["grid_timestamps"]
        self.interval_minutes = int(self.meta["interval_minutes"])
        dt = pd.to_datetime(pd.Series(self.grid_timestamps))
        self.weekday = dt.dt.weekday.to_numpy(np.int16)
        self.slot = (dt.dt.hour * (60 // self.interval_minutes) + (dt.dt.minute // self.interval_minutes)).to_numpy(np.int16)

        # buckets for neg sampling
        self.time_buckets = {}
        for t in range(self.Tg):
            key = (int(self.weekday[t]), int(self.slot[t]))
            self.time_buckets.setdefault(key, []).append(t)

        # validity based on residual channel 0
        self.valid0 = ~torch.isnan(self.B[..., 0])  # [Tg,N]

        # events mapped to grid
        events_df = pd.read_csv(osp.join(root_dir, "events_df.csv"))
        mapping_idx = np.array(self.meta["mapping_idx"], dtype=np.int64)

        events = []
        for _, r in events_df.iterrows():
            node_idx = int(r["node_idx"])
            s_raw = int(r["start_idx"])
            e_raw = int(r["end_idx"])

            if s_raw < 0 or e_raw < 0 or s_raw >= len(mapping_idx) or e_raw >= len(mapping_idx):
                continue
            s_g = int(mapping_idx[s_raw])
            e_g = int(mapping_idx[e_raw])
            if s_g < 0 or e_g < 0 or e_g < s_g:
                continue
            events.append((node_idx, s_g, e_g))

        if len(events) == 0:
            raise RuntimeError("No events mapped to grid.")
        self.events = events

    def __len__(self):
        return len(self.events)

    def _window_ok(self, t_end: int, node_idx: int) -> bool:
        if node_idx < 0:
            return True
        t0 = t_end - self.L + 1
        if t0 < 0:
            return False
        frac = float(self.valid0[t0:t_end + 1, node_idx].float().mean().item())
        return frac >= self.min_valid_frac

    def _get_window(self, t_end: int, node_idx: int) -> torch.Tensor:
        if node_idx < 0:
            return torch.zeros((self.L, self.C), dtype=torch.float32)
        t0 = t_end - self.L + 1
        x = self.B[t0:t_end + 1, node_idx, :]
        return torch.nan_to_num(x, nan=0.0)

    def _sample_negative_times(self, node_idx: int, w: int, s: int, avoid_t: int) -> list:
        if node_idx < 0:
            return []
        cand = self.time_buckets.get((w, s), [])
        if len(cand) == 0:
            return []
        if len(cand) > self.neg_pool_max:
            cand = random.sample(cand, self.neg_pool_max)

        neg_times = []
        tries = 0
        while len(neg_times) < self.M and tries < 3000:
            tries += 1
            t = random.choice(cand)
            if t == avoid_t:
                continue
            if bool(self.event_mask[t, node_idx].item()):
                continue
            if not self._window_ok(t, node_idx):
                continue
            neg_times.append(t)
        return neg_times

    def _alpha_from_neighbor_weights(self, neighbor_ws: list) -> np.ndarray:
        # neighbor_ws length = Kn, may include zeros for padded neighbors
        w = np.array(neighbor_ws, dtype=np.float32)

        if self.alpha_mode == "norm":
            mx = float(np.max(w)) if w.size > 0 else 1.0
            if mx > 1e-8:
                w = w / mx

        # floor only for real neighbors; padded zeros should stay zero
        w = np.where(w > 0, np.maximum(w, self.alpha_floor), 0.0)
        return w

    def __getitem__(self, idx: int):
        anchor_i, s_g, e_g = self.events[idx]

        # choose anchor time inside event with valid anchor window
        t_anchor = None
        for _ in range(60):
            t = random.randint(s_g, e_g)
            if self._window_ok(t, anchor_i):
                t_anchor = t
                break
        if t_anchor is None:
            t_anchor = e_g

        w = int(self.weekday[t_anchor])
        s = int(self.slot[t_anchor])

        # sample up to Kn neighbors
        nbrs = self.pos_nodes[anchor_i] if anchor_i < len(self.pos_nodes) else []
        ws = self.pos_w[anchor_i] if anchor_i < len(self.pos_w) else []

        neighbor_nodes = []
        neighbor_ws = []

        if len(nbrs) > 0:
            if len(nbrs) >= self.Kn:
                pick = random.sample(range(len(nbrs)), self.Kn)
            else:
                pick = [random.randrange(len(nbrs)) for _ in range(self.Kn)]
            neighbor_nodes = [int(nbrs[p]) for p in pick]
            neighbor_ws = [float(ws[p]) for p in pick]

        # PAD to fixed Kn
        while len(neighbor_nodes) < self.Kn:
            neighbor_nodes.append(-1)   # dummy
            neighbor_ws.append(0.0)     # no weight

        S_nodes = [int(anchor_i)] + neighbor_nodes  # fixed length S=1+Kn

        # alpha: anchor=1, neighbors based on w (padded -> 0)
        w_scaled = self._alpha_from_neighbor_weights(neighbor_ws)  # [Kn]
        alpha = np.concatenate([[1.0], w_scaled], axis=0).astype(np.float32)  # [S]

        x1_list, x2_list, neg_list = [], [], []

        for node_j in S_nodes:
            # closure window at node_j and time t_anchor
            if self._window_ok(t_anchor, node_j):
                x = self._get_window(t_anchor, node_j)
            else:
                x = torch.zeros((self.L, self.C), dtype=torch.float32)

            x1 = aug_dropout_timesteps(aug_jitter(x, self.jitter_sigma), self.dropout_p)
            x2 = aug_dropout_timesteps(aug_jitter(x, self.jitter_sigma), self.dropout_p)
            x1_list.append(x1)
            x2_list.append(x2)

            # negatives for node_j
            neg_times = self._sample_negative_times(node_j, w, s, avoid_t=t_anchor)
            if len(neg_times) == 0:
                neg = torch.zeros((self.M, self.L, self.C), dtype=torch.float32)
            else:
                windows = [self._get_window(tn, node_j) for tn in neg_times]
                while len(windows) < self.M:
                    windows.append(windows[-1].clone())
                neg = torch.stack(windows[:self.M], dim=0)
            neg_list.append(neg)

        x1 = torch.stack(x1_list, dim=0).to(self.device)   # [S,L,C]
        x2 = torch.stack(x2_list, dim=0).to(self.device)   # [S,L,C]
        neg = torch.stack(neg_list, dim=0).to(self.device) # [S,M,L,C]
        alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device) # [S]

        return {
            "x1": x1,
            "x2": x2,
            "neg": neg,
            "alpha": alpha,
            "meta": {
                "anchor_node": int(anchor_i),
                "S_nodes": S_nodes,
                "t_anchor": int(t_anchor),
                "weekday": int(w),
                "slot": int(s),
                "event_start": int(s_g),
                "event_end": int(e_g),
            }
        }


if __name__ == "__main__":
    root = osp.join("data", "processed_contrastive", "Tyson")
    ds = EventContrastiveDatasetV2(root, history_len=12, K_neighbors=8, M_neg=8, device="cpu")
    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    b = next(iter(dl))
    print("x1:", b["x1"].shape)      # [B,S,L,C]
    print("x2:", b["x2"].shape)
    print("neg:", b["neg"].shape)
    print("alpha:", b["alpha"].shape)