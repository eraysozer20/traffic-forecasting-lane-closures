# utils/features.py
import numpy as np
import pandas as pd
import torch
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import eigs

def set_dis(df_dis: pd.DataFrame, lambda_: float) -> pd.DataFrame:
    x = 1 - (df_dis / lambda_) ** 2
    x[x == 1] = 0
    x[x < 0] = 0
    # set diagonal = 1 (explicit)
    np.fill_diagonal(x.values, 1)
    return x

def Construction_Layer(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    lens, num_vertices = x.shape
    out = torch.zeros((lens, num_vertices), dtype=torch.float64)
    for i in range(lens):
        closed = (x[i] == 1).nonzero()
        for j in range(len(closed)):
            e = adj[closed[j]]
            out[i] = torch.max(out[i], e)
    return out

def set_time(times, df: pd.DataFrame) -> torch.Tensor:
    time_pd = pd.DataFrame(times, columns=["Times"])
    time_pd["stamp"] = [pd.Timestamp(t) for t in time_pd["Times"].values]
    time_pd["Hour"] = time_pd["stamp"].apply(lambda d: d.hour)

    cut_hour = [-1, 5, 11, 14, 17, 21, 23]
    cut_labels = ["last night", "morning", "Noon", "afternoon", "evening", "Night"]
    time_pd["Hour_cut"] = pd.cut(time_pd["Hour"], bins=cut_hour, labels=cut_labels)

    time = LabelEncoder().fit_transform(time_pd["Hour_cut"])
    out = torch.zeros(size=df.shape, dtype=torch.long)
    for i in range(len(time)):
        out[i] = time[i]
    return out

def normalize(x, mean, std):
    x[0, :, :] = (x[0, :, :] - mean) / std
    return x

def re_normalization(x, mean, std):
    return x * std + mean

def A_wave(args, A: np.ndarray) -> torch.Tensor:
    assert A.shape[0] == A.shape[1]
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 1e-4] = 1e-4
    diag = np.reciprocal(np.sqrt(D))
    A_w = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return torch.from_numpy(A_w).float().to(args.device)
