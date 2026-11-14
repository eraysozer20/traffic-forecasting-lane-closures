import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Sequence, Tuple
from sklearn.preprocessing import LabelEncoder


# =========================================================
# Closure event utilities
# =========================================================

def extract_closure_events(closure_df: pd.DataFrame) -> List[Dict]:
    """
    Extract contiguous closure events for each sensor.

    Args:
        closure_df: DataFrame of shape (T, N)
                    values > 0 are treated as active closure/context

    Returns:
        List of dicts:
        [
            {"node": n, "start_idx": s, "end_idx": e},
            ...
        ]
    """
    events = []
    total_time, num_nodes = closure_df.shape

    for n in range(num_nodes):
        values = closure_df.iloc[:, n].to_numpy()
        active = False
        start = None

        for t in range(total_time):
            if values[t] > 0 and not active:
                active = True
                start = t
            elif values[t] <= 0 and active:
                events.append({
                    "node": n,
                    "start_idx": start,
                    "end_idx": t - 1,
                })
                active = False
                start = None

        if active:
            events.append({
                "node": n,
                "start_idx": start,
                "end_idx": total_time - 1,
            })

    return events


def expand_event_window(
    event: Dict,
    T: int,
    P: int,
    max_len: int,
) -> Tuple[int, int]:
    """
    Expand one event using:
        [start - P, end + T]

    Args:
        event: {"start_idx": int, "end_idx": int, ...}
        T: input length
        P: forecast horizon
        max_len: full sequence length

    Returns:
        (expanded_start, expanded_end)
    """
    start = max(0, event["start_idx"] - P)
    end = min(max_len - 1, event["end_idx"] + T)
    return start, end


def get_event_time_indices(
    events: List[Dict],
    T: int,
    P: int,
    max_len: int,
) -> np.ndarray:
    """
    Convert expanded events into a boolean time mask.

    Returns:
        mask: (max_len,) boolean
    """
    mask = np.zeros(max_len, dtype=bool)

    for event in events:
        s, e = expand_event_window(event, T=T, P=P, max_len=max_len)
        mask[s:e + 1] = True

    return mask


def filter_samples_by_event_mask(
    sample_indices: Sequence[int],
    event_mask: np.ndarray,
    T: int,
    P: int,
) -> np.ndarray:
    """
    Keep samples whose [input + output] window overlaps an event region.

    A sample starting at idx uses:
        input  = [idx, idx + T - 1]
        output = [idx + T, idx + T + P - 1]

    Full sample window:
        [idx, idx + T + P - 1]

    Args:
        sample_indices: iterable of sample start indices
        event_mask: boolean mask over time
        T: input length
        P: forecast horizon

    Returns:
        kept sample start indices as np.ndarray
    """
    kept = []

    for idx in sample_indices:
        idx = int(idx)
        end_idx = idx + T + P - 1

        if end_idx >= len(event_mask):
            continue

        if event_mask[idx:end_idx + 1].any():
            kept.append(idx)

    return np.asarray(kept, dtype=np.int64)


# =========================================================
# Weekly baseline utilities
# =========================================================

def compute_weekly_average(
    speed_df: pd.DataFrame,
    closure_df: pd.DataFrame = None,
    exclude_closure: bool = True,
) -> pd.DataFrame:
    """
    Compute weekly average per sensor grouped by:
        (day_of_week, time_of_day)

    Args:
        speed_df: DataFrame (T, N) with DatetimeIndex
        closure_df: DataFrame (T, N), optional
        exclude_closure: if True, ignore closure periods in averaging

    Returns:
        weekly_avg_df: DataFrame (T, N), aligned with speed_df
    """
    df = speed_df.copy()

    if exclude_closure and closure_df is not None:
        df = df.mask(closure_df > 0)

    group_keys = list(zip(df.index.dayofweek, df.index.hour, df.index.minute))

    temp = df.copy()
    temp["_group_key"] = group_keys

    weekly_avg = temp.groupby("_group_key").transform("mean")
    weekly_avg = weekly_avg.drop(columns=["_group_key"], errors="ignore")

    return weekly_avg


def apply_weekly_deviation(
    speed_df: pd.DataFrame,
    weekly_avg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute speed deviation from weekly average.

    Returns:
        speed_dev_df = speed_df - weekly_avg_df
    """
    return speed_df - weekly_avg_df


# =========================================================
# Time feature utilities
# =========================================================

def generate_time_features_from_indices(
    df_index: pd.Index,
    sample_start_indices: Sequence[int],
    num_nodes: int,
    T: int,
) -> torch.Tensor:
    """
    Build time features aligned to sample windows.

    Features:
        0 -> normalized time-of-day in [0, 1]
        1 -> normalized day-of-week in [0, 1]

    Args:
        df_index: full datetime index of the dataframe
        sample_start_indices: sample start positions in the full timeline
        num_nodes: N
        T: input length

    Returns:
        time_feats: torch.Tensor of shape (S, T, N, 2)
    """
    feat_list = []

    for start_idx in sample_start_indices:
        start_idx = int(start_idx)
        ts_window = df_index[start_idx:start_idx + T]

        tod = []
        dow = []

        for ts in ts_window:
            minutes = ts.hour * 60 + ts.minute
            tod.append(minutes / (24 * 60 - 1))
            dow.append(ts.dayofweek / 6.0)

        tod = np.asarray(tod, dtype=np.float32)             # (T,)
        dow = np.asarray(dow, dtype=np.float32)             # (T,)

        tod = np.tile(tod[:, None], (1, num_nodes))         # (T,N)
        dow = np.tile(dow[:, None], (1, num_nodes))         # (T,N)

        sample_feat = np.stack([tod, dow], axis=-1)         # (T,N,2)
        feat_list.append(sample_feat)

    feat_array = np.stack(feat_list, axis=0)                # (S,T,N,2)
    return torch.tensor(feat_array, dtype=torch.float32)

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
