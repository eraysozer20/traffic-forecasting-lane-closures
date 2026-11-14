# utils/windowing.py
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def data_generate(dataset, args, df, save_file: str):
    cache_dir = osp.join("data", "processed", args.dataset_choice, args.model_variant)
    os.makedirs(cache_dir, exist_ok=True)
    path = osp.join(cache_dir, save_file)

    if not osp.exists(path):
        time_interval = 300
        dataset = dataset.transpose(1, 2, 0)  # (T, N, F)
        num_input, num_node, dims = dataset.shape
        num_sample = num_input - args.T - args.P + 1

        index = pd.DataFrame(df.index.values, columns=["time"])
        index.time = pd.to_datetime(index.time)

        x_set, y_set, x_time_set, y_time_set = [], [], [], []
        for i in tqdm(range(num_sample)):
            flag = 0
            x = np.zeros((args.T, num_node, dims), dtype=np.float32)
            y = np.zeros((args.P, num_node, dims), dtype=np.float32)
            x_time = np.empty((args.T,), dtype=object)
            y_time = np.empty((args.P,), dtype=object)

            x[0] = dataset[i]
            x_time[0] = index["time"].iloc[i]

            for j in range(1, args.T + args.P):
                ok = (index["time"].iloc[i + j] - index["time"].iloc[i + j - 1]).seconds == time_interval
                if not ok:
                    flag = 1
                    break

                if j <= (args.T - 1):
                    x[j] = dataset[i + j]
                    x_time[j] = index["time"].iloc[i + j]
                else:
                    y[j - args.T] = dataset[i + j]
                    y_time[j - args.T] = index["time"].iloc[i + j]

            if flag == 0:
                x_set.append(x)
                y_set.append(y)
                x_time_set.append(x_time)
                y_time_set.append(y_time)

        l = (len(x_set) // args.batch_size) * args.batch_size
        torch.save((x_set[:l], y_set[:l], x_time_set[:l], y_time_set[:l]), path)

    return torch.load(path, weights_only=False)
