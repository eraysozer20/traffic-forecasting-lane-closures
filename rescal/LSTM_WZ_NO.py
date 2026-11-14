import torch
import torch.nn as nn


class LSTM_WZ_NO(nn.Module):
    """main3.py-compatible closure-unaware LSTM. Input [B,T,N,F], output [B,P,N]."""
    def __init__(self, device, T, P, num_nodes, hidden_dim=64, layers=2, dropout=0.0):
        super().__init__()
        self.device = device
        self.T = T
        self.P = P
        self.num_nodes = num_nodes
        self.start_conv = nn.Conv2d(1, 32, kernel_size=(1, 1))
        self.lstm = nn.LSTM(32, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, P)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input [B,T,N,F], got {tuple(x.shape)}")
        if x.size(1) != self.T:
            raise ValueError(f"Expected T={self.T}, got T={x.size(1)}")
        if x.size(2) != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got N={x.size(2)}")

        B, T, N, _ = x.shape
        traffic = x[:, :, :, 0:1].permute(0, 3, 2, 1)  # [B,1,N,T]
        h = self.start_conv(traffic)                   # [B,32,N,T]
        h = h.permute(0, 2, 3, 1).contiguous().view(B * N, T, 32)
        out, _ = self.lstm(h)
        out = self.fc(out[:, -1, :])                   # [B*N,P]
        out = out.view(B, N, self.P).permute(0, 2, 1)
        return out                                     # [B,P,N]


def model(args, A):
    net = LSTM_WZ_NO(args.device, args.T, args.P, args.num_nodes, hidden_dim=64, layers=2, dropout=0.0)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return net


LSTM_uni = LSTM_WZ_NO
