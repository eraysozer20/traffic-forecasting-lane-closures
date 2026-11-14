import torch
import torch.nn as nn
import torch.nn.functional as F


class gwnet_nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class gwnet_linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gwnet_gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=1):
        super().__init__()
        self.nconv = gwnet_nconv()
        self.mlp = gwnet_linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet_WZ(nn.Module):
    """main3.py-compatible closure-aware GraphWaveNet. Input [B,T,N,F], output [B,P,N]."""
    def __init__(self, device, num_nodes, A, T=12, out_dim=6, dropout=0.0,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2, gcn_order=1):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.T = T
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.register_buffer("support", A.float())
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.closure_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.weight_con = nn.Parameter(torch.ones(T, num_nodes))
        self.bias_con = nn.Parameter(torch.zeros(num_nodes))
        self.weight_speed = nn.Parameter(torch.ones(T, num_nodes))
        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(gwnet_gcn(dilation_channels, residual_channels, dropout, support_len=1, order=gcn_order))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def _pad_time(self, x):
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        return x

    def _make_start_representation(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected [B,T,N,F], got {tuple(x.shape)}")
        if x.size(3) < 2:
            raise ValueError(f"GraphWaveNet_WZ expects at least 2 features [traffic, closure], got F={x.size(3)}")
        if x.size(1) != self.T:
            raise ValueError(f"Expected T={self.T}, got T={x.size(1)}")
        if x.size(2) != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, got N={x.size(2)}")
        traffic = x[:, :, :, 0:1].permute(0, 3, 1, 2)  # [B,1,T,N]
        closure = x[:, :, :, 1:2].permute(0, 3, 1, 2)  # [B,1,T,N]
        traffic = self.start_conv(traffic) * self.weight_speed
        closure = self.closure_conv(closure) * self.weight_con + self.bias_con
        h = traffic + closure  # [B,32,T,N]
        return self._pad_time(h.permute(0, 1, 3, 2))  # [B,32,N,T]

    def forward(self, x):
        support = [self.support.to(x.device)]
        x = self._make_start_representation(x)
        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, support)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[:, :, :, -1]


def model(args, A):
    net = GraphWaveNet_WZ(args.device, args.num_nodes, A.to(args.device), T=args.T, out_dim=args.P)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return net


gwnet = GraphWaveNet_WZ
