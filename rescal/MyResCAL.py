import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
#  Diffusion Graph Convolution
# ----------------------------

class NConv(nn.Module):
    """
    Neighborhood convolution.

    Supports:
      A: (N, N)       shared adjacency for whole batch
      A: (B, N, N)    one adjacency per sample
    """
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N, T)
        A: (N, N) or (B, N, N)

        returns: (B, C, N, T)
        """
        if A.dim() == 2:
            # shared support
            return torch.einsum("bcnt,nm->bcmt", x, A).contiguous()

        elif A.dim() == 3:
            # batched support
            return torch.einsum("bcnt,bnm->bcmt", x, A).contiguous()

        else:
            raise ValueError(f"A must have shape (N,N) or (B,N,N), got {tuple(A.shape)}")


class Linear1x1(nn.Module):
    """
    1x1 conv over (N, T), mixes channels only.
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DiffusionGCN(nn.Module):
    """
    Multi-support, multi-order diffusion graph conv.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int, order: int = 1):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.dropout = dropout

        c_in_total = (order * support_len + 1) * c_in
        self.mlp = Linear1x1(c_in_total, c_out)

    def forward(self, x: torch.Tensor, support) -> torch.Tensor:
        """
        x: (B, C, N, T)

        support can be:
          - list of supports, each (N,N) or (B,N,N)
          - single support (N,N) or (B,N,N)
        """
        if isinstance(support, torch.Tensor):
            support = [support]

        out = [x]

        for A in support:
            x1 = self.nconv(x, A)
            out.append(x1)

            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A)
                out.append(x1)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# ----------------------------
#  Spatio-Temporal Block
# ----------------------------

class STBlock(nn.Module):
    """
    One encoder block:
      - gated temporal conv
      - diffusion GCN
      - residual connection
      - batch norm
    """
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        support_len: int,
        gcn_order: int = 1,
    ):
        super().__init__()

        self.filter_conv = nn.Conv2d(
            in_channels=residual_channels,
            out_channels=dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            in_channels=residual_channels,
            out_channels=dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )

        self.gcn = DiffusionGCN(
            c_in=dilation_channels,
            c_out=residual_channels,
            dropout=dropout,
            support_len=support_len,
            order=gcn_order,
        )

        self.bn = nn.BatchNorm2d(residual_channels)

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, residual_channels, N, T)
        returns: (B, residual_channels, N, T')
        """
        residual = x

        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        x = filt * gate

        x = self.gcn(x, support)

        T_out = x.size(-1)
        x = x + residual[..., -T_out:]
        x = self.bn(x)
        return x


# ----------------------------
#  ResCAL Encoder
# ----------------------------

class ResCALEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        Tu: int,
        Cx: int = 4,
        dh: int = 32,
        dilation_channels: int = 32,
        kernel_size: int = 2,
        layers: int = 8,
        dropout: float = 0.0,
        support_len: int = 1,
        gcn_order: int = 1,
        use_input_padding: bool = True,
    ):
        """
        num_nodes: local patch size, not full network size
        Tu: number of residual-history channels (usually P)
        Cx: number of X channels
            default 4:
              [speed_minus_weekly_avg, closure_context, tod, dow]
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.Cx = Cx
        self.Tu = Tu
        self.dh = dh
        self.layers = layers
        self.kernel_size = kernel_size
        self.use_input_padding = use_input_padding

        in_channels = Cx + Tu
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=dh, kernel_size=(1, 1))

        self.stacks = nn.ModuleList()
        dilation = 1
        for _ in range(layers):
            self.stacks.append(
                STBlock(
                    residual_channels=dh,
                    dilation_channels=dilation_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    support_len=support_len,
                    gcn_order=gcn_order,
                )
            )
            dilation *= 2

        dilations = [2 ** i for i in range(layers)]
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, X: torch.Tensor, U: torch.Tensor, support: list[torch.Tensor]):
        """
        X: (B, Cx, N_local, T)
        U: (B, Tu, N_local, T)
        support: adjacency support(s), each either
         (N_local, N_local) or (B, N_local, N_local)

        returns:
          Z_seq: (B, dh, N_local, T')
          Z:     (B, dh, N_local)
        """
        assert X.size(1) == self.Cx, f"Expected X channels={self.Cx}, got {X.size(1)}"
        assert X.dim() == 4 and U.dim() == 4, "X and U must be (B, C, N, T)"
        assert X.size(2) == self.num_nodes and U.size(2) == self.num_nodes, "N mismatch"
        assert X.size(-1) == U.size(-1), "X and U must have same T"

        x = torch.cat([X, U], dim=1)  # (B, Cx+Tu, N, T)

        if self.use_input_padding:
            T = x.size(-1)
            if T < self.receptive_field:
                pad_len = self.receptive_field - T
                x = F.pad(x, (pad_len, 0, 0, 0))

        x = self.start_conv(x)

        for block in self.stacks:
            x = block(x, support)

        Z_seq = x
        Z = Z_seq[..., -1]  # (B, dh, N_local)
        return Z_seq, Z


# ----------------------------
#  Regression Head
# ----------------------------

class ResCALRegressionHead(nn.Module):
    """
    Regression branch:
      Z:      (B, dh, N)
      output: (B, out_dim, N)
    """
    def __init__(self, dh: int, out_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=dh,
            out_channels=dh,
            kernel_size=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=dh,
            out_channels=out_dim,
            kernel_size=(1, 1)
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, dh, N)
        returns: (B, out_dim, N)
        """
        assert Z.dim() == 3, "Z must be (B, dh, N)"

        x = Z.unsqueeze(-1)   # (B, dh, N, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        R_reg = x.squeeze(-1)  # (B, out_dim, N)
        return R_reg


# ----------------------------
#  Quantization helper
# ----------------------------

def grouped_st_gumbel_softmax(logits: torch.Tensor, dc: int, nc: int, tau: float):
    """
    logits: (B, dc*nc, N)
    returns Q: (B, dc*nc, N)
    where each group of nc behaves like one categorical selection.
    """
    B, D, N = logits.shape
    assert D == dc * nc, f"logits channel dim must be dc*nc={dc*nc}, got {D}"

    g = logits.reshape(B, dc, nc, N)
    q = F.gumbel_softmax(g, tau=tau, hard=True, dim=2)
    Q = q.reshape(B, dc * nc, N)
    return Q


# ----------------------------
#  Quantization Head
# ----------------------------

class ResCALQuantizationHead(nn.Module):
    """
    Quantization branch:
      Z:      (B, dh, N)
      output: (B, out_dim, N)
    """
    def __init__(self, dh: int, out_dim: int, dc: int, nc: int, de: int, tau: float = 1.0):
        super().__init__()
        self.dc = dc
        self.nc = nc
        self.de = de
        self.tau = tau

        self.conv1 = nn.Conv1d(dh, dh, kernel_size=1)
        self.conv2 = nn.Conv1d(dh, dc * nc, kernel_size=1)

        # codebook
        self.E = nn.Parameter(torch.randn(de, dc * nc) * 0.02)

        # project pattern embedding to output horizon
        self.out_proj = nn.Conv1d(de, out_dim, kernel_size=1)

    def forward(self, Z: torch.Tensor):
        """
        Z: (B, dh, N)

        returns:
          R_pat: (B, out_dim, N)
          extras: dict with logits / Q / pat
        """
        assert Z.dim() == 3, "Z must be (B, dh, N)"

        h = F.relu(self.conv1(Z))
        logits = self.conv2(h)  # (B, dc*nc, N)

        Q = grouped_st_gumbel_softmax(logits, dc=self.dc, nc=self.nc, tau=self.tau)

        # E: (de, dc*nc), Q: (B, dc*nc, N) -> pat: (B, de, N)
        pat = torch.einsum("ed,bdn->ben", self.E, Q)

        R_pat = self.out_proj(pat)  # (B, out_dim, N)

        extras = {"logits": logits, "Q": Q, "pat": pat}
        return R_pat, extras


# ----------------------------
#  Full ResCAL
# ----------------------------

class ResCAL(nn.Module):
    """
    Local closure-area ResCAL.

    Inputs:
      X       : (B, Cx, N_local, T)
      U       : (B, Tu, N_local, T)
      support : adjacency support(s), each either
          (N_local, N_local) or (B, N_local, N_local)

    Output:
      R_hat   : (B, P, N_local)
    """
    def __init__(
        self,
        num_nodes: int,
        Tu: int,
        out_dim: int,
        Cx: int = 4,
        dc: int = 4,
        nc: int = 8,
        de: int = 32,
        tau: float = 1.0,
        **encoder_kwargs
    ):
        super().__init__()
        self.encoder = ResCALEncoder(num_nodes=num_nodes, Tu=Tu, Cx=Cx, **encoder_kwargs)

        dh = self.encoder.dh
        self.reg_head = ResCALRegressionHead(dh=dh, out_dim=out_dim)
        self.q_head = ResCALQuantizationHead(dh=dh, out_dim=out_dim, dc=dc, nc=nc, de=de, tau=tau)

    def forward(self, X, U, support):
        """
        X:       (B, Cx, N_local, T)
        U:       (B, Tu, N_local, T)
        support: list[(N_local, N_local)]

        returns:
          R_hat: (B, P, N_local)
        """
        _, Z = self.encoder(X, U, support)    # (B, dh, N_local)

        R_reg = self.reg_head(Z)              # (B, P, N_local)
        R_pat, extras = self.q_head(Z)        # (B, P, N_local)

        R_hat = R_reg + R_pat
        return R_hat, {"R_reg": R_reg, "R_pat": R_pat, **extras}