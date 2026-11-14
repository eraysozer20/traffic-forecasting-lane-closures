import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
#  Diffusion Graph Convolution
# ----------------------------

class NConv(nn.Module):
    """
    Neighborhood convolution using einsum, like GraphWaveNet:
    x: (B, C, N, T)
    A: (N, N)
    returns: (B, C, N, T)
    """
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, A).contiguous()


class Linear1x1(nn.Module):
    """1x1 conv over (N, T), mixes channels only."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DiffusionGCN(nn.Module):
    """
    Multi-support, multi-order diffusion graph conv:
    out = concat([x, A1 x, A1^2 x, ..., A_s x, A_s^2 x, ...]) then 1x1 projection.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int, order: int = 1):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.dropout = dropout

        # After concatenating: (order * support_len + 1) * c_in channels
        c_in_total = (order * support_len + 1) * c_in
        self.mlp = Linear1x1(c_in_total, c_out)

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for A in support:
            x1 = self.nconv(x, A)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A)
                out.append(x1)

        h = torch.cat(out, dim=1)      # concat channels
        h = self.mlp(h)                # mix/compress channels
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# ----------------------------
#  Spatio-Temporal Block
# ----------------------------

class STBlock(nn.Module):
    """
    One encoder layer:
      - Gated temporal conv (dilated)
      - Diffusion GCN
      - Residual + BN
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

        # Temporal gated convs (like WaveNet)
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

        # Project back to residual_channels for residual path compatibility
        self.residual_proj = nn.Conv2d(
            in_channels=dilation_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        # Spatial mixing (GCN) operates on the gated output
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

        # --- Gated temporal conv ---
        # Note: temporal conv reduces T unless padding is applied before.
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        x = filt * gate                       # (B, dilation_channels, N, T')

        # --- Spatial GCN ---
        x = self.gcn(x, support)              # (B, residual_channels, N, T')

        # --- Residual connection ---
        # Need to align time dimension (use last T' steps of residual)
        T_out = x.size(-1)
        x = x + residual[..., -T_out:]        # (B, residual_channels, N, T')

        # --- Normalize ---
        x = self.bn(x)
        return x


# ----------------------------
#  ResCAL Encoder (no branches)
# ----------------------------

class ResCALEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        Tu: int,
        Cx: int = 3,          # <--- ADD (default 3)
        dh: int = 32,
        dilation_channels: int = 32,
        kernel_size: int = 2,
        layers: int = 8,
        dropout: float = 0.0,
        support_len: int = 1, # <--- SET DEFAULT 1 (we’ll use one support)
        gcn_order: int = 1,
        use_input_padding: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.Cx = Cx                  
        self.Tu = Tu
        self.dh = dh
        self.layers = layers
        self.kernel_size = kernel_size
        self.use_input_padding = use_input_padding

        # Input channels = (speed + timeofday + dayofweek) + Tu residual channels
        in_channels = Cx + Tu

        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=dh, kernel_size=(1, 1))


        # Build stacked ST blocks with increasing dilation (WaveNet style)
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
            dilation *= 2  # double each layer

        # Compute receptive field so we can left-pad time if needed
        # WaveNet receptive field approx: 1 + (kernel_size-1) * sum(dilations)
        dilations = [2**i for i in range(layers)]
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, X: torch.Tensor, U: torch.Tensor, support: list[torch.Tensor]):
        """
        X: (B, 2,  N, T)
        U: (B, Tu, N, T)
        support: list of adjacency matrices, each (N, N)

        returns:
          Z_seq: (B, dh, N, T')
          Z:     (B, dh, N)
        """
        assert X.size(1) == self.Cx, f"Expected X channels={self.Cx}, got {X.size(1)}"
        assert X.dim() == 4 and U.dim() == 4, "X and U must be (B, C, N, T)"
        assert X.size(2) == self.num_nodes and U.size(2) == self.num_nodes, "N mismatch"
        assert X.size(-1) == U.size(-1), "X and U must have same T"

        # --- concat along channel dimension (dim=1) ---
        x = torch.cat([X, U], dim=1)  # (B, Cx+Tu, N, T)

        # --- pad time on the left to match receptive field (optional) ---
        if self.use_input_padding:
            T = x.size(-1)
            if T < self.receptive_field:
                pad_len = self.receptive_field - T
                x = F.pad(x, (pad_len, 0, 0, 0))  # pad last dim (time): (left, right)

        # --- project to hidden channels ---
        x = self.start_conv(x)  # (B, dh, N, T_pad)

        # --- stacked spatio-temporal blocks ---
        for block in self.stacks:
            x = block(x, support)  # (B, dh, N, T')

        Z_seq = x
        Z = Z_seq[..., -1]  # last time step -> (B, dh, N)
        return Z_seq, Z


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
        """
        assert Z.dim() == 3, "Z must be (B, dh, N)"

        # add dummy time dimension → (B, dh, N, 1)
        x = Z.unsqueeze(-1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # remove dummy time dim → (B, out_dim, N)
        R_reg = x.squeeze(-1)
        return R_reg



def grouped_st_gumbel_softmax(logits: torch.Tensor, dc: int, nc: int, tau: float):
    """
    logits: (B, dc*nc, N)
    returns Q: (B, dc*nc, N) where each group of nc is (almost) one-hot
    """
    B, D, N = logits.shape
    assert D == dc * nc, f"logits channel dim must be dc*nc={dc*nc}, got {D}"

    # reshape -> (B, dc, nc, N)
    g = logits.view(B, dc, nc, N)

    # apply gumbel-softmax over the category axis (nc)
    # hard=True -> straight-through: forward is one-hot, backward is soft
    q = F.gumbel_softmax(g, tau=tau, hard=True, dim=2)  # (B, dc, nc, N)

    # back to (B, dc*nc, N)
    Q = q.view(B, dc * nc, N)
    return Q


class ResCALQuantizationHead(nn.Module):
    """
    Quantization branch:
      Z:      (B, dh, N)
      output: (B, out_dim, N)

    Internal:
      logits: (B, dc*nc, N)
      Q:      (B, dc*nc, N)  (ST Gumbel-Softmax)
      codebook E: (de, dc*nc)
      pattern vec: (B, de, N)
    """
    def __init__(self, dh: int, out_dim: int, dc: int, nc: int, de: int, tau: float = 1.0):
        super().__init__()
        self.dc = dc
        self.nc = nc
        self.de = de
        self.tau = tau

        # produce logits for categories
        self.conv1 = nn.Conv1d(dh, dh, kernel_size=1)
        self.conv2 = nn.Conv1d(dh, dc * nc, kernel_size=1)

        # codebook / embedding matrix
        # E: (de, dc*nc)
        self.E = nn.Parameter(torch.randn(de, dc * nc) * 0.02)

        # project pattern embedding to residual horizon
        self.out_proj = nn.Conv1d(de, out_dim, kernel_size=1)

    def forward(self, Z: torch.Tensor):
        """
        Z: (B, dh, N)

        returns:
          R_pat: (B, out_dim, N)
          extras: dict with logits and Q (useful for debugging)
        """
        assert Z.dim() == 3, "Z must be (B, dh, N)"

        # logits: (B, dc*nc, N)
        h = F.relu(self.conv1(Z))
        logits = self.conv2(h)

        # Q: (B, dc*nc, N) via grouped ST Gumbel-Softmax
        Q = grouped_st_gumbel_softmax(logits, dc=self.dc, nc=self.nc, tau=self.tau)

        # pattern embedding: (B, de, N) = E @ Q per node
        # E: (de, dc*nc), Q: (B, dc*nc, N) -> (B, de, N)
        pat = torch.einsum("ed,bdn->ben", self.E, Q)

        # project to (B, out_dim, N)
        R_pat = self.out_proj(pat)

        extras = {"logits": logits, "Q": Q, "pat": pat}
        return R_pat, extras


# ----------------------------
#  Full ResCAL (placeholder heads)
# ----------------------------

class ResCAL(nn.Module):
    """
    Full model skeleton: encoder + branches.
    """
    def __init__(
        self,
        num_nodes: int,
        Tu: int,
        out_dim: int,
        Cx: int = 3,
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
        _, Z = self.encoder(X, U, support)       # Z: (B, dh, N)

        R_reg = self.reg_head(Z)                 # (B, out_dim, N)
        R_pat, extras = self.q_head(Z)           # (B, out_dim, N)

        R_hat = R_reg + R_pat
        return R_hat, {"R_reg": R_reg, "R_pat": R_pat, **extras}
