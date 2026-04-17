# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict


class DynamicTemporalBranchAttention(nn.Module):
    """
    Dynamic Temporal Branch Attention (方案 2).

    Computes per-timestep softmax weights over branches:
        w_t = softmax(FC2(ReLU(FC1(concat(F_1,...,F_K)_t))))   # [B, T', K]
        Y_t = Σ_i w_t_i × F_i_t

    Params (channels=512, K=4, hidden_dim=32): ~66K
    """

    def __init__(self, channels: int = 512, num_branches: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_branches = num_branches
        self.fc1 = nn.Linear(channels * num_branches, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_branches)

    def forward(self, branches: List[Tensor]) -> Tensor:
        """
        Args:
            branches: list of K tensors, each [B, C, T']
        Returns:
            fused: [B, C, T']
        """
        # [B, K*C, T'] -> [B, T', K*C]
        x = torch.cat(branches, dim=1).permute(0, 2, 1)
        h = F.relu(self.fc1(x))                  # [B, T', hidden_dim]
        w = F.softmax(self.fc2(h), dim=-1)       # [B, T', K]

        stacked = torch.stack(branches, dim=-1)  # [B, C, T', K]
        w = w.unsqueeze(1)                        # [B, 1, T', K]
        return (stacked * w).sum(dim=-1)          # [B, C, T']


class DualAxisSelectiveFusion(nn.Module):
    """
    Dual-Axis Selective Fusion (方案 4): SK-style channel attention + dynamic
    temporal attention. Each (channel, timestep) gets its own branch weights.

    Channel axis (Selective Kernel style):
        s = GAP(Σ_i F_i)
        z = ReLU(W_down s)
        a_ch_i = W_up_i z                       # [B, C] for each branch i
    Temporal axis:
        a_t = FC2(ReLU(FC1(concat(F_1,...,F_K)_t)))   # [B, T', K]
    Combination:
        w[c, t, i] = softmax_i(a_ch[c, i] + a_t[t, i])
        Y = Σ_i w_i ⊙ F_i

    Params (C=512, K=4, r=32, d=16): ~76K
    """

    def __init__(
        self,
        channels: int = 512,
        num_branches: int = 4,
        reduction: int = 32,
    ):
        super().__init__()
        self.num_branches = num_branches
        d = max(channels // reduction, 8)

        # Channel attention (SK)
        self.ch_down = nn.Linear(channels, d)
        self.ch_ups = nn.ModuleList([nn.Linear(d, channels) for _ in range(num_branches)])

        # Temporal attention
        self.t_fc1 = nn.Linear(channels * num_branches, d)
        self.t_fc2 = nn.Linear(d, num_branches)

    def forward(self, branches: List[Tensor]) -> Tensor:
        """
        Args:
            branches: list of K tensors, each [B, C, T']
        Returns:
            fused: [B, C, T']
        """
        # ---- Channel attention (SK) ----
        U = sum(branches)                                       # [B, C, T']
        s = U.mean(dim=-1)                                       # [B, C]
        z = F.relu(self.ch_down(s))                              # [B, d]
        ch_logits = torch.stack(
            [fc(z) for fc in self.ch_ups], dim=-1
        )                                                        # [B, C, K]

        # ---- Temporal attention ----
        x_t = torch.cat(branches, dim=1).permute(0, 2, 1)        # [B, T', K*C]
        t_logits = self.t_fc2(F.relu(self.t_fc1(x_t)))           # [B, T', K]

        # ---- Dual-axis combination ----
        # [B, C, 1, K] + [B, 1, T', K] -> [B, C, T', K]
        logits = ch_logits.unsqueeze(2) + t_logits.unsqueeze(1)
        w = F.softmax(logits, dim=-1)                            # [B, C, T', K]

        stacked = torch.stack(branches, dim=-1)                  # [B, C, T', K]
        return (stacked * w).sum(dim=-1)                          # [B, C, T']


class MultiBranchEncoder(nn.Module):
    """
    Multi-branch encoder (HoMi-style): body / lhand / rhand / head.
    Each branch is an independent Encoder (1D Conv + ResNet).
    Outputs are fused via one of:
        - 'mlp':       concat + 2-layer MLP (default)
        - 'temporal':  Dynamic Temporal Branch Attention (方案 2, ~66K params)
        - 'dual_axis': Dual-Axis Selective Fusion        (方案 4, ~76K params)
    """
    DEFAULT_BRANCH_SLICES = {
        'body':  (0, 30),    # upper_body_pose: 10 joints × 3
        'lhand': (30, 75),   # lhand_pose: 15 joints × 3
        'rhand': (75, 120),  # rhand_pose: 15 joints × 3
        'head':  (120, 133), # jaw_pose(3) + expr(10)
    }

    def __init__(self,
                 nfeats: int,
                 output_emb_width: int = 512,
                 down_t: int = 2,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 norm=None,
                 branch_slices=None,
                 fusion_type: str = 'mlp'):
        super().__init__()

        self.branch_slices = branch_slices if branch_slices is not None else self.DEFAULT_BRANCH_SLICES
        self.fusion_type = fusion_type

        # One encoder per branch
        self.branches = nn.ModuleDict()
        for name, (start, end) in self.branch_slices.items():
            branch_dim = end - start
            self.branches[name] = Encoder(
                branch_dim, width, down_t, stride_t,
                width, depth, dilation_growth_rate,
                activation=activation, norm=norm
            )

        n_branches = len(self.branch_slices)

        if fusion_type == 'mlp':
            # Fusion MLP: n_branches * width → output_emb_width
            self.fusion = nn.Sequential(
                nn.Linear(width * n_branches, output_emb_width),
                nn.ReLU(),
                nn.Linear(output_emb_width, output_emb_width),
            )
        elif fusion_type == 'temporal':
            # Dynamic Temporal Branch Attention (方案 2)
            self.fusion = DynamicTemporalBranchAttention(
                channels=width, num_branches=n_branches, hidden_dim=32
            )
            # Project from width → output_emb_width if needed
            self.out_proj = (
                nn.Linear(width, output_emb_width) if width != output_emb_width else nn.Identity()
            )
        elif fusion_type == 'dual_axis':
            # Dual-Axis Selective Fusion (方案 4)
            self.fusion = DualAxisSelectiveFusion(
                channels=width, num_branches=n_branches, reduction=32
            )
            self.out_proj = (
                nn.Linear(width, output_emb_width) if width != output_emb_width else nn.Identity()
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. "
                             f"Choose from ['mlp', 'temporal', 'dual_axis'].")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, nfeats, T]
        Returns:
            [B, output_emb_width, T']
        """
        branch_outputs = []
        for name, (start, end) in self.branch_slices.items():
            branch_feat = x[:, start:end, :]
            branch_out = self.branches[name](branch_feat)  # [B, width, T']
            branch_outputs.append(branch_out)

        if self.fusion_type == 'mlp':
            fused = torch.cat(branch_outputs, dim=1)        # [B, K*width, T']
            fused = fused.permute(0, 2, 1)                   # [B, T', K*width]
            fused = self.fusion(fused)                        # [B, T', output_emb_width]
            fused = fused.permute(0, 2, 1)                   # [B, output_emb_width, T']
        else:
            # Attention fusion: [B, width, T']
            fused = self.fusion(branch_outputs)
            # Channel projection if needed
            if not isinstance(self.out_proj, nn.Identity):
                fused = fused.permute(0, 2, 1)               # [B, T', width]
                fused = self.out_proj(fused)                  # [B, T', output_emb_width]
                fused = fused.permute(0, 2, 1)               # [B, output_emb_width, T']
        return fused


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.code_num = code_num

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
