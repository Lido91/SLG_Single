"""
Hierarchical Residual Quantizer with EMA codebook updates.

This module implements multi-scale residual vector quantization with:
- Single shared codebook across all scales
- EMA-based codebook updates with reset mechanism
- Gumbel sampling for soft quantization during training
- Learnable residual convolutions between scales (V2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def length_to_mask(length, max_len=None, device=None):
    """Convert sequence lengths to boolean mask."""
    if device is None:
        device = length.device if isinstance(length, torch.Tensor) else "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)

    if max_len is None:
        max_len = int(length.max().item())

    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def mean_flat(tensor, mask=None):
    """Take the mean over all non-batch dimensions with optional masking."""
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 3
        denom = mask.sum() * tensor.shape[-1]
        loss = (tensor * mask).sum() / denom
        return loss


def gumbel_sample(logits, temperature=1., stochastic=False, dim=-1, training=True):
    """Sample from logits with optional Gumbel noise."""
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    return ind


class HRQuantizeEMAReset(nn.Module):
    """
    Hierarchical Residual Quantizer with EMA codebook updates and reset mechanism.

    This quantizer uses multi-scale residual quantization where:
    - Each scale encodes the residual from previous scales
    - Different scales operate at different temporal resolutions
    - A single shared codebook is used across all scales
    - EMA updates maintain codebook quality without gradients

    Args:
        nb_code: Codebook size (number of codes)
        code_dim: Code embedding dimension
        mu: EMA decay rate (default: 0.99)
        scales: List of temporal scale factors (e.g., [1, 2, 4, 8])
    """
    def __init__(self, nb_code, code_dim, mu=0.99, scales=[1, 2, 4, 8]):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.scales = scales
        self.reset_codebook()

    def reset_codebook(self):
        """Reset codebook and related buffers."""
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        """Tile input to match codebook size with small noise."""
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        """Initialize codebook from input samples."""
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        """
        Quantize input to nearest codebook entries.

        Args:
            x: [N, C] input vectors
            sample_codebook_temp: Gumbel temperature for soft sampling

        Returns:
            code_idx: [N] indices of nearest codes
        """
        k_w = self.codebook.t()  # [C, nb_code]
        # Compute distances: ||x - c||^2 = ||x||^2 - 2<x,c> + ||c||^2
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True) -
            2 * torch.matmul(x, k_w) +
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        code_idx = gumbel_sample(
            -distance, dim=-1, temperature=sample_codebook_temp,
            stochastic=True, training=self.training
        )
        return code_idx

    def dequantize(self, code_idx):
        """
        Look up codebook entries for given indices.

        Args:
            code_idx: [N] code indices (-1 indicates padding)

        Returns:
            x: [N, C] codebook entries
        """
        mask = code_idx == -1
        code_idx = code_idx.masked_fill(mask, 0)
        x = F.embedding(code_idx, self.codebook)
        x[mask] = 0.
        return x

    def get_codebook_entry(self, indices):
        """Get codebook entries and transpose for Conv1d."""
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        """Compute codebook perplexity (usage diversity metric)."""
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """Update codebook using EMA with reset for unused codes."""
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        # Random codes for reset
        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # EMA update
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        # Reset unused codes
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        if len(code_idx) > self.nb_code * 5:
            self.codebook = usage * code_update + (1 - usage) * code_rand
        else:
            self.codebook = code_update

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """
        Quantize input across all scales (inference mode).

        Args:
            x: [B, C, T] input features
            m_lens: [B] sequence lengths
            return_latent: Whether to return continuous latent

        Returns:
            idx_list: List of [B, T_scale] indices per scale
            f_hat: [B, C, T] reconstructed features (if return_latent)
        """
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        idx_list = []

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, T, x.device)

        for i, scale in enumerate(self.scales):
            if m_lens is not None:
                residual = residual * full_scale_mask.unsqueeze(1)

            # Downsample residual
            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T // scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens // scale).long(), rest_down.shape[-1], x.device)
                mask_flat = rearrange(mask, 'n t -> (n t)')

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')

            # Quantize
            code_idx = self.quantize(rest_down)
            x_d = self.dequantize(code_idx)

            if m_lens is not None:
                x_d[~mask_flat] = 0
                code_idx[~mask_flat] = -1

            idx_list.append(rearrange(code_idx, '(n t) -> n t', n=N))

            # Upsample and accumulate
            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            residual = residual - up_x_d
            f_hat = f_hat + up_x_d

        if return_latent:
            return idx_list, f_hat
        return idx_list

    def get_codes_from_indices(self, indices_list):
        """
        Reconstruct continuous features from indices.

        Args:
            indices_list: List of [B, T_scale] indices per scale

        Returns:
            code: [B, T, C] reconstructed features
        """
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0

        for indices, scale in zip(indices_list, self.scales):
            N, _ = indices.shape
            indices = rearrange(indices, 'n t -> (n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code = code + up_x_d

        return code.permute(0, 2, 1)

    def forward(self, x, temperature=0., m_lens=None, start_drop=1, quantize_dropout_prob=0.):
        """
        Forward pass with training-time quantization.

        Args:
            x: [B, C, T] input features
            temperature: Gumbel temperature
            m_lens: [B] sequence lengths
            start_drop: Start index for quantizer dropout
            quantize_dropout_prob: Probability of dropping quantizers

        Returns:
            f_hat: [B, C, T] quantized features (with straight-through gradient)
            mean_vq_loss: Commitment loss
            perplexity: Codebook usage metric
        """
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        mean_vq_loss = 0.

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, T, x.device)
        else:
            full_scale_mask = torch.ones((N, T), device=x.device, dtype=torch.bool)

        all_rest_down = []
        all_code_indices = []
        all_mask = []

        # Quantizer dropout during training
        if self.training and random.random() < quantize_dropout_prob:
            start_drop_quantize_index = random.randint(start_drop, len(self.scales) - 1)
        else:
            start_drop_quantize_index = len(self.scales)

        for i, scale in enumerate(self.scales):
            if i >= start_drop_quantize_index:
                break

            residual = residual * full_scale_mask.unsqueeze(1)

            # Downsample residual
            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T // scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens // scale).long(), rest_down.shape[-1], x.device)
                all_mask.append(rearrange(mask, 'n t -> (n t)'))
            else:
                all_mask.append(torch.ones(N * rest_down.shape[-1], device=x.device, dtype=torch.bool))

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')

            # Initialize codebook on first batch
            if self.training and not self.init:
                self.init_codebook(rest_down[all_mask[-1]])

            # Quantize
            code_idx = self.quantize(rest_down, temperature)
            x_d = self.dequantize(code_idx)
            x_d[~all_mask[-1]] = 0

            all_rest_down.append(rest_down)
            all_code_indices.append(code_idx)

            # Upsample and accumulate
            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            residual = residual - up_x_d
            f_hat = f_hat + up_x_d

            # Commitment loss
            if m_lens is not None:
                mean_vq_loss += mean_flat((x - f_hat.detach()).pow(2), full_scale_mask.unsqueeze(1))
            else:
                mean_vq_loss += F.mse_loss(x, f_hat.detach())

        # Update codebook
        all_code_indices = torch.cat(all_code_indices, dim=0)
        all_rest_down = torch.cat(all_rest_down, dim=0)
        if m_lens is not None:
            all_mask = torch.cat(all_mask, dim=0)
            all_code_indices = all_code_indices[all_mask]
            all_rest_down = all_rest_down[all_mask]

        if self.training:
            perplexity = self.update_codebook(all_rest_down, all_code_indices)
        else:
            perplexity = self.compute_perplexity(all_code_indices)

        mean_vq_loss /= len(self.scales)

        # Straight-through gradient estimator
        f_hat = x + (f_hat - x).detach()

        return f_hat, mean_vq_loss, perplexity

    def idx_to_var_input(self, indices_list):
        """Get VAR training inputs from indices."""
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        next_scale_input = []

        for i in range(len(indices_list) - 1):
            indices = indices_list[i]
            N, _ = indices.shape
            indices = rearrange(indices, 'n t -> (n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code = code + up_x_d
            next_scale = F.interpolate(code, size=int(T // self.scales[i + 1]), mode='linear')
            next_scale_input.append(next_scale)

        return torch.cat(next_scale_input, dim=-1).permute(0, 2, 1)

    def get_next_var_input(self, level, indices, code, T):
        """Get next VAR inference input."""
        N, _ = indices.shape
        indices = rearrange(indices, 'n t -> (n t)')
        x_d = self.dequantize(indices)
        x_d = rearrange(x_d, '(n t) d -> n d t', n=N)

        if level != len(self.scales) - 1:
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            code = code + up_x_d
            next_scale = F.interpolate(code, size=int(T // self.scales[level + 1]), mode='linear')
        else:
            code = code + x_d
            next_scale = code

        return code, next_scale


class Phi(nn.Conv1d):
    """Learnable residual convolution for scale refinement."""
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h):
        return h.mul(1 - self.resi_ratio) + super().forward(h).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    """Single shared Phi for all scales."""
    def __init__(self, qresi):
        super().__init__()
        self.qresi = qresi

    def __getitem__(self, _):
        return self.qresi


class PhiPartiallyShared(nn.Module):
    """Partially shared Phi modules across scales."""
    def __init__(self, qresi_ls):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    """Non-shared Phi modules (one per scale)."""
    def __init__(self, qresi):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self):
        return f'ticks={self.ticks}'


class HRQuantizeEMAResetV2(nn.Module):
    """
    Hierarchical Residual Quantizer V2 with learnable residual refinement.

    Extends HRQuantizeEMAReset with:
    - Learnable Phi convolutions for scale refinement
    - Per-sample quantizer dropout
    - Configurable Phi sharing modes

    Args:
        nb_code: Codebook size
        code_dim: Code embedding dimension
        mu: EMA decay rate
        scales: List of temporal scale factors
        share_quant_resi: Phi sharing mode (0=non-shared, 1=shared, >1=partially shared)
        quant_resi: Residual ratio for Phi
    """
    def __init__(self, nb_code, code_dim, mu=0.99, scales=[1, 2, 4, 8], share_quant_resi=4, quant_resi=0.5):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.scales = scales
        self.reset_codebook()
        self.quant_resi_ratio = quant_resi

        # Setup Phi modules based on sharing mode
        if share_quant_resi == 0:  # Non-shared
            self.quant_resi = PhiNonShared([
                (Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(len(self.scales))
            ])
        elif share_quant_resi == 1:  # Fully shared
            self.quant_resi = PhiShared(
                Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # Partially shared
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([
                (Phi(code_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                for _ in range(share_quant_resi)
            ]))

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True) -
            2 * torch.matmul(x, k_w) +
            torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        code_idx = gumbel_sample(
            -distance, dim=-1, temperature=sample_codebook_temp,
            stochastic=True, training=self.training
        )
        return code_idx

    def dequantize(self, code_idx):
        mask = code_idx == -1
        code_idx = code_idx.masked_fill(mask, 0)
        x = F.embedding(code_idx, self.codebook)
        x[mask] = 0.
        return x

    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        if len(code_idx) > self.nb_code * 5:
            self.codebook = usage * code_update + (1 - usage) * code_rand
        else:
            self.codebook = code_update

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """Quantize across all scales (inference mode)."""
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        idx_list = []

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, T, x.device)
        else:
            full_scale_mask = torch.ones((N, T), device=x.device, dtype=torch.bool)

        for i, scale in enumerate(self.scales):
            residual = residual * full_scale_mask.unsqueeze(1)

            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T // scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens // scale).long(), rest_down.shape[-1], x.device)
                mask = rearrange(mask, 'n t -> (n t)')
            else:
                mask = torch.ones(N * rest_down.shape[-1], device=x.device, dtype=torch.bool)

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')

            code_idx = self.quantize(rest_down)
            x_d = self.dequantize(code_idx)
            x_d[~mask] = 0
            code_idx[~mask] = -1

            idx_list.append(rearrange(code_idx, '(n t) -> n t', n=N))

            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) - 1)](up_x_d)

            residual = residual - up_x_d
            f_hat = f_hat + up_x_d

        if return_latent:
            return idx_list, f_hat
        return idx_list

    def get_codes_from_indices(self, indices_list):
        """Reconstruct from indices with Phi refinement."""
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0

        for i, (indices, scale) in enumerate(zip(indices_list, self.scales)):
            N, _ = indices.shape
            indices = rearrange(indices, 'n t -> (n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) - 1)](up_x_d)
            code = code + up_x_d

        return code.permute(0, 2, 1)

    def forward(self, x, temperature=0., m_lens=None, start_drop=1, quantize_dropout_prob=0.):
        """Forward pass with per-sample quantizer dropout."""
        N, width, T = x.shape

        residual = x.clone()
        f_hat = torch.zeros_like(x)
        mean_vq_loss = 0.

        if m_lens is not None:
            full_scale_mask = length_to_mask(m_lens, T, x.device)
        else:
            full_scale_mask = torch.ones((N, T), device=x.device, dtype=torch.bool)

        all_rest_down = []
        all_code_indices = []
        all_mask = []

        # Per-sample quantizer dropout
        if self.training and quantize_dropout_prob != 0:
            n_quantizers = torch.randint(start_drop, len(self.scales) + 1, (N,))
            n_dropout = int(N * quantize_dropout_prob)
            n_quantizers[n_dropout:] = len(self.scales) + 1
            n_quantizers = n_quantizers.to(x.device)
        else:
            n_quantizers = torch.full((N,), len(self.scales) + 1, device=x.device)

        for i, scale in enumerate(self.scales):
            residual = residual * full_scale_mask.unsqueeze(1)
            keep_mask = (torch.full((N,), fill_value=i, device=x.device) < n_quantizers)

            if scale != 1:
                rest_down = F.interpolate(residual, size=int(T // scale), mode='area')
            else:
                rest_down = residual

            if m_lens is not None:
                mask = length_to_mask((m_lens // scale).long(), rest_down.shape[-1], x.device)
                mask = mask & keep_mask[:, None]
                all_mask.append(rearrange(mask, 'n t -> (n t)'))
            else:
                mask = keep_mask[:, None].expand(-1, rest_down.shape[-1])
                all_mask.append(rearrange(mask, 'n t -> (n t)'))

            rest_down = rearrange(rest_down, 'n c t -> (n t) c')

            if self.training and not self.init:
                self.init_codebook(rest_down[all_mask[-1]])

            code_idx = self.quantize(rest_down, temperature)
            x_d = self.dequantize(code_idx)
            x_d[~all_mask[-1]] = 0.

            all_rest_down.append(rest_down)
            all_code_indices.append(code_idx)

            x_d = rearrange(x_d, '(n t) c -> n c t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            if len(self.scales) > 1:
                up_x_d = self.quant_resi[i / (len(self.scales) - 1)](up_x_d)
            up_x_d[~keep_mask] = 0.

            residual = residual - up_x_d
            f_hat = f_hat + up_x_d

            # Commitment loss
            if m_lens is not None:
                loss_mask = full_scale_mask & keep_mask[:, None]
                mean_vq_loss += mean_flat((x - f_hat.detach()).pow(2), loss_mask.unsqueeze(1))
            else:
                mean_vq_loss += mean_flat((x - f_hat.detach()).pow(2), keep_mask[:, None, None])

        # Update codebook
        all_code_indices = torch.cat(all_code_indices, dim=0)
        all_rest_down = torch.cat(all_rest_down, dim=0)
        if len(all_mask) > 0:
            all_mask = torch.cat(all_mask, dim=0)
            all_code_indices = all_code_indices[all_mask]
            all_rest_down = all_rest_down[all_mask]

        if self.training:
            perplexity = self.update_codebook(all_rest_down, all_code_indices)
        else:
            perplexity = self.compute_perplexity(all_code_indices)

        mean_vq_loss /= len(self.scales)

        # Straight-through gradient
        f_hat = x + (f_hat - x).detach()

        return f_hat, mean_vq_loss, perplexity

    def idx_to_var_input(self, indices_list):
        """Get VAR training inputs with Phi refinement."""
        assert len(indices_list) == len(self.scales)
        T = indices_list[-1].shape[-1]
        code = 0.0
        next_scale_input = []

        for i in range(len(indices_list) - 1):
            indices = indices_list[i]
            N, _ = indices.shape
            indices = rearrange(indices, 'n t -> (n t)')
            x_d = self.dequantize(indices)
            x_d = rearrange(x_d, '(n t) d -> n d t', n=N)
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            up_x_d = self.quant_resi[i / (len(self.scales) - 1)](up_x_d)
            code = code + up_x_d
            next_scale = F.interpolate(code, size=int(T // self.scales[i + 1]), mode='linear')
            next_scale_input.append(next_scale)

        return torch.cat(next_scale_input, dim=-1).permute(0, 2, 1)

    def get_next_var_input(self, level, indices, code, T):
        """Get next VAR inference input with Phi refinement."""
        N, _ = indices.shape
        indices = rearrange(indices, 'n t -> (n t)')
        x_d = self.dequantize(indices)
        x_d = rearrange(x_d, '(n t) d -> n d t', n=N)

        if level != len(self.scales) - 1:
            up_x_d = F.interpolate(x_d, size=T, mode='linear')
            up_x_d = self.quant_resi[level / (len(self.scales) - 1)](up_x_d)
            code = code + up_x_d
            next_scale = F.interpolate(code, size=int(T // self.scales[level + 1]), mode='linear')
        else:
            code = code + x_d
            next_scale = code

        return code, next_scale
