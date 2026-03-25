"""
Finite Scalar Quantization (FSQ) for Motion Tokenization

Implementation based on "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)
Adapted for sign language motion generation with SMPLX parameters.

FSQ eliminates learned codebooks entirely, using mathematically defined quantization.
This avoids codebook collapse and simplifies training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from einops import rearrange, pack, unpack


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for rounding.
    Forward: discrete (rounded), Backward: identity (gradient passes through)
    """
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    """
    Finite Scalar Quantization

    Quantizes continuous vectors to a fixed set of discrete codes without learned codebooks.
    Each dimension is independently quantized to one of L levels.

    Args:
        levels: List of quantization levels per dimension (e.g., [5, 5, 5, 5] for 625 codes)
        dim: Optional embedding dimension (if different from len(levels))
        num_codebooks: Number of codebooks (default: 1)
        keep_num_codebooks_dim: Whether to keep codebook dimension
        scale: Optional scale factor for inputs
        allowed_dtypes: Allowed data types for quantization
        force_quantization_f32: Force FP32 for stability
        return_indices: Whether to return indices
    """

    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        force_quantization_f32: bool = True,
        return_indices: bool = True,
        **kwargs
    ):
        super().__init__()

        # Validate FSQ configuration
        assert len(levels) > 0, "FSQ levels list cannot be empty"
        assert all(isinstance(l, int) and l > 0 for l in levels), \
            f"All FSQ levels must be positive integers, got {levels}"
        assert num_codebooks > 0, f"num_codebooks must be positive, got {num_codebooks}"

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim if keep_num_codebooks_dim is not None else (num_codebooks > 1)
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = dim if dim is not None else len(_levels)
        self.codebook_size = self._levels.prod().item()

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        )
        self.has_projections = has_projections

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32
        self.return_indices = return_indices

        # For calculating codebook indices
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous vectors to discrete levels.

        Args:
            z: Input tensor [..., codebook_dim]

        Returns:
            Quantized tensor in range [-1, 1] per dimension
        """
        # Move levels to same device as z
        if self._levels.device != z.device:
            self._levels = self._levels.to(z.device)
            self._basis = self._basis.to(z.device)
            self.implicit_codebook = self.implicit_codebook.to(z.device)

        # Quantization levels
        half_l = (self._levels - 1) * (1 + 1e-5) / 2

        # Bound and shift: tanh to [-1, 1], scale, shift
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        z = torch.tanh(z + shift) * half_l - offset

        # Round with straight-through estimator
        quantized = round_ste(z)

        # Normalize to [-1, 1]
        half_width = self._levels // 2
        quantized = quantized / half_width

        return quantized

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized codes to discrete indices.

        Args:
            zhat: Quantized codes [..., codebook_dim] in range [-1, 1]

        Returns:
            Indices [...] in range [0, codebook_size-1]
        """
        assert zhat.shape[-1] == self.codebook_dim

        # Denormalize from [-1, 1] to level indices
        half_width = self._levels // 2
        zhat = zhat * half_width

        # Shift to [0, L-1] per dimension
        zhat_shifted = zhat + half_width

        # Convert to linear index
        indices = (zhat_shifted * self._basis).sum(dim=-1).long()

        return indices

    def indices_to_codes(
        self,
        indices: torch.Tensor,
        project_out: bool = True
    ) -> torch.Tensor:
        """
        Convert discrete indices back to quantized codes.

        Args:
            indices: Discrete indices [...] in range [0, codebook_size-1]
            project_out: Whether to project to output dimension

        Returns:
            Quantized codes [..., dim] or [..., codebook_dim]
        """
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # Decompose linear index into per-dimension level indices
        indices_flattened = rearrange(indices, "... -> (...)")
        codes_non_centered = (
            (indices_flattened.unsqueeze(-1) // self._basis) % self._levels
        )
        codes = codes_non_centered - self._levels // 2

        # Normalize to [-1, 1]
        codes = codes / (self._levels // 2)

        # Reshape
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "(b ...) c d -> b ... (c d)", b=indices.shape[0])
        else:
            codes = rearrange(codes, "(b ... 1) d -> b ... d", b=indices.shape[0])

        # Project out if needed
        if project_out:
            codes = self.project_out(codes)

        # For image/video, rearrange spatial dimensions
        if is_img_or_video and codes.ndim == 2:
            codes = rearrange(codes, "b d -> b d 1 1")

        return codes

    def forward(
        self,
        z: torch.Tensor,
        return_loss_breakdown: bool = False,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass: quantize input and optionally return indices.

        Args:
            z: Input tensor [B, N, D] where N is spatial positions
            return_loss_breakdown: Whether to return loss breakdown
            mask: Optional mask for valid positions

        Returns:
            - Quantized output (same shape as input)
            - Indices [B, N] or [B, N, num_codebooks]
            - Loss breakdown tuple (commit_loss=0, perplexity)
        """
        # Validate input
        assert z.dim() >= 2, f"Input must be at least 2D, got shape {z.shape}"
        assert z.shape[-1] == self.dim, \
            f"Input last dimension must match FSQ dim {self.dim}, got {z.shape[-1]}"

        orig_dtype = z.dtype

        # Ensure correct dtype for quantization
        if self.force_quantization_f32 and orig_dtype not in self.allowed_dtypes:
            z = z.float()

        # Project in if needed
        z = self.project_in(z)

        # Split into num_codebooks if needed
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # Quantize
        codes = self.quantize(z)

        # Get indices
        indices = self.codes_to_indices(codes)

        # Flatten codes back
        codes = rearrange(codes, "b n c d -> b n (c d)")

        # Project out
        out = self.project_out(codes)

        # Restore original dtype
        out = out.type(orig_dtype)

        # Calculate perplexity (codebook usage)
        if self.return_indices:
            # Flatten indices for perplexity calculation
            if mask is not None:
                indices_flat = indices[mask].flatten()
            else:
                indices_flat = indices.flatten()

            # Calculate perplexity
            if len(indices_flat) > 0:
                n_e = self.codebook_size
                indices_count = torch.bincount(indices_flat, minlength=n_e)
                avg_probs = indices_count.float() / indices_count.sum()
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            else:
                perplexity = torch.tensor(0.0, device=z.device)
        else:
            perplexity = torch.tensor(0.0, device=z.device)

        # FSQ has no commitment loss
        commit_loss = torch.tensor(0.0, device=z.device)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        if return_loss_breakdown:
            return out, indices, (commit_loss, perplexity)

        return out, indices, commit_loss

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode indices to continuous codes (for VAE decoder).

        Args:
            indices: Token indices [B, N] or [B, N, num_codebooks]

        Returns:
            Continuous codes [B, N, dim]
        """
        codes = self.indices_to_codes(indices, project_out=True)
        return codes
