"""
RVQ-VAE with Transformer Encoder/Decoder (no LGVQ).

Replaces Conv1D encoder/decoder with:
- Transformer encoder: part-aware patch embedding + learnable latent tokens +
  U-Net self-attention → fixed latent tokens
- Transformer decoder: learnable mask tokens + cross-attention to z_hat +
  U-Net → part-aware unpatch

Keeps RVQ quantizer unchanged. No text/CLIP dependency.
Use this to verify the transformer backbone works before adding LGVQ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

from .mgpt_rvq import RVQVae
from .tools.transformer_vae_blocks import (
    UNetTransformerEncoder,
    UNetTransformerDecoder,
)
from .tools.llamagen_blocks import precompute_freqs_cis_1d


class RVQVaeTransformer(RVQVae):
    """
    RVQ-VAE with Transformer encoder/decoder.

    Extends RVQVae — same RVQ quantizer, same encode()/decode() interface.
    No CLIP, no LGVQ. Pure motion autoencoder with transformer backbone.
    """

    # Body part dimension ranges within the 133D SMPL-X feature vector
    BODY_DIMS = (0, 33)
    LHAND_DIMS = (33, 78)
    RHAND_DIMS = (78, 123)
    FACE_DIMS = (123, 133)

    def __init__(
        self,
        # Transformer params
        latent_dim: int = 512,
        num_latent_tokens: int = 49,
        patch_size: int = 4,
        num_layers: int = 9,
        n_head: int = 8,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        # Parent params passed through
        **kwargs,
    ) -> None:
        # Initialize parent: creates Conv1D encoder/decoder + RVQ
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size

        # Remove Conv1D encoder/decoder (replaced by transformer)
        del self.encoder
        del self.decoder

        # --- Part-aware patch embedding ---
        body_d = self.BODY_DIMS[1] - self.BODY_DIMS[0]
        lhand_d = self.LHAND_DIMS[1] - self.LHAND_DIMS[0]
        rhand_d = self.RHAND_DIMS[1] - self.RHAND_DIMS[0]
        face_d = self.FACE_DIMS[1] - self.FACE_DIMS[0]

        self.body_patch_embed = nn.Conv1d(body_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.lhand_patch_embed = nn.Conv1d(lhand_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.rhand_patch_embed = nn.Conv1d(rhand_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.face_patch_embed = nn.Conv1d(face_d, latent_dim, kernel_size=patch_size, stride=patch_size)

        # --- Learnable latent tokens ---
        scale = latent_dim ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_latent_tokens, latent_dim))

        # --- Encoder output projection ---
        self.latent_proj = nn.Conv1d(latent_dim, self.code_dim, kernel_size=3, stride=1, padding=1)

        # --- Transformer encoder ---
        self.transformer_encoder = UNetTransformerEncoder(
            dim=latent_dim, n_head=n_head, num_layers=num_layers,
            dropout=dropout, drop_path_rate=drop_path_rate,
        )

        # --- Decoder input projection ---
        self.z_hat_proj = nn.Conv1d(self.code_dim, latent_dim, kernel_size=3, stride=1, padding=1)

        # --- Learnable mask token + positional embedding ---
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, latent_dim))
        max_output_tokens = 400 // patch_size
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, max_output_tokens, latent_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        # --- Transformer decoder ---
        self.transformer_decoder = UNetTransformerDecoder(
            dim=latent_dim, n_head=n_head, num_layers=num_layers,
            dropout=dropout, drop_path_rate=drop_path_rate,
        )

        # --- Part-aware unpatch ---
        self.body_unpatch = nn.ConvTranspose1d(latent_dim, body_d, kernel_size=patch_size, stride=patch_size)
        self.lhand_unpatch = nn.ConvTranspose1d(latent_dim, lhand_d, kernel_size=patch_size, stride=patch_size)
        self.rhand_unpatch = nn.ConvTranspose1d(latent_dim, rhand_d, kernel_size=patch_size, stride=patch_size)
        self.face_unpatch = nn.ConvTranspose1d(latent_dim, face_d, kernel_size=patch_size, stride=patch_size)

        # --- RoPE cache ---
        self._freqs_cache = None

        # Re-initialize new parameters (skip RVQ codebooks)
        self.apply(self._init_transformer_weights)

    def _init_transformer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        head_dim = self.latent_dim // self.transformer_encoder.input_blocks[0].attn.n_head
        if self._freqs_cache is None or self._freqs_cache.shape[0] < seq_len:
            self._freqs_cache = precompute_freqs_cis_1d(
                seq_len=seq_len, n_elem=head_dim, base=100.0, cls_token_num=0,
            )
        return self._freqs_cache[:seq_len].to(device)

    def _patch_embed(self, x: Tensor) -> Tensor:
        """Part-aware patch embedding. x: (B, D, T) → (B, T/patch, latent_dim)"""
        body = self.body_patch_embed(x[:, self.BODY_DIMS[0]:self.BODY_DIMS[1], :])
        lhand = self.lhand_patch_embed(x[:, self.LHAND_DIMS[0]:self.LHAND_DIMS[1], :])
        rhand = self.rhand_patch_embed(x[:, self.RHAND_DIMS[0]:self.RHAND_DIMS[1], :])
        face = self.face_patch_embed(x[:, self.FACE_DIMS[0]:self.FACE_DIMS[1], :])
        return (body + lhand + rhand + face).transpose(1, 2)

    def _unpatch(self, x: Tensor) -> Tensor:
        """Part-aware unpatch. x: (B, latent_dim, T/patch) → (B, D, T)"""
        body = self.body_unpatch(x)
        lhand = self.lhand_unpatch(x)
        rhand = self.rhand_unpatch(x)
        face = self.face_unpatch(x)
        return torch.cat([body, lhand, rhand, face], dim=1)

    def forward_encoder(self, features: Tensor) -> Tensor:
        """Encode motion → fixed-length latent. (B, T, D) → (B, code_dim, 49)"""
        B, T, D = features.shape
        patches = self._patch_embed(features.transpose(1, 2))  # (B, T/patch, latent_dim)
        latent = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)
        enc_input = torch.cat([latent, patches], dim=1)
        freqs = self._get_freqs(enc_input.shape[1], enc_input.device)
        enc_output = self.transformer_encoder(enc_input, freqs)
        z = enc_output[:, :self.num_latent_tokens, :]
        return self.latent_proj(z.transpose(1, 2))  # (B, code_dim, 49)

    def forward_decoder(self, z_hat: Tensor, target_len: int) -> Tensor:
        """Decode quantized codes → motion. (B, code_dim, 49) → (B, T, D)"""
        B = z_hat.shape[0]
        num_out = target_len // self.patch_size
        memory = self.z_hat_proj(z_hat).transpose(1, 2)  # (B, 49, latent_dim)
        mask_tokens = self.mask_token.expand(B, num_out, -1) + self.dec_pos_embed[:, :num_out, :]
        freqs = self._get_freqs(num_out, z_hat.device)
        dec_output = self.transformer_decoder(mask_tokens, memory, freqs)
        motion = self._unpatch(dec_output.transpose(1, 2))  # (B, D, T)
        return motion.transpose(1, 2)  # (B, T, D)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Same interface as RVQVae.forward().

        Returns: (x_out, loss_commit, perplexity)
        """
        B, T, D = features.shape
        x_encoder = self.forward_encoder(features)
        x_quantized, all_indices, loss_commit, perplexity = self.quantizer(x_encoder)
        x_out = self.forward_decoder(x_quantized, T)
        if x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
        return x_out, loss_commit, perplexity

    def encode(self, features: Tensor) -> Tuple[Tensor, None]:
        """Encode → discrete codes. Returns (B, 49, num_quantizers)."""
        B = features.shape[0]
        x_encoder = self.forward_encoder(features)
        all_indices = self.quantizer.quantize(x_encoder)
        code_idx = torch.stack(
            [idx.view(B, self.num_latent_tokens) for idx in all_indices], dim=-1
        )
        return code_idx, None

    def decode(self, code_idx: Tensor, target_len: Optional[int] = None) -> Tensor:
        """Decode discrete codes → motion."""
        if code_idx.dim() == 2:
            code_idx = code_idx.unsqueeze(0)
        B, T_prime, n_q = code_idx.shape
        if target_len is None:
            target_len = T_prime * self.patch_size

        x_quantized = None
        for i in range(n_q):
            indices_flat = code_idx[:, :, i].reshape(-1)
            quantizer = self.quantizer._get_quantizer(i)
            z_q = quantizer.dequantize(indices_flat)
            z_q = z_q.view(B, T_prime, self.code_dim).permute(0, 2, 1).contiguous()
            x_quantized = z_q if x_quantized is None else x_quantized + z_q

        x_out = self.forward_decoder(x_quantized, target_len)
        if x_out.shape[1] > target_len:
            x_out = x_out[:, :target_len, :]
        return x_out
