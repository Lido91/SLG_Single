"""
RVQ-VAE with Transformer Encoder/Decoder and Language-Guided Codebook Learning.

Replaces the Conv1D encoder/decoder from RVQVaeLGVQ with:
- Transformer encoder: part-aware patch embedding + learnable latent tokens +
  U-Net self-attention → fixed 49 latent tokens
- Transformer decoder: learnable mask tokens + cross-attention to z_hat +
  U-Net → part-aware unpatch

Keeps:
- RVQ quantizer (3-layer, 512 codes, EMA+Reset) from RVQVae
- LGVQ contrastive alignment (NCE, Mask, WRS) from RVQVaeLGVQ
- Text-free inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

from .mgpt_rvq_lgvq import RVQVaeLGVQ
from .tools.transformer_vae_blocks import (
    UNetTransformerEncoder,
    UNetTransformerDecoder,
)
from .tools.llamagen_blocks import precompute_freqs_cis_1d


class RVQVaeLGVQTransformer(RVQVaeLGVQ):
    """
    RVQ-VAE with Transformer encoder/decoder + LGVQ alignment.

    Architecture:
        Input [B, T, 133]
            ↓ part-aware patch embedding (body/lhand/rhand/face → sum)
        [B, T/patch, latent_dim]
            ↓ concat with learnable latent_tokens (49)
            ↓ UNet Transformer Encoder (self-attention)
            ↓ extract latent tokens → project
        [B, code_dim, 49]
            ↓ ResidualVQ (3 layers, unchanged)
        [B, code_dim, 49]
            ↓ project → cross-attended by mask tokens
            ↓ UNet Transformer Decoder
            ↓ part-aware unpatch
        Output [B, T, 133]
    """

    # Body part dimension ranges within the 133D SMPL-X feature vector
    BODY_DIMS = (0, 33)      # upper body: 33D
    LHAND_DIMS = (33, 78)    # left hand: 45D
    RHAND_DIMS = (78, 123)   # right hand: 45D
    FACE_DIMS = (123, 133)   # jaw + expression: 10D

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
        # Initialize parent: creates Conv1D encoder/decoder + RVQ + LGVQ
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size

        # Remove Conv1D encoder/decoder from parent (not needed)
        del self.encoder
        del self.decoder

        # --- Part-aware patch embedding (encoder input) ---
        body_d = self.BODY_DIMS[1] - self.BODY_DIMS[0]    # 33
        lhand_d = self.LHAND_DIMS[1] - self.LHAND_DIMS[0]  # 45
        rhand_d = self.RHAND_DIMS[1] - self.RHAND_DIMS[0]  # 45
        face_d = self.FACE_DIMS[1] - self.FACE_DIMS[0]    # 10

        self.body_patch_embed = nn.Conv1d(body_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.lhand_patch_embed = nn.Conv1d(lhand_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.rhand_patch_embed = nn.Conv1d(rhand_d, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.face_patch_embed = nn.Conv1d(face_d, latent_dim, kernel_size=patch_size, stride=patch_size)

        # --- Learnable latent tokens (fixed bottleneck) ---
        scale = latent_dim ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_latent_tokens, latent_dim))

        # --- Encoder: latent projection ---
        self.latent_proj = nn.Conv1d(latent_dim, self.code_dim, kernel_size=3, stride=1, padding=1)

        # --- Transformer encoder ---
        self.transformer_encoder = UNetTransformerEncoder(
            dim=latent_dim,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        # --- Decoder: z_hat projection ---
        self.z_hat_proj = nn.Conv1d(self.code_dim, latent_dim, kernel_size=3, stride=1, padding=1)

        # --- Learnable mask token (decoder query) ---
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, latent_dim))

        # --- Decoder position embedding (learnable, for mask tokens) ---
        max_output_tokens = 400 // patch_size  # 100 tokens for max 400 frames
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, max_output_tokens, latent_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        # --- Transformer decoder ---
        self.transformer_decoder = UNetTransformerDecoder(
            dim=latent_dim,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        # --- Part-aware unpatch (decoder output) ---
        self.body_unpatch = nn.ConvTranspose1d(latent_dim, body_d, kernel_size=patch_size, stride=patch_size)
        self.lhand_unpatch = nn.ConvTranspose1d(latent_dim, lhand_d, kernel_size=patch_size, stride=patch_size)
        self.rhand_unpatch = nn.ConvTranspose1d(latent_dim, rhand_d, kernel_size=patch_size, stride=patch_size)
        self.face_unpatch = nn.ConvTranspose1d(latent_dim, face_d, kernel_size=patch_size, stride=patch_size)

        # --- RoPE frequency cache ---
        self._freqs_cis_enc = None
        self._freqs_cis_dec = None

        # Re-initialize new parameters
        self.apply(self._init_transformer_weights)

    def _init_transformer_weights(self, module):
        """Initialize transformer-specific weights. Skip CLIP and RVQ modules."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def _get_encoder_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get RoPE frequencies for encoder (latent tokens get zero rotation)."""
        head_dim = self.latent_dim // self.transformer_encoder.input_blocks[0].attn.n_head
        if self._freqs_cis_enc is None or self._freqs_cis_enc.shape[0] < seq_len:
            # All tokens get sequential positions (no prefix distinction)
            self._freqs_cis_enc = precompute_freqs_cis_1d(
                seq_len=seq_len, n_elem=head_dim, base=100.0, cls_token_num=0
            )
        return self._freqs_cis_enc[:seq_len].to(device)

    def _get_decoder_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get RoPE frequencies for decoder mask tokens."""
        head_dim = self.latent_dim // self.transformer_decoder.input_blocks[0].self_attn.n_head
        if self._freqs_cis_dec is None or self._freqs_cis_dec.shape[0] < seq_len:
            self._freqs_cis_dec = precompute_freqs_cis_1d(
                seq_len=seq_len, n_elem=head_dim, base=100.0, cls_token_num=0
            )
        return self._freqs_cis_dec[:seq_len].to(device)

    def _patch_embed(self, x: Tensor) -> Tensor:
        """
        Part-aware patch embedding: separate projection per body part, then sum.

        Args:
            x: (B, D, T) — motion features in channel-first format

        Returns:
            patches: (B, T/patch, latent_dim)
        """
        body = self.body_patch_embed(x[:, self.BODY_DIMS[0]:self.BODY_DIMS[1], :])
        lhand = self.lhand_patch_embed(x[:, self.LHAND_DIMS[0]:self.LHAND_DIMS[1], :])
        rhand = self.rhand_patch_embed(x[:, self.RHAND_DIMS[0]:self.RHAND_DIMS[1], :])
        face = self.face_patch_embed(x[:, self.FACE_DIMS[0]:self.FACE_DIMS[1], :])
        return (body + lhand + rhand + face).transpose(1, 2)  # (B, T/patch, latent_dim)

    def _unpatch(self, x: Tensor) -> Tensor:
        """
        Part-aware unpatch: separate projection per body part, then concat.

        Args:
            x: (B, latent_dim, T/patch)

        Returns:
            motion: (B, D, T) — reconstructed motion in channel-first format
        """
        body = self.body_unpatch(x)
        lhand = self.lhand_unpatch(x)
        rhand = self.rhand_unpatch(x)
        face = self.face_unpatch(x)
        return torch.cat([body, lhand, rhand, face], dim=1)  # (B, 133, T)

    def forward_encoder(self, features: Tensor) -> Tensor:
        """
        Encode motion to fixed-length latent representation.

        Args:
            features: (B, T, D) — input motion

        Returns:
            z: (B, code_dim, num_latent_tokens) — latent for RVQ
        """
        B, T, D = features.shape
        x = features.transpose(1, 2)  # (B, D, T)

        # Part-aware patch embedding
        patches = self._patch_embed(x)  # (B, T/patch, latent_dim)
        num_patches = patches.shape[1]

        # Expand latent tokens per batch
        latent = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 49, latent_dim)

        # Concatenate: [latent_tokens | motion_patches]
        enc_input = torch.cat([latent, patches], dim=1)  # (B, 49+T/patch, latent_dim)

        # RoPE frequencies for full sequence
        freqs = self._get_encoder_freqs(enc_input.shape[1], enc_input.device)

        # Transformer encoder (bidirectional self-attention)
        enc_output = self.transformer_encoder(enc_input, freqs)

        # Extract latent tokens only
        z = enc_output[:, :self.num_latent_tokens, :]  # (B, 49, latent_dim)

        # Project to code_dim
        z = self.latent_proj(z.transpose(1, 2))  # (B, code_dim, 49)
        return z

    def forward_decoder(self, z_hat: Tensor, target_len: int) -> Tensor:
        """
        Decode quantized latent codes to motion.

        Args:
            z_hat: (B, code_dim, num_latent_tokens) — quantized codes
            target_len: target temporal length T (before patching)

        Returns:
            motion: (B, T, D) — reconstructed motion
        """
        B = z_hat.shape[0]
        num_output_tokens = target_len // self.patch_size

        # Project z_hat to latent_dim
        memory = self.z_hat_proj(z_hat).transpose(1, 2)  # (B, 49, latent_dim)

        # Create mask tokens for output
        mask_tokens = self.mask_token.expand(B, num_output_tokens, -1)  # (B, T/patch, latent_dim)
        mask_tokens = mask_tokens + self.dec_pos_embed[:, :num_output_tokens, :]

        # RoPE frequencies for decoder
        freqs = self._get_decoder_freqs(num_output_tokens, z_hat.device)

        # Transformer decoder (self-attn + cross-attn to z_hat)
        dec_output = self.transformer_decoder(mask_tokens, memory, freqs)  # (B, T/patch, latent_dim)

        # Part-aware unpatch
        motion = self._unpatch(dec_output.transpose(1, 2))  # (B, D, T)
        return motion.transpose(1, 2)  # (B, T, D)

    def forward(
        self,
        features: Tensor,
        texts: Optional[List[str]] = None,
        clip_text_features: Optional[Tensor] = None,
    ):
        """
        Forward pass: encode → RVQ → decode + LGVQ alignment.

        Same interface as RVQVaeLGVQ.forward().

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss_commit: Commitment loss from RVQ
            perplexity: Codebook usage metric
            motion_emb: [B, 512] L2-normed motion embedding (or None)
            text_emb: [B, 512] L2-normed text embedding (or None)
        """
        B, T, D = features.shape

        # Encode
        x_encoder = self.forward_encoder(features)  # (B, code_dim, 49)

        # RVQ quantization
        x_quantized, all_indices, loss_commit, perplexity = self.quantizer(x_encoder)

        # Decode
        x_out = self.forward_decoder(x_quantized, T)  # (B, T, D)

        # Zero out padding if T is not divisible by patch_size
        # (ConvTranspose1d may produce slightly longer output)
        if x_out.shape[1] > T:
            x_out = x_out[:, :T, :]

        # LGVQ alignment (same as parent)
        device = features.device
        motion_emb = None
        text_emb = None

        if not self.training or (self.nce_weight == 0 and self.mask_weight == 0 and self.wrs_weight == 0):
            return x_out, loss_commit, perplexity, motion_emb, text_emb

        # Get text embedding: precomputed or CLIP forward
        if clip_text_features is not None:
            text_emb = F.normalize(clip_text_features.to(device).float(), dim=-1)
        elif texts is not None and len(texts) > 1:
            _, last_text_feature, _, _, _ = self._encode_text_clip(texts, device)
            text_emb = F.normalize(last_text_feature.detach(), dim=-1)
        else:
            return x_out, loss_commit, perplexity, motion_emb, text_emb

        # Get motion embedding from encoder output
        global_feat, _ = self.motion_projector(x_encoder)
        motion_emb = F.normalize(global_feat, dim=-1)

        return x_out, loss_commit, perplexity, motion_emb, text_emb

    def encode(self, features: Tensor) -> Tuple[Tensor, None]:
        """
        Encode motion to discrete codes.

        Args:
            features: [B, T, D]

        Returns:
            code_idx: [B, 49, num_quantizers]
            dist: None
        """
        B = features.shape[0]

        x_encoder = self.forward_encoder(features)  # (B, code_dim, 49)
        all_indices = self.quantizer.quantize(x_encoder)  # List of [B*49] tensors

        code_idx_list = []
        for indices in all_indices:
            indices_reshaped = indices.view(B, self.num_latent_tokens)
            code_idx_list.append(indices_reshaped)

        code_idx = torch.stack(code_idx_list, dim=-1)  # (B, 49, num_quantizers)
        return code_idx, None

    def decode(self, code_idx: Tensor, target_len: Optional[int] = None) -> Tensor:
        """
        Decode discrete codes to motion.

        Args:
            code_idx: [B, 49, num_quantizers] or [49, num_quantizers]
            target_len: target motion length T. If None, uses 49 * patch_size.

        Returns:
            x_out: [B, T, D]
        """
        if code_idx.dim() == 2:
            code_idx = code_idx.unsqueeze(0)

        B, T_prime, n_quantizers = code_idx.shape

        if target_len is None:
            target_len = T_prime * self.patch_size

        # Dequantize: sum across quantizer layers
        x_quantized = None
        for i in range(n_quantizers):
            indices = code_idx[:, :, i]  # (B, 49)
            indices_flat = indices.reshape(-1)
            quantizer = self.quantizer._get_quantizer(i)
            z_q = quantizer.dequantize(indices_flat)  # (B*49, code_dim)
            z_q = z_q.view(B, T_prime, self.code_dim).permute(0, 2, 1).contiguous()
            if x_quantized is None:
                x_quantized = z_q
            else:
                x_quantized = x_quantized + z_q

        x_out = self.forward_decoder(x_quantized, target_len)

        if x_out.shape[1] > target_len:
            x_out = x_out[:, :target_len, :]

        return x_out

    def encode_continuous(self, features: Tensor) -> Tensor:
        """
        Extract continuous embeddings before quantization (for contrastive learning).

        Args:
            features: [B, T, D]

        Returns:
            embeddings: [B, code_dim] pooled continuous embeddings
        """
        x_encoder = self.forward_encoder(features)  # (B, code_dim, 49)
        return x_encoder.mean(dim=-1)  # (B, code_dim)
