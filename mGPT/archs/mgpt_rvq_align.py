"""
RVQ-VAE with Multi-Scale Contrastive Alignment (MotionBind-inspired)

Aligns motion and text in a shared embedding space using:
1. Multi-scale encoder features: extract features from each Conv1D downsampling
   level, fuse via learnable softmax-weighted average (like MuTMoT)
2. Improved InfoNCE loss with:
   - Length-aware weighting: penalize negatives with large duration differences
   - Semantic margin: penalize negatives that are semantically dissimilar in
     the frozen text encoder space

Reference: MotionBind (Kinfu & Vidal, NeurIPS 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

from .mgpt_rvq import RVQVae

import clip


class RVQVaeAlign(RVQVae):
    """
    RVQ-VAE with multi-scale contrastive text alignment.

    Extracts features from multiple encoder levels, fuses them with
    learnable weights, and aligns with text via improved InfoNCE.

    Additional Args (beyond RVQVae):
        lambda_align: Weight for alignment loss. Default: 0.01
        tau_align: InfoNCE temperature. Default: 0.07
        align_proj_dim: Projection dimension for ITC. Default: 256
        clip_model: CLIP model variant. Default: 'ViT-B/32'
        length_penalty_lambda: Strength of length-aware weighting. Default: 1.0
        length_penalty_delta: Threshold for length difference. Default: 0.2
        semantic_margin_rho: Similarity threshold for margin. Default: 0.5
    """

    def __init__(
        self,
        lambda_align: float = 0.01,
        tau_align: float = 0.07,
        align_proj_dim: int = 256,
        clip_model: str = 'ViT-B/32',
        length_penalty_lambda: float = 1.0,
        length_penalty_delta: float = 0.2,
        semantic_margin_rho: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.lambda_align = lambda_align
        self.tau_align = tau_align
        self.align_proj_dim = align_proj_dim
        self.length_penalty_lambda = length_penalty_lambda
        self.length_penalty_delta = length_penalty_delta
        self.semantic_margin_rho = semantic_margin_rho

        # Frozen CLIP text encoder
        self.clip_model, _ = clip.load(clip_model, device='cpu', jit=False)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        self.clip_dim = 512  # CLIP ViT-B/32 output dim

        # Text projection head (adapter f_LB in MotionBind)
        self.proj_text = nn.Sequential(
            nn.Linear(self.clip_dim, align_proj_dim),
            nn.ReLU(),
            nn.Linear(align_proj_dim, align_proj_dim),
        )

        # Number of encoder scales to extract
        # Encoder structure: [Conv+ReLU, down_block_0, ..., down_block_{n-1}, Conv_final]
        # We extract after each down_block + after final conv
        # With down_t=2: 3 scales (after down_0, after down_1, after final_conv)
        n_scales = kwargs.get('down_t', 2) + 1
        self.n_scales = n_scales

        # Learnable scale fusion weights (softmax-weighted average, Eq.3 in MotionBind)
        self.scale_weights = nn.Parameter(torch.zeros(n_scales))

        # Single motion projection head (applied after fusion)
        self.proj_motion = nn.Sequential(
            nn.Linear(self.code_dim, align_proj_dim),
            nn.ReLU(),
            nn.Linear(align_proj_dim, align_proj_dim),
        )

    def train(self, mode: bool = True):
        """Override to keep CLIP frozen in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def _encode_text(self, texts: List[str], device: torch.device) -> Tensor:
        """Encode texts with frozen CLIP. Returns [B, 512]."""
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(tokens).float()  # [B, 512]
        return text_emb

    def _encode_multiscale(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Run encoder once, return final output AND multi-scale pooled features.

        Encoder structure (nn.Sequential):
            [0] Conv1d(nfeats, width, 3, 1, 1)
            [1] ReLU
            [2] Sequential(Conv1d_down + Resnet1D)  # down block 0
            [3] Sequential(Conv1d_down + Resnet1D)  # down block 1
            ...
            [-1] Conv1d(width, output_emb_width, 3, 1, 1)  # final projection

        Args:
            x: [B, D, T] preprocessed input

        Returns:
            x_encoder: [B, code_dim, T'] final encoder output
            scale_features: List of [B, code_dim] pooled features per scale
        """
        layers = list(self.encoder.model.children())
        scale_features = []

        h = x
        for i, layer in enumerate(layers):
            h = layer(h)
            # Skip initial Conv+ReLU (indices 0, 1)
            # Collect after each downsampling block and final conv
            if i >= 2:
                pooled = h.mean(dim=-1)  # [B, C] pool over time
                scale_features.append(pooled)

        x_encoder = h  # final encoder output
        return x_encoder, scale_features

    def _fuse_multiscale(self, features: List[Tensor]) -> Tensor:
        """
        Fuse multi-scale features via learnable softmax-weighted average.
        (Eq. 3 in MotionBind: z^motion = sum(alpha_b * E_0^(b)))

        Args:
            features: List of [B, code_dim] per-scale features

        Returns:
            fused: [B, code_dim]
        """
        alpha = F.softmax(self.scale_weights[:len(features)], dim=0)
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused = fused + alpha[i] * feat
        return fused

    def _itc_loss(
        self,
        motion_emb: Tensor,
        text_emb: Tensor,
        text_raw_emb: Tensor,
        lengths: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Improved InfoNCE loss (MotionBind Eq. 5-7).

        For negative pair (i, j):
            effective_logit = w_ij * exp((s_ij - m_ij) / tau)

        Implemented as log-space adjustment:
            adjusted_logit_ij = s_ij/tau - m_ij/tau + log(w_ij)

        Args:
            motion_emb: [B, proj_dim] L2-normalized motion embeddings
            text_emb: [B, proj_dim] L2-normalized text embeddings (projected)
            text_raw_emb: [B, clip_dim] L2-normalized raw CLIP text embeddings
            lengths: Optional list of motion lengths for length-aware weighting

        Returns:
            loss: Scalar ITC loss
        """
        B = motion_emb.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=motion_emb.device)

        # Raw similarity / tau  [B, B]
        logits = motion_emb @ text_emb.t() / self.tau_align

        # Length-aware weighting (Eq. 6): log(w_ij) added to negative logits
        if lengths is not None:
            len_t = torch.tensor(lengths, dtype=torch.float32, device=motion_emb.device)
            max_len = len_t.max().clamp(min=1.0)
            delta = torch.abs(len_t.unsqueeze(1) - len_t.unsqueeze(0)) / max_len  # [B, B]
            log_w = torch.where(
                delta > self.length_penalty_delta,
                torch.log(1.0 + self.length_penalty_lambda * delta ** 2),
                torch.zeros_like(delta),
            )  # [B, B]
        else:
            log_w = torch.zeros(B, B, device=motion_emb.device)

        # Semantic margin (Eq. 7): subtract m_ij/tau from negative logits
        with torch.no_grad():
            text_sim = text_raw_emb @ text_raw_emb.t()  # [B, B] cosine sim
            margin = torch.where(
                text_sim < self.semantic_margin_rho,
                (1.0 - text_sim) / self.tau_align,
                torch.zeros_like(text_sim),
            )  # [B, B]

        # Only apply adjustments to negative pairs
        mask_pos = torch.eye(B, device=logits.device, dtype=torch.bool)
        neg_adjustment = log_w - margin  # positive = harder negative
        neg_adjustment[mask_pos] = 0.0

        # Labels: diagonal = positive pairs
        labels = torch.arange(B, device=logits.device)

        # Motion-to-text
        logits_m2t = logits + neg_adjustment
        loss_m2t = F.cross_entropy(logits_m2t, labels)

        # Text-to-motion
        logits_t2m = logits.t() + neg_adjustment.t()
        loss_t2m = F.cross_entropy(logits_t2m, labels)

        return (loss_m2t + loss_t2m) / 2

    def forward(
        self,
        features: Tensor,
        texts: Optional[List[str]] = None,
        lengths: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through Align RVQ-VAE.

        Args:
            features: Input motion features [B, T, D]
            texts: Optional list of text strings for alignment
            lengths: Optional list of motion frame lengths

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss_commit: Commitment loss
            perplexity: Codebook usage metric
            loss_align: Contrastive alignment loss (weighted by lambda_align)
        """
        # Preprocess: [B, T, D] -> [B, D, T]
        x_in = self.preprocess(features)

        # Encode once: get final output + multi-scale features
        need_align = texts is not None and self.training and len(texts) > 1
        if need_align:
            x_encoder, multiscale_feats = self._encode_multiscale(x_in)
        else:
            x_encoder = self.encoder(x_in)

        # Residual quantization
        x_quantized, all_indices, loss_commit, perplexity = \
            self.quantizer(x_encoder)

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        # Compute alignment loss
        device = features.device
        if need_align:
            # Encode text with frozen CLIP
            text_emb = self._encode_text(texts, device)  # [B, 512]
            text_raw_norm = F.normalize(text_emb, p=2, dim=-1)  # for semantic margin

            # Project text
            text_proj = self.proj_text(text_emb)  # [B, proj_dim]
            text_proj = F.normalize(text_proj, p=2, dim=-1)

            # Fuse multi-scale motion features
            motion_fused = self._fuse_multiscale(multiscale_feats)  # [B, code_dim]

            # Project motion
            motion_proj = self.proj_motion(motion_fused)  # [B, proj_dim]
            motion_proj = F.normalize(motion_proj, p=2, dim=-1)

            # Improved InfoNCE with length weighting + semantic margin
            loss_align = self.lambda_align * self._itc_loss(
                motion_proj, text_proj, text_raw_norm, lengths
            )
        else:
            loss_align = torch.tensor(0.0, device=device)

        return x_out, loss_commit, perplexity, loss_align
