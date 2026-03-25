"""
RVQ-VAE with Speech-Guided Codebook Learning (Speech LG-VQ).

Replaces CLIP text guidance from the original LG-VQ with a frozen speech encoder
(e.g. Whisper) for InfoNCE contrastive alignment between motion and speech.

Only the NCE loss is used (mask prediction and WRS are CLIP-specific and omitted).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

from .mgpt_rvq import RVQVae
from .speech_encoder import SpeechEncoder, SPEECH_ENCODER_CONFIGS


class MotionProjector(nn.Module):
    """
    Lightweight projector: mean pooling + linear for global NCE feature.
    Projects encoder output (code_dim) to speech embedding space (speech_dim).
    """

    def __init__(self, code_dim: int, speech_dim: int):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(code_dim, speech_dim),
            nn.GELU(),
            nn.LayerNorm(speech_dim),
        )
        nn.init.normal_(self.global_proj[0].weight, std=0.01)
        nn.init.zeros_(self.global_proj[0].bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, code_dim, T'] encoder features

        Returns:
            global_feat: [B, speech_dim] mean-pooled + projected
        """
        x = x.permute(0, 2, 1)                        # [B, T', code_dim]
        global_feat = self.global_proj(x.mean(dim=1))  # [B, speech_dim]
        return global_feat


class RVQVaeLGVQSpeech(RVQVae):
    """
    RVQ-VAE with Speech-Guided Codebook Learning.

    Uses a frozen speech encoder (Whisper/HuBERT/etc.) instead of CLIP for
    InfoNCE contrastive alignment between motion and speech embeddings.
    """

    def __init__(
        self,
        speech_encoder_type: str = 'whisper-base',
        freeze_speech_encoder: bool = True,
        nce_weight: float = 0.001,
        use_precomputed_speech: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.nce_weight = nce_weight
        self.speech_encoder_type = speech_encoder_type
        self.use_precomputed_speech = use_precomputed_speech
        self.speech_dim = SPEECH_ENCODER_CONFIGS[speech_encoder_type]['dim']

        # Only load speech encoder if not using precomputed features
        if not use_precomputed_speech:
            self.speech_encoder = SpeechEncoder(speech_encoder_type, freeze=freeze_speech_encoder)

        # Project motion encoder output to speech embedding space
        self.motion_projector = MotionProjector(self.code_dim, self.speech_dim)

    def train(self, mode: bool = True):
        """Keep speech encoder frozen in eval mode."""
        super().train(mode)
        if hasattr(self, 'speech_encoder') and self.speech_encoder.freeze:
            self.speech_encoder.eval()
        return self

    def _encode_speech(self, audio_waveforms: Tensor) -> Tensor:
        """
        Encode audio with speech encoder and mean-pool to global embedding.

        Args:
            audio_waveforms: [B, num_samples] raw audio at 16kHz

        Returns:
            speech_emb: [B, speech_dim] mean-pooled speech embedding
        """
        speech_feats, attn_mask = self.speech_encoder(audio_waveforms, return_attention_mask=True)
        # speech_feats: [B, seq_len, speech_dim]
        # attn_mask: [B, seq_len]

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1)  # [B, seq_len, 1]
            speech_emb = (speech_feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            speech_emb = speech_feats.mean(dim=1)

        return speech_emb  # [B, speech_dim]

    def forward(
        self,
        features: Tensor,
        audio_waveforms: Optional[Tensor] = None,
        audio_lengths: Optional[list] = None,
        speech_feats: Optional[Tensor] = None,
        speech_mask: Optional[Tensor] = None,
    ):
        """
        Forward pass through Speech LG-VQ RVQ-VAE.

        Args:
            features: Input motion features [B, T, D]
            audio_waveforms: Optional raw audio [B, num_samples] at 16kHz
            audio_lengths: Optional original audio lengths before padding
            speech_feats: Optional precomputed speech features [B, T_s, speech_dim]
            speech_mask: Optional mask for precomputed features [B, T_s]

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss_commit: Commitment loss from RVQ
            perplexity: Codebook usage metric
            motion_emb: [B, speech_dim] L2-normed motion embedding (or None)
            speech_emb: [B, speech_dim] L2-normed speech embedding (or None)
        """
        # Standard RVQ forward
        x_in = self.preprocess(features)          # [B, D, T]
        x_encoder = self.encoder(x_in)            # [B, code_dim, T']
        x_quantized, all_indices, loss_commit, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)     # [B, D, T]
        x_out = self.postprocess(x_decoder)       # [B, T, D]

        motion_emb = None
        speech_emb = None

        if not self.training or self.nce_weight == 0:
            return x_out, loss_commit, perplexity, motion_emb, speech_emb

        # Speech embedding: prefer precomputed features, fallback to raw audio
        if speech_feats is not None:
            # Precomputed features: mean-pool with mask
            if speech_mask is not None:
                mask = speech_mask.unsqueeze(-1)  # [B, T_s, 1]
                speech_emb = (speech_feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                speech_emb = speech_feats.mean(dim=1)
            speech_emb = F.normalize(speech_emb.detach(), dim=-1)
        elif audio_waveforms is not None:
            speech_emb = self._encode_speech(audio_waveforms)
            speech_emb = F.normalize(speech_emb.detach(), dim=-1)
        else:
            return x_out, loss_commit, perplexity, None, None

        # Motion embedding: project encoder output + mean pool
        motion_emb = self.motion_projector(x_encoder)
        motion_emb = F.normalize(motion_emb, dim=-1)

        return x_out, loss_commit, perplexity, motion_emb, speech_emb
