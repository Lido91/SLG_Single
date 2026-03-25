"""
Fine-Grained Multi-Modal Motion Retrieval Model (paper 2507.23188).

Adapted for Speech-Motion-Text (3 modalities, no Video).

Key components:
1. VaeMotionEncoder: frozen RVQVae encoder + trainable transformer (sequence-level tokens)
2. MemoryRetrievalAudioEncoder: WavLM + memory-retrieval cross-attention compression
3. SequenceTextEncoder: DistilBERT/CLIP + transformer (keeps sequence-level tokens)
4. Sequence-level contrastive loss with token-level max similarity + KL divergence
5. ReconstructionDecoder: auxiliary masked motion reconstruction
"""

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mGPT.archs.speech_encoder import SpeechEncoder, SPEECH_ENCODER_CONFIGS
from mGPT.archs.mgpt_rvq import RVQVae
from mGPT.archs.pos_encoding import PositionEmbedding
from contrastive.loss_finegrained import compute_alignment_loss, sequence_level_similarity


# ============================================================
# VAE-based Motion Encoder (sequence-level tokens, no body-part)
# ============================================================

class VaeMotionEncoder(nn.Module):
    """
    Motion encoder using frozen RVQVae encoder + trainable transformer.

    Unlike the old model which pools into a single global vector,
    this keeps the full sequence of tokens for sequence-level contrastive loss.

    Pipeline (paper Section IV: "avg pooling inserted between every two layers"):
        Input [B, T, 133]
        → vae.preprocess → [B, 133, T]
        → vae.encoder (frozen) → [B, 512, T']  (T' = T / temporal_compression)
        → permute → [B, T', 512]
        → Linear(512, C) + PosEnc
        → TransformerEncoder (2 layers)
        → AvgPool1d(kernel=2, stride=2) → [B, T'//2, C]
        → TransformerEncoder (2 layers)
        → L2 normalize per token → [B, T'//2, C]
    """

    def __init__(
        self,
        vae_config: dict,
        vae_checkpoint: str = None,
        latent_dim: int = 512,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        preload_features: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.preload_features = preload_features

        # Frozen VAE encoder (skipped when using precomputed features)
        if not preload_features:
            self.vae = self._load_frozen_vae(vae_config, vae_checkpoint)
        vae_dim = vae_config['params']['code_dim']  # typically 512

        stride_t = vae_config['params'].get('stride_t', 2)
        down_t = vae_config['params'].get('down_t', 2)
        self._temporal_compression = stride_t ** down_t  # typically 4

        # Trainable projection + transformer
        self.input_proj = nn.Linear(vae_dim, latent_dim)
        self.pos_enc = PositionEmbedding(max_seq_len, latent_dim, dropout)

        # Split into two transformer blocks with avg pooling in between
        # (paper: "four-layer transformer encoder, with an average pooling layer
        #  inserted between every two layers to reduce the temporal size")
        n_first = num_layers // 2
        n_second = num_layers - n_first
        layer_fn = lambda: nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=latent_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer_1 = nn.TransformerEncoder(layer_fn(), num_layers=n_first)
        self.transformer_2 = nn.TransformerEncoder(layer_fn(), num_layers=n_second)

    @staticmethod
    def _load_frozen_vae(vae_config, vae_checkpoint):
        params = vae_config['params']
        vae = RVQVae(**params)
        if vae_checkpoint is not None and os.path.exists(vae_checkpoint):
            ckpt = torch.load(vae_checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                vae_state = {}
                for k, v in state_dict.items():
                    if k.startswith('vae.'):
                        vae_state[k[len('vae.'):]] = v
                if vae_state:
                    state_dict = vae_state
            else:
                state_dict = ckpt
            vae.load_state_dict(state_dict, strict=False)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    def forward(self, motion: torch.Tensor, lengths: list) -> tuple:
        """
        Args:
            motion: [B, T, 133] normalized motion features
            lengths: list of original frame counts

        Returns:
            tokens: [B, T_out, C] L2-normalized motion token sequence
            mask: [B, T_out] validity mask (T_out ≈ T / temporal_compression / 2)
        """
        B = motion.shape[0]

        # Frozen VAE encoder
        with torch.no_grad():
            x = self.vae.preprocess(motion)  # [B, 133, T]
            x = self.vae.encoder(x)           # [B, 512, T']
            x = x.permute(0, 2, 1)            # [B, T', 512]

        T_prime = x.shape[1]

        # Trainable projection + positional encoding
        x = self.input_proj(x)  # [B, T', C]
        x = self.pos_enc(x)

        # First 2 transformer layers
        mask_pre = self._build_mask(lengths, T_prime, motion.device, self._temporal_compression)
        x = self.transformer_1(x, src_key_padding_mask=~mask_pre.bool())

        # Average pooling: halve temporal dimension (paper Section IV)
        x = x.permute(0, 2, 1)                        # [B, C, T']
        x = F.avg_pool1d(x, kernel_size=2, stride=2)  # [B, C, T'//2]
        x = x.permute(0, 2, 1)                        # [B, T'//2, C]

        T_pooled = x.shape[1]

        # Recompute mask for pooled length (total compression = vae * 2)
        mask = self._build_mask(lengths, T_pooled, motion.device, self._temporal_compression * 2)

        # Last 2 transformer layers
        x = self.transformer_2(x, src_key_padding_mask=~mask.bool())

        tokens = F.normalize(x, dim=-1)
        return tokens, mask

    def _build_mask(self, lengths, T_out, device, total_compression):
        B = len(lengths)
        mask = torch.zeros(B, T_out, device=device)
        for i, l in enumerate(lengths):
            if isinstance(l, torch.Tensor):
                l = l.item()
            valid = min(math.ceil(l / total_compression), T_out)
            mask[i, :valid] = 1.0
        return mask

    def forward_from_feats(self, vae_feats: torch.Tensor, motion_lengths: list) -> tuple:
        """
        Forward pass from pre-extracted VAE encoder features (skips frozen VAE).

        Args:
            vae_feats: [B, T', vae_dim] pre-extracted VAE encoder output
            motion_lengths: list of valid lengths in T' space (already compressed)

        Returns:
            tokens: [B, T_out, C] L2-normalized motion token sequence
            mask: [B, T_out] validity mask
        """
        B, T_prime, _ = vae_feats.shape

        x = self.input_proj(vae_feats)
        x = self.pos_enc(x)

        # Pre-pool mask (lengths already in T' space, no compression needed)
        mask_pre = self._build_mask_from_valid(motion_lengths, T_prime, vae_feats.device)
        x = self.transformer_1(x, src_key_padding_mask=~mask_pre.bool())

        # Average pooling: halve temporal dimension
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.permute(0, 2, 1)

        T_pooled = x.shape[1]

        # Post-pool mask (pool halves the valid length)
        mask = torch.zeros(B, T_pooled, device=vae_feats.device)
        for i, ml in enumerate(motion_lengths):
            if isinstance(ml, torch.Tensor):
                ml = ml.item()
            valid_pooled = min(math.ceil(ml / 2), T_pooled)
            mask[i, :valid_pooled] = 1.0

        x = self.transformer_2(x, src_key_padding_mask=~mask.bool())
        tokens = F.normalize(x, dim=-1)
        return tokens, mask

    @staticmethod
    def _build_mask_from_valid(valid_lengths, T_out, device):
        """Build mask from pre-computed valid lengths (no compression applied)."""
        B = len(valid_lengths)
        mask = torch.zeros(B, T_out, device=device)
        for i, vl in enumerate(valid_lengths):
            if isinstance(vl, torch.Tensor):
                vl = vl.item()
            mask[i, :min(vl, T_out)] = 1.0
        return mask


# ============================================================
# Memory-Retrieval Audio Encoder
# ============================================================

class MemoryRetrievalAudioEncoder(nn.Module):
    """
    Audio encoder with memory-retrieval compression (Section III-A.4 / Fig.5 of paper).

    Pipeline:
        WavLM(frozen) → [B, T_s, dim_speech]
        → Linear → [B, T_s, C]
        → Cross-Attention(Q=features, K=memory, V=memory) → [B, T_s, C]
        → Positional Encoding
        → AvgPool → [B, L_a, C]
        → L2 normalize

    The learnable memory tokens compress variable-length audio into semantically
    meaningful representations.
    """

    def __init__(
        self,
        speech_encoder_type: str = 'wavlm-large',
        latent_dim: int = 512,
        num_memory_tokens: int = 128,
        num_attn_layers: int = 2,
        target_length: int = 32,
        nhead: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1500,
        preload_features: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_length = target_length

        # Frozen speech encoder (skipped when using precomputed features)
        if not preload_features:
            self.speech_encoder = SpeechEncoder(speech_encoder_type, freeze=True)
        speech_dim = SPEECH_ENCODER_CONFIGS[speech_encoder_type]['dim']

        # Input projection
        self.input_proj = nn.Linear(speech_dim, latent_dim)

        # Learnable memory tokens [num_memory, C]
        self.memory_tokens = nn.Parameter(
            torch.randn(num_memory_tokens, latent_dim) * 0.02
        )

        # Cross-attention layers: Q=audio features, K=V=memory
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(latent_dim, nhead, dropout=dropout, batch_first=True)
            )
            self.cross_attn_norms.append(nn.LayerNorm(latent_dim))

        # Positional encoding after attention
        self.pos_enc = PositionEmbedding(max_seq_len, latent_dim, dropout)

    def forward(self, audio_waveforms: torch.Tensor, audio_lengths: list = None) -> tuple:
        """
        Args:
            audio_waveforms: [B, num_samples] raw audio at 16kHz
            audio_lengths: list of original sample counts before padding

        Returns:
            tokens: [B, L_a, C] L2-normalized audio token sequence
            mask: [B, L_a] validity mask
        """
        # 1. Extract features from frozen encoder
        with torch.no_grad():
            feats, _ = self.speech_encoder(audio_waveforms)  # [B, T_s, speech_dim]
        B, T_s, _ = feats.shape

        # 2. Project to latent dim
        x = self.input_proj(feats)  # [B, T_s, C]

        # 3. Create audio mask from lengths
        if audio_lengths is not None:
            audio_mask = torch.zeros(B, T_s, device=x.device)
            for i, al in enumerate(audio_lengths):
                if isinstance(al, torch.Tensor):
                    al = al.item()
                valid_frames = min(al // 320, T_s)
                audio_mask[i, :valid_frames] = 1.0
        else:
            audio_mask = torch.ones(B, T_s, device=x.device)

        # 4. Cross-attention with memory tokens
        # Q = audio features [B, T_s, C], K = V = memory [N_mem, C]
        memory = self.memory_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, N_mem, C]

        for attn, norm in zip(self.cross_attn_layers, self.cross_attn_norms):
            # Query = audio features, Key/Value = memory
            attn_out, _ = attn(x, memory, memory)  # [B, T_s, C]
            x = norm(x + attn_out)

        # 5. Add positional encoding
        x = self.pos_enc(x)

        # 6. Zero out padding before pooling to prevent padding leaking into valid tokens
        x = x * audio_mask.unsqueeze(-1)  # [B, T_s, C] — padding positions → 0

        # 7. Adaptive average pool to target_length
        # [B, T_s, C] → [B, C, T_s] → pool → [B, C, L_a] → [B, L_a, C]
        x_pooled = x.permute(0, 2, 1)  # [B, C, T_s]
        x_pooled = F.adaptive_avg_pool1d(x_pooled, self.target_length)  # [B, C, L_a]
        x_pooled = x_pooled.permute(0, 2, 1)  # [B, L_a, C]

        # 8. L2 normalize per token
        tokens = F.normalize(x_pooled, dim=-1)

        # 9. Mask: after adaptive pooling, compute valid portion
        mask = torch.ones(B, self.target_length, device=x.device)
        if audio_lengths is not None:
            for i, al in enumerate(audio_lengths):
                if isinstance(al, torch.Tensor):
                    al = al.item()
                valid_frames = min(al // 320, T_s)
                valid_ratio = valid_frames / max(T_s, 1)
                valid_pooled = max(1, int(valid_ratio * self.target_length))
                mask[i, valid_pooled:] = 0.0

        return tokens, mask

    def forward_from_feats(self, speech_feats: torch.Tensor, speech_lengths: list) -> tuple:
        """
        Forward pass from pre-extracted speech features (skips frozen encoder).

        Args:
            speech_feats: [B, T_s, speech_dim] pre-extracted features
            speech_lengths: list of valid frame counts in T_s space

        Returns:
            tokens: [B, L_a, C] L2-normalized audio token sequence
            mask: [B, L_a] validity mask
        """
        B, T_s, _ = speech_feats.shape

        x = self.input_proj(speech_feats)  # [B, T_s, C]

        # Build audio mask from precomputed lengths
        audio_mask = torch.zeros(B, T_s, device=x.device)
        for i, sl in enumerate(speech_lengths):
            if isinstance(sl, torch.Tensor):
                sl = sl.item()
            audio_mask[i, :min(sl, T_s)] = 1.0

        # Cross-attention with memory tokens
        memory = self.memory_tokens.unsqueeze(0).expand(B, -1, -1)
        for attn, norm in zip(self.cross_attn_layers, self.cross_attn_norms):
            attn_out, _ = attn(x, memory, memory)
            x = norm(x + attn_out)

        x = self.pos_enc(x)
        x = x * audio_mask.unsqueeze(-1)

        # Adaptive average pool to target_length
        x_pooled = x.permute(0, 2, 1)
        x_pooled = F.adaptive_avg_pool1d(x_pooled, self.target_length)
        x_pooled = x_pooled.permute(0, 2, 1)

        tokens = F.normalize(x_pooled, dim=-1)

        # Mask after pooling
        mask = torch.ones(B, self.target_length, device=x.device)
        for i, sl in enumerate(speech_lengths):
            if isinstance(sl, torch.Tensor):
                sl = sl.item()
            valid_ratio = min(sl, T_s) / max(T_s, 1)
            valid_pooled = max(1, int(valid_ratio * self.target_length))
            mask[i, valid_pooled:] = 0.0

        return tokens, mask


# ============================================================
# Sequence-Level Text Encoder
# ============================================================

class SequenceTextEncoder(nn.Module):
    """
    Text encoder that preserves sequence-level tokens (Section III-A.2 of paper).

    Supports two modes:
    - 'distilbert': DistilBERT + 2-layer Transformer (paper's approach)
    - 'clip': CLIP text encoder + projection (for compatibility)

    Output is a sequence of L2-normalized token embeddings.
    """

    def __init__(
        self,
        encoder_type: str = 'distilbert',
        latent_dim: int = 512,
        transformer_layers: int = 2,
        nhead: int = 8,
        max_text_len: int = 77,
        dropout: float = 0.1,
        preload_features: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.max_text_len = max_text_len

        if encoder_type == 'distilbert':
            self._init_distilbert(latent_dim, transformer_layers, nhead, dropout, preload_features)
        elif encoder_type == 'clip':
            self._init_clip(latent_dim, transformer_layers, nhead, dropout, preload_features)
        else:
            raise ValueError(f"Unknown text encoder type: {encoder_type}")

    def _init_distilbert(self, latent_dim, transformer_layers, nhead, dropout, preload_features=False):
        bert_dim = 768  # DistilBERT base output dim

        if not preload_features:
            from transformers import DistilBertModel, DistilBertTokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            # Freeze BERT
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()

        self.input_proj = nn.Linear(bert_dim, latent_dim)

        # Trainable transformer layers on top
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=latent_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.pos_enc = PositionEmbedding(self.max_text_len + 10, latent_dim, dropout)

    def _init_clip(self, latent_dim, transformer_layers, nhead, dropout, preload_features=False):
        from mGPT.archs.mgpt_rvq_hierarchical import TEXT_ENCODER_CONFIGS
        clip_dim = TEXT_ENCODER_CONFIGS['clip']['dim']  # 512

        if not preload_features:
            from mGPT.archs.mgpt_rvq_hierarchical import TextEncoder
            self.clip_encoder = TextEncoder('clip', freeze=True)

        self.input_proj = nn.Linear(clip_dim, latent_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=latent_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.pos_enc = PositionEmbedding(self.max_text_len + 10, latent_dim, dropout)

    def forward(self, texts: list) -> tuple:
        """
        Args:
            texts: list of strings

        Returns:
            tokens: [B, L_t, C] L2-normalized text token embeddings
            mask: [B, L_t] validity mask
        """
        if self.encoder_type == 'distilbert':
            return self._forward_distilbert(texts)
        else:
            return self._forward_clip(texts)

    def _forward_distilbert(self, texts: list) -> tuple:
        device = next(self.bert.parameters()).device

        # Tokenize
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Frozen DistilBERT forward
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            feats = outputs.last_hidden_state  # [B, L_t, 768]

        # Project and transform
        x = self.input_proj(feats)  # [B, L_t, C]
        x = self.pos_enc(x)

        pad_mask = ~attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        tokens = F.normalize(x, dim=-1)
        mask = attention_mask.float()

        return tokens, mask

    def _forward_clip(self, texts: list) -> tuple:
        with torch.no_grad():
            feats, clip_mask = self.clip_encoder(texts)  # [B, 1, 512], None

        x = self.input_proj(feats)  # [B, L, C]
        x = self.pos_enc(x)
        x = self.transformer(x)

        tokens = F.normalize(x, dim=-1)
        B, L, _ = tokens.shape
        mask = torch.ones(B, L, device=tokens.device) if clip_mask is None else clip_mask.float()

        return tokens, mask

    def forward_from_feats(self, text_feats: torch.Tensor, text_mask: torch.Tensor) -> tuple:
        """
        Forward pass from pre-extracted text features (skips frozen encoder).

        Args:
            text_feats: [B, seq, D] pre-extracted features
            text_mask: [B, seq] validity mask (1=valid, 0=pad)

        Returns:
            tokens: [B, seq, C] L2-normalized text token embeddings
            mask: [B, seq] validity mask
        """
        x = self.input_proj(text_feats)
        x = self.pos_enc(x)

        if text_mask is not None:
            pad_mask = ~text_mask.bool()
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)

        tokens = F.normalize(x, dim=-1)

        if text_mask is None:
            B, L, _ = tokens.shape
            text_mask = torch.ones(B, L, device=tokens.device)

        return tokens, text_mask.float()


# ============================================================
# Reconstruction Decoder
# ============================================================

class ReconstructionDecoder(nn.Module):
    """
    Auxiliary decoder for masked motion token reconstruction (Eq. 10-11 in paper).

    Concatenates tokens from all modalities (with some motion tokens masked),
    then uses a transformer to reconstruct the masked motion tokens.
    """

    def __init__(self, latent_dim: int = 512, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        # Modality type embeddings
        self.modality_embed = nn.Embedding(3, latent_dim)  # 0=motion, 1=text, 2=audio

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead,
            dim_feedforward=latent_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_mask: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            motion_tokens: [B, L_m, C] (with some tokens zeroed/masked)
            motion_mask: [B, L_m]
            text_tokens: [B, L_t, C]
            text_mask: [B, L_t]
            audio_tokens: [B, L_a, C]
            audio_mask: [B, L_a]
            mask_indices: [B, L_m] bool tensor (True = masked position to reconstruct)

        Returns:
            reconstructed: [B, L_m, C] reconstructed motion tokens (only masked positions matter)
        """
        B = motion_tokens.shape[0]
        device = motion_tokens.device

        # Add modality type embeddings
        m_type = self.modality_embed(torch.zeros(B, motion_tokens.shape[1], dtype=torch.long, device=device))
        t_type = self.modality_embed(torch.ones(B, text_tokens.shape[1], dtype=torch.long, device=device))
        a_type = self.modality_embed(torch.full((B, audio_tokens.shape[1]), 2, dtype=torch.long, device=device))

        m_input = motion_tokens + m_type
        t_input = text_tokens + t_type
        a_input = audio_tokens + a_type

        # Concatenate all modalities
        concat_tokens = torch.cat([m_input, t_input, a_input], dim=1)  # [B, L_m+L_t+L_a, C]
        concat_mask = torch.cat([motion_mask, text_mask, audio_mask], dim=1)  # [B, L_total]

        pad_mask = ~concat_mask.bool()
        x = self.transformer(concat_tokens, src_key_padding_mask=pad_mask)

        # Extract motion portion and project
        L_m = motion_tokens.shape[1]
        motion_reconstructed = self.output_proj(x[:, :L_m, :])  # [B, L_m, C]

        return motion_reconstructed


# ============================================================
# Main Model: FineGrainedContrastiveModel
# ============================================================

class FineGrainedContrastiveModel(pl.LightningModule):
    """
    Fine-grained multi-modal contrastive model for motion retrieval.

    Encoders (frozen base + trainable heads):
        - VaeMotionEncoder: frozen RVQVae encoder + trainable transformer
        - MemoryRetrievalAudioEncoder: WavLM + memory-retrieval compression
        - SequenceTextEncoder: DistilBERT/CLIP + transformer

    Loss:
        - Sequence-level alignment loss (KL divergence) for all 3 modality pairs
        - Reconstruction loss for masked motion tokens
    """

    def __init__(
        self,
        # Encoder configs
        speech_encoder_type: str = 'wavlm-large',
        text_encoder_type: str = 'distilbert',
        latent_dim: int = 512,
        # Motion encoder (VAE-based)
        vae_config: dict = None,
        vae_checkpoint: str = None,
        motion_num_layers: int = 4,
        motion_nhead: int = 8,
        # Audio encoder
        audio_num_memory: int = 128,
        audio_num_attn_layers: int = 2,
        audio_target_length: int = 32,
        # Text encoder
        text_transformer_layers: int = 2,
        text_max_len: int = 77,
        # Loss
        temperature_init: float = 0.07,
        lambda_recon: float = 0.1,
        mask_ratio: float = 0.5,
        # Reconstruction
        recon_num_layers: int = 2,
        recon_nhead: int = 8,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 200,
        # Precomputed features
        preload_features: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae_config'])
        self.lambda_recon = lambda_recon
        self.mask_ratio = mask_ratio
        self.preload_features = preload_features

        # --- Encoders (preload_features propagated to skip frozen base models) ---
        self.motion_encoder = VaeMotionEncoder(
            vae_config=vae_config,
            vae_checkpoint=vae_checkpoint,
            latent_dim=latent_dim,
            num_layers=motion_num_layers,
            nhead=motion_nhead,
            preload_features=preload_features,
        )

        self.audio_encoder = MemoryRetrievalAudioEncoder(
            speech_encoder_type=speech_encoder_type,
            latent_dim=latent_dim,
            num_memory_tokens=audio_num_memory,
            num_attn_layers=audio_num_attn_layers,
            target_length=audio_target_length,
            nhead=motion_nhead,
            preload_features=preload_features,
        )

        self.text_encoder = SequenceTextEncoder(
            encoder_type=text_encoder_type,
            latent_dim=latent_dim,
            transformer_layers=text_transformer_layers,
            nhead=motion_nhead,
            max_text_len=text_max_len,
            preload_features=preload_features,
        )

        # --- Token weight heads for sequence-level similarity ---
        # 3 weight heads: one per modality, shared across pairs (paper Eq. 8)
        # For pair (x, y): weight_head_x computes w_x, weight_head_y computes w_y
        self.weight_heads = nn.ModuleDict({
            'speech': nn.Linear(latent_dim, 1),
            'text': nn.Linear(latent_dim, 1),
            'motion': nn.Linear(latent_dim, 1),
        })

        # --- Learnable temperature ---
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

        # --- Reconstruction decoder ---
        self.recon_decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            num_layers=recon_num_layers,
            nhead=recon_nhead,
        )

        # Mask token for masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

    def encode_motion(self, motion: torch.Tensor, lengths: list) -> tuple:
        return self.motion_encoder(motion, lengths)

    def encode_speech(self, audio_waveforms: torch.Tensor, audio_lengths: list = None) -> tuple:
        return self.audio_encoder(audio_waveforms, audio_lengths)

    def encode_text(self, texts: list) -> tuple:
        return self.text_encoder(texts)

    def _compute_all_alignment_losses(
        self, speech_tokens, speech_mask, text_tokens, text_mask, motion_tokens, motion_mask,
    ):
        """Compute alignment losses for all 3 modality pairs.

        Forces FP32 to prevent gradient overflow in FP16 (standard practice in CLIP/OpenCLIP).
        """
        # Cast to FP32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            speech_tokens = speech_tokens.float()
            text_tokens = text_tokens.float()
            motion_tokens = motion_tokens.float()
            speech_mask = speech_mask.float()
            text_mask = text_mask.float()
            motion_mask = motion_mask.float()

            # Clamp BEFORE exp to prevent gradient overflow
            # exp(4.6) ≈ 100, so this effectively clamps temperature to [1, 100]
            temperature = self.logit_scale.clamp(min=0.0, max=4.6).exp()

            # Speech <-> Text
            loss_st = compute_alignment_loss(
                speech_tokens, text_tokens, speech_mask, text_mask,
                self.weight_heads['speech'], self.weight_heads['text'],
                temperature,
            )

            # Speech <-> Motion
            loss_sm = compute_alignment_loss(
                speech_tokens, motion_tokens, speech_mask, motion_mask,
                self.weight_heads['speech'], self.weight_heads['motion'],
                temperature,
            )

            # Text <-> Motion
            loss_tm = compute_alignment_loss(
                text_tokens, motion_tokens, text_mask, motion_mask,
                self.weight_heads['text'], self.weight_heads['motion'],
                temperature,
            )

        return loss_st, loss_sm, loss_tm

    def _compute_reconstruction_loss(
        self, motion_tokens, motion_mask, text_tokens, text_mask, audio_tokens, audio_mask,
    ):
        """Randomly mask motion tokens and reconstruct them."""
        # Force FP32
        with torch.cuda.amp.autocast(enabled=False):
            motion_tokens = motion_tokens.float()
            text_tokens = text_tokens.float()
            audio_tokens = audio_tokens.float()
            motion_mask = motion_mask.float()
            text_mask = text_mask.float()
            audio_mask = audio_mask.float()

            B, L_m, C = motion_tokens.shape

            # Create random mask
            mask_indices = torch.rand(B, L_m, device=motion_tokens.device) < self.mask_ratio
            # Don't mask padded positions
            mask_indices = mask_indices & (motion_mask.bool())

            # Replace masked tokens with learnable mask token
            masked_motion = motion_tokens.clone()
            masked_motion[mask_indices] = self.mask_token.float().squeeze(0).squeeze(0)

            # Reconstruct
            reconstructed = self.recon_decoder(
                masked_motion, motion_mask, text_tokens, text_mask, audio_tokens, audio_mask, mask_indices,
            )

            # L2 loss on masked positions only
            if mask_indices.any():
                target = motion_tokens.detach()  # stop gradient from target
                diff = reconstructed[mask_indices] - target[mask_indices]
                loss = (diff ** 2).mean()
            else:
                loss = torch.tensor(0.0, device=motion_tokens.device)

        return loss

    def _encode_all(self, batch):
        """Encode all modalities, dispatching based on preload_features."""
        if self.preload_features:
            motion_tokens, motion_mask = self.motion_encoder.forward_from_feats(
                batch['motion_feats'], batch['motion_length'])
            speech_tokens, speech_mask = self.audio_encoder.forward_from_feats(
                batch['speech_feats'], batch['speech_length'])
            text_tokens, text_mask = self.text_encoder.forward_from_feats(
                batch['text_feats'], batch['text_mask'])
        else:
            motion_tokens, motion_mask = self.encode_motion(
                batch['motion'], batch['length'])
            speech_tokens, speech_mask = self.encode_speech(
                batch['audio'], batch.get('audio_length', None))
            text_tokens, text_mask = self.encode_text(batch['text'])
        return motion_tokens, motion_mask, speech_tokens, speech_mask, text_tokens, text_mask

    def _shared_step(self, batch, prefix):
        texts = batch['text']

        # Encode all modalities → sequence-level tokens
        motion_tokens, motion_mask, speech_tokens, speech_mask, text_tokens, text_mask = \
            self._encode_all(batch)

        # Alignment losses
        loss_st, loss_sm, loss_tm = self._compute_all_alignment_losses(
            speech_tokens, speech_mask, text_tokens, text_mask, motion_tokens, motion_mask,
        )
        loss_align = loss_st + loss_sm + loss_tm  # sum, not average (paper Eq. 4, weight=1 each)

        # Reconstruction loss
        loss_recon = self._compute_reconstruction_loss(
            motion_tokens, motion_mask, text_tokens, text_mask, speech_tokens, speech_mask,
        )

        # Total loss
        total = loss_align + self.lambda_recon * loss_recon

        # NaN detection — prints once then stops checking
        if torch.isnan(total) and prefix == 'train' and not getattr(self, '_nan_logged', False):
            self._nan_logged = True
            T = self.logit_scale.clamp(min=0.0, max=4.6).exp().item()
            print(f"\n{'='*60}")
            print(f"[NaN DETECTED] step={self.global_step}")
            print(f"  logit_scale={self.logit_scale.item():.4f}, temperature={T:.4f}")
            print(f"  loss_st={loss_st.item()}, loss_sm={loss_sm.item()}, loss_tm={loss_tm.item()}")
            print(f"  loss_recon={loss_recon.item()}")
            for name, tok in [('motion', motion_tokens), ('speech', speech_tokens), ('text', text_tokens)]:
                has_nan = torch.isnan(tok).any().item()
                has_inf = torch.isinf(tok).any().item()
                print(f"  {name}_tokens: min={tok.min().item():.4f} max={tok.max().item():.4f} nan={has_nan} inf={has_inf}")
            # Check params for NaN
            for pname, p in self.named_parameters():
                if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                    print(f"  PARAM NaN/Inf: {pname}")
            print(f"{'='*60}\n")

        # Logging
        log_kwargs = dict(prog_bar=True, sync_dist=True, batch_size=len(texts))
        if prefix == 'train':
            log_kwargs.update(on_step=True, on_epoch=True)
        else:
            log_kwargs.update(on_step=False, on_epoch=True)

        self.log_dict({
            f'{prefix}/loss': total,
            f'{prefix}/loss_align': loss_align,
            f'{prefix}/loss_st': loss_st,
            f'{prefix}/loss_sm': loss_sm,
            f'{prefix}/loss_tm': loss_tm,
            f'{prefix}/loss_recon': loss_recon,
            f'{prefix}/temperature': self.logit_scale.clamp(min=0.0, max=4.6).exp(),
        }, **log_kwargs)

        return total

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        return self._shared_step(batch, 'train')

    # ---- Validation with retrieval metrics ----

    def on_validation_epoch_start(self):
        self._val_speech = []
        self._val_text = []
        self._val_motion = []
        self._val_speech_mask = []
        self._val_text_mask = []
        self._val_motion_mask = []

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        texts = batch['text']

        motion_tokens, motion_mask, speech_tokens, speech_mask, text_tokens, text_mask = \
            self._encode_all(batch)

        self._val_speech.append(speech_tokens.detach())
        self._val_text.append(text_tokens.detach())
        self._val_motion.append(motion_tokens.detach())
        self._val_speech_mask.append(speech_mask.detach())
        self._val_text_mask.append(text_mask.detach())
        self._val_motion_mask.append(motion_mask.detach())

        # Log validation loss
        loss_st, loss_sm, loss_tm = self._compute_all_alignment_losses(
            speech_tokens, speech_mask, text_tokens, text_mask, motion_tokens, motion_mask,
        )
        loss_align = loss_st + loss_sm + loss_tm
        loss_recon = self._compute_reconstruction_loss(
            motion_tokens, motion_mask, text_tokens, text_mask, speech_tokens, speech_mask,
        )
        total = loss_align + self.lambda_recon * loss_recon

        self.log_dict({
            'val/loss': total,
            'val/loss_align': loss_align,
            'val/loss_recon': loss_recon,
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(texts))

        return total

    def on_validation_epoch_end(self):
        if not self._val_speech:
            return

        # For retrieval, we compute global embeddings from token sequences
        # using weighted pooling (same weights as in loss)
        speech_emb = self._pool_for_retrieval(self._val_speech, self._val_speech_mask, 'speech')
        text_emb = self._pool_for_retrieval(self._val_text, self._val_text_mask, 'text')
        motion_emb = self._pool_for_retrieval(self._val_motion, self._val_motion_mask, 'motion')

        # Gather from all GPUs in DDP so retrieval metrics use full val set
        if self.trainer.world_size > 1:
            speech_emb = self.all_gather(speech_emb.to(self.device)).flatten(0, 1).cpu()
            text_emb = self.all_gather(text_emb.to(self.device)).flatten(0, 1).cpu()
            motion_emb = self.all_gather(motion_emb.to(self.device)).flatten(0, 1).cpu()

        # Compute R@K for all 6 directions
        pairs = [
            ('S2T', speech_emb, text_emb),
            ('T2S', text_emb, speech_emb),
            ('S2M', speech_emb, motion_emb),
            ('M2S', motion_emb, speech_emb),
            ('T2M', text_emb, motion_emb),
            ('M2T', motion_emb, text_emb),
        ]

        all_r1, all_r5, all_r10 = [], [], []
        metrics = {}
        for name, q, g in pairs:
            r1, r5, r10 = self._recall_at_k(q, g, ks=(1, 5, 10))
            metrics[f'val/{name}_R@1'] = r1
            metrics[f'val/{name}_R@5'] = r5
            metrics[f'val/{name}_R@10'] = r10
            all_r1.append(r1)
            all_r5.append(r5)
            all_r10.append(r10)

        metrics['val/avg_R@1'] = np.mean(all_r1)
        metrics['val/avg_R@5'] = np.mean(all_r5)
        metrics['val/avg_R@10'] = np.mean(all_r10)

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        # Cleanup
        self._val_speech.clear()
        self._val_text.clear()
        self._val_motion.clear()
        self._val_speech_mask.clear()
        self._val_text_mask.clear()
        self._val_motion_mask.clear()

    def _pool_for_retrieval(self, token_list, mask_list, modality_name):
        """Pool sequence tokens into global embeddings for retrieval evaluation."""
        # Pad all batches to the same max sequence length before concat
        max_len = max(t.shape[1] for t in token_list)
        padded_tokens = []
        padded_masks = []
        for tokens, mask in zip(token_list, mask_list):
            L = tokens.shape[1]
            if L < max_len:
                tokens = F.pad(tokens, (0, 0, 0, max_len - L))  # pad seq dim
                mask = F.pad(mask, (0, max_len - L))              # pad with 0
            padded_tokens.append(tokens)
            padded_masks.append(mask)

        # Stay on GPU — no CPU shuffling (safe for DDP)
        all_tokens = torch.cat(padded_tokens, dim=0)  # [N, max_len, C]
        all_masks = torch.cat(padded_masks, dim=0)     # [N, max_len]

        # Weighted mean pooling using weight head
        with torch.no_grad():
            w = self.weight_heads[modality_name](all_tokens).squeeze(-1)  # [N, max_len]
            w = w.masked_fill(all_masks == 0, float('-inf'))
            w = F.softmax(w, dim=-1)
            w = w.nan_to_num(0.0)
            pooled = (w.unsqueeze(-1) * all_tokens).sum(dim=1)  # [N, C]
            pooled = F.normalize(pooled, dim=-1)

        return pooled.cpu()

    @staticmethod
    def _recall_at_k(query, gallery, ks=(1, 5, 10)):
        sim = query @ gallery.T
        sorted_indices = sim.argsort(dim=1, descending=True)
        gt = torch.arange(sim.shape[0]).unsqueeze(1)
        ranks = (sorted_indices == gt).nonzero(as_tuple=True)[1]
        return tuple((ranks < k).float().mean().item() * 100.0 for k in ks)

    def configure_optimizers(self):
        # Collect all trainable parameters (excluding frozen VAE & frozen speech/text base)
        params = []
        # Motion encoder: only trainable parts (input_proj, pos_enc, transformer_1, transformer_2)
        params += list(self.motion_encoder.input_proj.parameters())
        params += list(self.motion_encoder.pos_enc.parameters())
        params += list(self.motion_encoder.transformer_1.parameters())
        params += list(self.motion_encoder.transformer_2.parameters())
        # Audio encoder: trainable parts
        params += list(self.audio_encoder.input_proj.parameters())
        params += [self.audio_encoder.memory_tokens]
        params += list(self.audio_encoder.cross_attn_layers.parameters())
        params += list(self.audio_encoder.cross_attn_norms.parameters())
        params += list(self.audio_encoder.pos_enc.parameters())
        # Text encoder: trainable parts
        params += list(self.text_encoder.input_proj.parameters())
        params += list(self.text_encoder.transformer.parameters())
        params += list(self.text_encoder.pos_enc.parameters())
        # Weight heads, temperature, mask token, reconstruction decoder
        params += list(self.weight_heads.parameters())
        params += [self.logit_scale, self.mask_token]
        params += list(self.recon_decoder.parameters())

        optimizer = torch.optim.AdamW(
            params, lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay)
        )

        # Linear decay: lr decays to 1e-5 after first 100 epochs
        def lr_lambda(epoch):
            if epoch < self.hparams.max_epochs // 2:
                return 1.0 - (1.0 - 0.1) * epoch / (self.hparams.max_epochs // 2)
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]
