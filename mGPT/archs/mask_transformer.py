"""
Stage 2: Masked Transformer for Text-to-Motion Generation

Generates coarse motion tokens (Q0) from text descriptions using
masked generative modeling (similar to BERT/MaskGIT).

Based on MoMask: https://arxiv.org/abs/2312.00063
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple
from functools import partial
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical

from .tools.mask_tools import (
    lengths_to_mask, uniform, cosine_schedule, top_k, gumbel_sample,
    get_mask_subset_prob, cal_performance, eval_decorator
)


class InputProcess(nn.Module):
    """Project token embeddings to latent dimension."""

    def __init__(self, input_feats: int, latent_dim: int):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_feats) token embeddings

        Returns:
            x: (N, B, latent_dim) projected embeddings
        """
        x = x.permute(1, 0, 2)  # (N, B, input_feats)
        x = self.poseEmbedding(x)  # (N, B, latent_dim)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, B, d_model) input tensor

        Returns:
            x: (N, B, d_model) with positional encoding added
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess(nn.Module):
    """BERT-style output projection with dense + GELU + LayerNorm."""

    def __init__(self, out_feats: int, latent_dim: int):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (N, B, latent_dim)

        Returns:
            output: (B, out_feats, N)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # (N, B, out_feats)
        output = output.permute(1, 2, 0)  # (B, out_feats, N)
        return output


class MaskTransformer(nn.Module):
    """
    Stage 2: Masked Transformer for coarse motion token generation.

    Generates Q0 tokens from text using masked generative modeling:
    1. During training: randomly mask tokens and predict them
    2. During inference: iteratively unmask from all-masked state

    Args:
        num_tokens: codebook size (number of motion tokens)
        code_dim: dimension of token embeddings
        latent_dim: transformer hidden dimension
        ff_size: feedforward dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        dropout: dropout rate
        cond_drop_prob: classifier-free guidance dropout probability
        clip_dim: CLIP text embedding dimension
        clip_version: CLIP model version (e.g., 'ViT-B/32')
        num_quantizers: number of RVQ quantizers (for compatibility)
    """

    def __init__(
        self,
        num_tokens: int = 512,
        code_dim: int = 512,
        latent_dim: int = 384,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 6,
        dropout: float = 0.1,
        cond_drop_prob: float = 0.1,
        clip_dim: int = 512,
        clip_version: str = 'ViT-B/32',
        num_quantizers: int = 3,
        cond_mode: str = 'text',
        **kwargs
    ):
        super().__init__()

        print(f'MaskTransformer: latent_dim={latent_dim}, ff_size={ff_size}, '
              f'nlayers={num_layers}, nheads={num_heads}, dropout={dropout}')

        self.num_tokens = num_tokens
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.cond_drop_prob = cond_drop_prob
        self.cond_mode = cond_mode
        self.num_quantizers = num_quantizers
        self.clip_dim = clip_dim

        # Special token IDs
        _num_tokens = num_tokens + 2  # +2 for MASK and PAD tokens
        self.mask_id = num_tokens
        self.pad_id = num_tokens + 1

        # Token embedding
        self.token_emb = nn.Embedding(_num_tokens, code_dim)

        # Input/output processing
        self.input_process = InputProcess(code_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.output_process = OutputProcess(out_feats=num_tokens, latent_dim=latent_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (seq, batch, dim)
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Condition embedding
        if cond_mode == 'text':
            self.cond_emb = nn.Linear(clip_dim, latent_dim)
        else:
            self.cond_emb = nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

        # Load CLIP for text encoding
        if cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self._load_and_freeze_clip(clip_version)

        # Noise schedule
        self.noise_schedule = cosine_schedule

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _load_and_freeze_clip(self, clip_version: str):
        """Load and freeze CLIP model for text encoding."""
        import clip
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        import clip
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Apply classifier-free guidance dropout to condition."""
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(
        self,
        motion_ids: torch.Tensor,
        cond: torch.Tensor,
        padding_mask: torch.Tensor,
        force_mask: bool = False
    ) -> torch.Tensor:
        """
        Transformer forward pass.

        Args:
            motion_ids: (B, N) token indices
            cond: (B, cond_dim) condition embedding
            padding_mask: (B, N) TRUE for padding positions
            force_mask: force zero condition (for CFG)

        Returns:
            logits: (B, num_tokens, N) prediction logits
        """
        cond = self.mask_cond(cond, force_mask=force_mask)

        # Embed tokens
        x = self.token_emb(motion_ids)  # (B, N, code_dim)
        x = self.input_process(x)  # (N, B, latent_dim)

        # Embed condition
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, B, latent_dim)

        # Add positional encoding
        x = self.position_enc(x)

        # Concatenate condition token
        xseq = torch.cat([cond, x], dim=0)  # (N+1, B, latent_dim)

        # Extend padding mask for condition token
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        # Transformer forward
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]  # (N, B, latent_dim)

        # Project to logits
        logits = self.output_process(output)  # (B, num_tokens, N)

        return logits

    def forward(
        self,
        ids: torch.Tensor,
        y: List[str],
        m_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Training forward pass with random masking.

        Args:
            ids: (B, N) ground truth token indices
            y: list of text descriptions
            m_lens: (B,) sequence lengths in tokens

        Returns:
            ce_loss: cross-entropy loss
            pred_id: (B, N) predicted token indices
            acc: prediction accuracy
        """
        bs, ntokens = ids.shape
        device = ids.device

        # Create non-padding mask (TRUE for valid positions)
        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        # Encode text condition
        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        else:
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True

        # Sample random masking ratio (cosine schedule)
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        # Create random mask
        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask

        # Target: predict masked tokens, ignore non-masked
        labels = torch.where(mask, ids, self.mask_id)

        # Apply BERT-style masking
        x_ids = ids.clone()

        # 10% replace with random token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)

        # 88% of remaining replace with MASK token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        # Forward pass
        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)

        # Compute loss
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(
        self,
        motion_ids: torch.Tensor,
        cond_vector: torch.Tensor,
        padding_mask: torch.Tensor,
        cond_scale: float = 3.0,
        force_mask: bool = False
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)

        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conds: List[str],
        m_lens: torch.Tensor,
        timesteps: int = 10,
        cond_scale: float = 4.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
        gsample: bool = False,
        force_mask: bool = False
    ) -> torch.Tensor:
        """
        Generate Q0 tokens from text using iterative unmasking.

        Args:
            conds: list of text descriptions
            m_lens: (B,) target sequence lengths in tokens
            timesteps: number of unmasking iterations
            cond_scale: classifier-free guidance scale
            temperature: sampling temperature
            topk_filter_thres: top-k filtering threshold
            gsample: use Gumbel sampling instead of categorical
            force_mask: force unconditional generation

        Returns:
            ids: (B, max_len) generated token indices, -1 for padding
        """
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        # Encode text condition
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        else:
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start with all MASK tokens
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)

        starting_temperature = temperature

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)

            # Determine how many tokens to keep masked
            num_token_masked = torch.round(rand_mask_prob * m_lens.float()).clamp(min=1)

            # Select tokens with lowest confidence to re-mask
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            # Forward pass with CFG
            logits = self.forward_with_cond_scale(
                ids, cond_vector, padding_mask,
                cond_scale=cond_scale, force_mask=force_mask
            )
            logits = logits.permute(0, 2, 1)  # (B, N, num_tokens)

            # Top-k filtering
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            # Sample
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            # Update ids at masked positions
            ids = torch.where(is_mask, pred_ids, ids)

            # Update confidence scores
            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)

        # Mark padding as -1
        ids = torch.where(padding_mask, -1, ids)

        return ids

    def parameters_wo_clip(self):
        """Return parameters excluding CLIP model."""
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_token_emb(self, codebook: torch.Tensor):
        """
        Initialize token embeddings from RVQ codebook.

        Args:
            codebook: (num_tokens, code_dim) codebook embeddings
        """
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(
            torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)
        )
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized from codebook!")