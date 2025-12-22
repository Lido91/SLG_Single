"""
Stage 3: Residual Transformer for Motion Refinement

Predicts residual tokens (Q1-Q5) conditioned on previous quantizer layers,
progressively refining motion quality.

Based on MoMask: https://arxiv.org/abs/2312.00063
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from functools import partial
from einops import rearrange, repeat

from .tools.mask_tools import (
    lengths_to_mask, uniform, cosine_schedule, top_k, gumbel_sample,
    q_schedule, cal_performance, eval_decorator
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
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess(nn.Module):
    """Output projection to code dimension."""

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


class ResidualTransformer(nn.Module):
    """
    Stage 3: Residual Transformer for refining motion tokens.

    Predicts tokens for quantizer layers Q1 through Q_{num_quantizers-1},
    conditioned on the cumulative sum of previous layers' embeddings.

    Args:
        num_tokens: codebook size (number of motion tokens per quantizer)
        code_dim: dimension of token embeddings
        latent_dim: transformer hidden dimension
        ff_size: feedforward dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        dropout: dropout rate
        cond_drop_prob: classifier-free guidance dropout probability
        clip_dim: CLIP text embedding dimension
        clip_version: CLIP model version
        num_quantizers: number of RVQ quantizers
        shared_codebook: whether RVQ uses shared codebook
        share_weight: share embedding and output projection weights
    """

    def __init__(
        self,
        num_tokens: int = 512,
        code_dim: int = 512,
        latent_dim: int = 384,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 6,
        dropout: float = 0.2,
        cond_drop_prob: float = 0.2,
        clip_dim: int = 512,
        clip_version: str = 'ViT-B/32',
        num_quantizers: int = 3,
        shared_codebook: bool = False,
        share_weight: bool = True,
        cond_mode: str = 'text',
        **kwargs
    ):
        super().__init__()

        print(f'ResidualTransformer: latent_dim={latent_dim}, ff_size={ff_size}, '
              f'nlayers={num_layers}, nheads={num_heads}, dropout={dropout}')

        self.num_tokens = num_tokens
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.cond_drop_prob = cond_drop_prob
        self.cond_mode = cond_mode
        self.num_quantizers = num_quantizers
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight
        self.clip_dim = clip_dim

        # Special token ID for padding
        _num_tokens = num_tokens + 1  # +1 for PAD token
        self.pad_id = num_tokens

        # Input/output processing
        self.input_process = InputProcess(code_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Quantizer layer embedding
        self.encode_quant = partial(F.one_hot, num_classes=num_quantizers)
        self.quant_emb = nn.Linear(num_quantizers, latent_dim)

        # Condition embedding
        if cond_mode == 'text':
            self.cond_emb = nn.Linear(clip_dim, latent_dim)
        else:
            self.cond_emb = nn.Identity()

        # Token embeddings and output projections
        # We need embeddings for quantizers 0 to num_quantizers-2 (to predict 1 to num_quantizers-1)
        if shared_codebook:
            # Single shared embedding
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(num_quantizers - 1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(num_quantizers - 1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(num_quantizers - 1, _num_tokens)
        else:
            if share_weight:
                # Share middle layers between embedding and projection
                self.embed_proj_shared_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(num_quantizers - 2, _num_tokens, code_dim))
                )
                self.token_embed_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim))
                )
                self.output_proj_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim))
                )
                self.output_proj_bias = None
            else:
                # Separate embeddings for each quantizer layer
                self.token_embed_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(num_quantizers - 1, _num_tokens, code_dim))
                )
                self.output_proj_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(num_quantizers - 1, _num_tokens, code_dim))
                )
                self.output_proj_bias = nn.Parameter(
                    torch.zeros(size=(num_quantizers - 1, _num_tokens))
                )

        # Initialize weights
        self.apply(self._init_weights)

        # Load CLIP for text encoding
        if cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self._load_and_freeze_clip(clip_version)

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

    def process_embed_proj_weight(self):
        """Process weight sharing for non-shared codebook mode."""
        if self.share_weight and (not self.shared_codebook):
            self.output_proj_weight = torch.cat(
                [self.embed_proj_shared_weight, self.output_proj_weight_], dim=0
            )
            self.token_embed_weight = torch.cat(
                [self.token_embed_weight_, self.embed_proj_shared_weight], dim=0
            )

    def output_project(self, logits: torch.Tensor, qids: torch.Tensor) -> torch.Tensor:
        """
        Project output to token logits for specific quantizer layers.

        Args:
            logits: (B, code_dim, N) output features
            qids: (B,) quantizer layer indices (0 to num_quantizers-2, already adjusted by caller)

        Returns:
            output: (B, num_tokens, N) token logits
        """
        # Get projection weights for these quantizer layers (qids already 0-indexed by caller)
        output_proj_weight = self.output_proj_weight[qids]  # (B, num_tokens, code_dim)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        # Project: (B, num_tokens, code_dim) @ (B, code_dim, N) -> (B, num_tokens, N)
        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)

        if output_proj_bias is not None:
            output = output + output_proj_bias.unsqueeze(-1)

        return output

    def trans_forward(
        self,
        motion_codes: torch.Tensor,
        qids: torch.Tensor,
        cond: torch.Tensor,
        padding_mask: torch.Tensor,
        force_mask: bool = False
    ) -> torch.Tensor:
        """
        Transformer forward pass.

        Args:
            motion_codes: (B, N, code_dim) sum of previous quantizer embeddings
            qids: (B,) quantizer layer indices being predicted
            cond: (B, cond_dim) condition embedding
            padding_mask: (B, N) TRUE for padding positions
            force_mask: force zero condition (for CFG)

        Returns:
            logits: (B, code_dim, N) output features (before projection)
        """
        cond = self.mask_cond(cond, force_mask=force_mask)

        # Process input
        x = self.input_process(motion_codes)  # (N, B, latent_dim)

        # Embed quantizer layer
        q_onehot = self.encode_quant(qids).float().to(x.device)
        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, B, latent_dim)

        # Embed condition
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, B, latent_dim)

        # Add positional encoding
        x = self.position_enc(x)

        # Concatenate: [cond, quant_id, motion_tokens]
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (N+2, B, latent_dim)

        # Extend padding mask
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)

        # Transformer forward
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[2:]  # (N, B, latent_dim)

        # Output projection (to code_dim, not logits yet)
        logits = self.output_process(output)  # (B, code_dim, N)

        return logits

    def forward(
        self,
        all_indices: torch.Tensor,
        y: List[str],
        m_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Training forward pass.

        Args:
            all_indices: (B, N, num_quantizers) all RVQ token indices
            y: list of text descriptions
            m_lens: (B,) sequence lengths in tokens

        Returns:
            ce_loss: cross-entropy loss
            pred_id: predicted token indices
            acc: prediction accuracy
        """
        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        # Create non-padding mask
        non_pad_mask = lengths_to_mask(m_lens, ntokens)

        # Apply padding to all quantizer layers
        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)

        # Randomly sample which quantizer layer to predict (1 to num_quantizers-1)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # Get token embeddings for all layers except the last
        token_embed = repeat(self.token_embed_weight, 'q c d -> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        all_codes = token_embed.gather(1, gather_indices)  # (B, N, code_dim, num_q-1)

        # Cumulative sum of embeddings
        cumsum_codes = torch.cumsum(all_codes, dim=-1)  # (B, N, code_dim, num_q-1)

        # Get target indices and history for active layers
        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (B, N)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]  # (B, N, code_dim)

        # Encode text condition
        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        else:
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True

        # Forward pass
        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
        logits = self.output_project(logits, active_q_layers - 1)  # MoMask: qids-1 for 0-indexed projection

        # Compute loss
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(
        self,
        motion_codes: torch.Tensor,
        q_id: int,
        cond_vector: torch.Tensor,
        padding_mask: torch.Tensor,
        cond_scale: float = 2.0,
        force_mask: bool = False
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)

        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
            logits = self.output_project(logits, qids - 1)  # MoMask: qids-1 for 0-indexed projection
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask)
        logits = self.output_project(logits, qids - 1)  # MoMask: qids-1 for 0-indexed projection

        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids - 1)  # MoMask: qids-1 for 0-indexed projection

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        motion_ids: torch.Tensor,
        conds: List[str],
        m_lens: torch.Tensor,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
        cond_scale: float = 2.0,
        num_res_layers: int = -1
    ) -> torch.Tensor:
        """
        Generate residual tokens (Q1 to Q_{num_quantizers-1}) from Q0.

        Args:
            motion_ids: (B, N) Q0 token indices from MaskTransformer
            conds: list of text descriptions
            m_lens: (B,) sequence lengths in tokens
            temperature: sampling temperature
            topk_filter_thres: top-k filtering threshold
            cond_scale: classifier-free guidance scale
            num_res_layers: number of residual layers to generate (-1 for all)

        Returns:
            all_indices: (B, N, num_quantizers) all token indices
        """
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        # Encode text condition
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        else:
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)

        all_indices = [motion_ids]
        history_sum = 0  # MoMask style: start with 0, accumulate with +=

        num_quant_layers = self.num_quantizers if num_res_layers == -1 else num_res_layers + 1

        for i in range(1, num_quant_layers):
            # Get embeddings for previous layer
            token_embed = self.token_embed_weight[i - 1]  # (num_tokens+1, code_dim)
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)  # MoMask style: += accumulation

            # Forward with CFG
            logits = self.forward_with_cond_scale(
                history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale
            )
            logits = logits.permute(0, 2, 1)  # (B, N, num_tokens)

            # Top-k filtering and sampling
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            # Apply padding
            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        # Stack all indices
        all_indices = torch.stack(all_indices, dim=-1)  # (B, N, num_quantizers)

        # Mark padding as -1
        all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)

        return all_indices

    def parameters_wo_clip(self):
        """Return parameters excluding CLIP model."""
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]