"""
Transformer building blocks for VAE Encoder/Decoder.

Bidirectional (non-causal) attention with RoPE, SwiGLU FFN, and U-Net
skip connections. Designed for motion tokenizer (encode/decode), not
autoregressive generation.

Reuses from llamagen_blocks.py:
  RMSNorm, SwiGLUFeedForward, DropPath, apply_rotary_emb, precompute_freqs_cis_1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .llamagen_blocks import (
    RMSNorm,
    SwiGLUFeedForward,
    DropPath,
    apply_rotary_emb,
    precompute_freqs_cis_1d,
)


# ===========================================================================
#  Bidirectional Self-Attention (non-causal, with RoPE)
# ===========================================================================
class BidirectionalAttention(nn.Module):
    """
    Multi-head self-attention WITHOUT causal mask.
    Uses RoPE and F.scaled_dot_product_attention (Flash Attention when available).
    Supports key_padding_mask for variable-length sequences.
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, dim)
            freqs_cis: (S, head_dim//2, 2)
            key_padding_mask: (B, S) — True for valid, False for padding
        """
        B, S, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.dim, dim=-1)

        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # (B, n_head, S, head_dim)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        # Build attention mask from key_padding_mask
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) True=valid, False=pad
            # Need (B, 1, 1, S) for broadcast: 0=attend, -inf=block
            attn_mask = key_padding_mask[:, None, None, :].float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous().view(B, S, self.dim)
        return self.resid_dropout(self.wo(output))


# ===========================================================================
#  Cross-Attention (query attends to memory, with RoPE on query)
# ===========================================================================
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: query from decoder, key/value from memory (e.g. z_hat).
    No RoPE — matches LG-Tok which uses standard nn.MultiheadAttention
    for cross-attention (positional info is already in the representations).
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, n_head, dropout=dropout, batch_first=True,
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, Sq, dim) — query
            memory: (B, Sm, dim) — key/value source
            memory_key_padding_mask: (B, Sm) — True=pad (nn.MHA convention: True=ignore)
        """
        output, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        return self.resid_dropout(output)


# ===========================================================================
#  Transformer Encoder Layer (self-attn only)
# ===========================================================================
class VAETransformerEncoderLayer(nn.Module):
    """
    Pre-norm encoder layer: RMSNorm → BidirectionalAttention → RMSNorm → SwiGLU.
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.attn = BidirectionalAttention(dim, n_head, dropout)
        self.ffn = SwiGLUFeedForward(dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis, key_padding_mask))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ===========================================================================
#  Transformer Decoder Layer (self-attn + cross-attn to memory)
# ===========================================================================
class VAETransformerDecoderLayer(nn.Module):
    """
    Pre-norm decoder layer:
      RMSNorm → BidirectionalAttention (self)
      RMSNorm → CrossAttention (to memory/z_hat)
      RMSNorm → SwiGLU
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.self_attn = BidirectionalAttention(dim, n_head, dropout)
        self.cross_attn = CrossAttentionBlock(dim, n_head, dropout)
        self.ffn = SwiGLUFeedForward(dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.norm1(x), freqs_cis, key_padding_mask))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), memory, memory_key_padding_mask))
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        return x


# ===========================================================================
#  U-Net Transformer Encoder
# ===========================================================================
class UNetTransformerEncoder(nn.Module):
    """
    Transformer encoder with U-Net skip connections.

    num_layers must be odd: (num_layers-1)//2 input blocks + 1 middle + (num_layers-1)//2 output blocks.
    Output blocks receive skip connections: cat([x, skip], dim=-1) → Linear(2d, d).
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        num_layers: int = 9,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert num_layers % 2 == 1, f"num_layers must be odd for U-Net, got {num_layers}"

        n_skip = (num_layers - 1) // 2  # 4 for 9 layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.input_blocks = nn.ModuleList([
            VAETransformerEncoderLayer(dim, n_head, dropout, dpr[i])
            for i in range(n_skip)
        ])
        self.middle_block = VAETransformerEncoderLayer(dim, n_head, dropout, dpr[n_skip])
        self.output_blocks = nn.ModuleList([
            VAETransformerEncoderLayer(dim, n_head, dropout, dpr[n_skip + 1 + i])
            for i in range(n_skip)
        ])
        self.skip_linears = nn.ModuleList([
            nn.Linear(2 * dim, dim, bias=False)
            for _ in range(n_skip)
        ])
        self.norm_pre = RMSNorm(dim)
        self.norm_post = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm_pre(x)

        skips = []
        for block in self.input_blocks:
            x = block(x, freqs_cis, key_padding_mask)
            skips.append(x)

        x = self.middle_block(x, freqs_cis, key_padding_mask)

        for block, skip_linear in zip(self.output_blocks, self.skip_linears):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)
            x = skip_linear(x)
            x = block(x, freqs_cis, key_padding_mask)

        return self.norm_post(x)


# ===========================================================================
#  U-Net Transformer Decoder
# ===========================================================================
class UNetTransformerDecoder(nn.Module):
    """
    Transformer decoder with U-Net skip connections and cross-attention to memory.

    Same U-Net structure as encoder, but each layer has an additional
    cross-attention to memory (e.g. quantized latent codes z_hat).
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        num_layers: int = 9,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert num_layers % 2 == 1, f"num_layers must be odd for U-Net, got {num_layers}"

        n_skip = (num_layers - 1) // 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.input_blocks = nn.ModuleList([
            VAETransformerDecoderLayer(dim, n_head, dropout, dpr[i])
            for i in range(n_skip)
        ])
        self.middle_block = VAETransformerDecoderLayer(dim, n_head, dropout, dpr[n_skip])
        self.output_blocks = nn.ModuleList([
            VAETransformerDecoderLayer(dim, n_head, dropout, dpr[n_skip + 1 + i])
            for i in range(n_skip)
        ])
        self.skip_linears = nn.ModuleList([
            nn.Linear(2 * dim, dim, bias=False)
            for _ in range(n_skip)
        ])
        self.norm_pre = RMSNorm(dim)
        self.norm_post = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm_pre(x)

        skips = []
        for block in self.input_blocks:
            x = block(x, memory, freqs_cis, key_padding_mask, memory_key_padding_mask)
            skips.append(x)

        x = self.middle_block(x, memory, freqs_cis, key_padding_mask, memory_key_padding_mask)

        for block, skip_linear in zip(self.output_blocks, self.skip_linears):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)
            x = skip_linear(x)
            x = block(x, memory, freqs_cis, key_padding_mask, memory_key_padding_mask)

        return self.norm_post(x)
