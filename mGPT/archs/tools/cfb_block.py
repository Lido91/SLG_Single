"""
Condition Fusion Block (CFB)

A lightweight pre-fusion module that injects condition features (e.g. speech
or lower-level quantizer embeddings) into a stream of "previous Q features".

Pipeline per layer:
    x → LN → CrossAttn(Q=x, K/V=cond) ─┐
                                        + residual
    x → LN → FFN (optionally gated)  ──┘
                                        + residual

A CFBStack stacks M such layers. The output has the same shape as the input
stream and is "condition-aware", ready to be consumed by a downstream decoder
as cross-attention context.
"""

from typing import Optional

import torch
import torch.nn as nn

from .cross_attention import CrossAttention


class GatedFFN(nn.Module):
    """Feed-forward network with an optional learned gate on the residual."""

    def __init__(self, embed_dim: int, fc_rate: int = 4, dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        if use_gate:
            self.gate = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        if self.use_gate:
            g = torch.sigmoid(self.gate(x))
            return g * h
        return h


class ConditionFusionBlock(nn.Module):
    """
    Single CFB layer.

    Args:
        embed_dim: Hidden dimension (must match both stream and condition).
        n_head: Attention heads.
        dropout: Dropout rate for attention and FFN.
        fc_rate: FFN expansion factor (hidden = fc_rate * embed_dim).
        use_gate: If True, apply a learned sigmoid gate on the FFN output
                  (initialized to zero → starts as identity residual).
    """

    def __init__(self, embed_dim: int = 1024, n_head: int = 16,
                 dropout: float = 0.1, fc_rate: int = 4, use_gate: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim, n_head=n_head, drop_out_rate=dropout,
        )
        self.ffn = GatedFFN(
            embed_dim=embed_dim, fc_rate=fc_rate,
            dropout=dropout, use_gate=use_gate,
        )

    def forward(self, x: torch.Tensor, cond_ctx: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) stream of previous Q features.
            cond_ctx: (B, S, D) condition features (text/speech/lower Q).
            cond_mask: (B, S) optional mask for cond_ctx (1=valid).
        Returns:
            (B, T, D) condition-aware stream.
        """
        x = x + self.cross_attn(self.ln1(x), cond_ctx, context_mask=cond_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CFBStack(nn.Module):
    """Stack of M ConditionFusionBlocks with a final LayerNorm."""

    def __init__(self, num_layers: int, embed_dim: int = 1024, n_head: int = 16,
                 dropout: float = 0.1, fc_rate: int = 4, use_gate: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConditionFusionBlock(
                embed_dim=embed_dim, n_head=n_head, dropout=dropout,
                fc_rate=fc_rate, use_gate=use_gate,
            )
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, cond_ctx: torch.Tensor,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cond_ctx, cond_mask)
        return self.ln_out(x)
