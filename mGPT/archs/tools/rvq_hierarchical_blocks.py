"""
Hierarchical RVQ-GPT Transformer Blocks

Implements hierarchical prediction for residual quantized codes:
- Q0 Decoder: Standard GPT decoder with cross-attention to text (coarse codes)
- Q1 Decoder: Conditioned GPT decoder that also attends to Q0 embeddings (medium refinement)
- Q2 Decoder: Conditioned GPT decoder that attends to Q0+Q1 embeddings (fine refinement)

Key innovation: Q1 and Q2 are explicitly conditioned on earlier quantizers,
matching the natural RVQ hierarchy: Q0 (coarse) → Q1 (medium) → Q2 (fine)

Adapted from SOKE's hierarchical sign language generation architecture.
"""

import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .cross_attention import CrossAttention


class Q0DecoderBlock(nn.Module):
    """
    Transformer block for Q0 decoder (coarse quantizer).
    Standard GPT block with cross-attention to text.

    Architecture:
        1. Causal self-attention (attend to previous Q0 tokens)
        2. Cross-attention (attend to text features)
        3. Feed-forward network
    """

    def __init__(self, embed_dim=1024, block_size=200, n_head=16,
                 drop_out_rate=0.1, fc_rate=4):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        # Self-attention (causal)
        self.self_attn = CausalSelfAttention(
            embed_dim=embed_dim,
            block_size=block_size,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Cross-attention to text
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, text_context, text_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - Q0 token embeddings
            text_context: (B, S, text_dim) - Text features
            text_mask: (B, S) - Optional text mask

        Returns:
            (B, T, embed_dim) - Updated Q0 embeddings
        """
        # Self-attention
        x = x + self.self_attn(self.ln1(x))

        # Cross-attention to text
        x = x + self.cross_attn(self.ln2(x), text_context, context_mask=text_mask)

        # Feedforward
        x = x + self.mlp(self.ln3(x))

        return x


class Q1DecoderBlock(nn.Module):
    """
    Transformer block for Q1 decoder (medium refinement, conditioned on Q0).

    Architecture:
        1. Causal self-attention (attend to previous Q1 tokens)
        2. Cross-attention to TEXT features
        3. Cross-attention to Q0 embeddings (KEY: hierarchical conditioning)
        4. Feed-forward network

    The critical difference: Q1 decoder attends to BOTH text AND Q0,
    creating the hierarchical dependency P(Q1 | Q0, text).
    """

    def __init__(self, embed_dim=1024, block_size=200, n_head=16,
                 drop_out_rate=0.1, fc_rate=4):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)  # For text cross-attention
        self.ln3 = nn.LayerNorm(embed_dim)  # For Q0 cross-attention
        self.ln4 = nn.LayerNorm(embed_dim)  # For FFN

        # Self-attention (causal)
        self.self_attn = CausalSelfAttention(
            embed_dim=embed_dim,
            block_size=block_size,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Cross-attention to text
        self.text_cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Cross-attention to Q0 (hierarchical conditioning)
        self.q0_cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, text_context, q0_context, text_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - Q1 token embeddings
            text_context: (B, S_text, text_dim) - Text features
            q0_context: (B, T_q0, embed_dim) - Q0 token embeddings
            text_mask: (B, S_text) - Optional text mask

        Returns:
            (B, T, embed_dim) - Updated Q1 embeddings
        """
        # Self-attention
        x = x + self.self_attn(self.ln1(x))

        # Cross-attention to text
        x = x + self.text_cross_attn(self.ln2(x), text_context, context_mask=text_mask)

        # Cross-attention to Q0 (KEY: hierarchical dependency)
        x = x + self.q0_cross_attn(self.ln3(x), q0_context)

        # Feedforward
        x = x + self.mlp(self.ln4(x))

        return x


class Q2DecoderBlock(nn.Module):
    """
    Transformer block for Q2 decoder (fine refinement, conditioned on Q0+Q1).

    Architecture:
        1. Causal self-attention (attend to previous Q2 tokens)
        2. Cross-attention to TEXT features
        3. Cross-attention to Q0+Q1 embeddings (KEY: hierarchical conditioning)
        4. Feed-forward network

    Q2 attends to both Q0 and Q1, creating the hierarchical dependency
    P(Q2 | Q0, Q1, text).
    """

    def __init__(self, embed_dim=1024, block_size=200, n_head=16,
                 drop_out_rate=0.1, fc_rate=4):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)  # For text cross-attention
        self.ln3 = nn.LayerNorm(embed_dim)  # For Q0+Q1 cross-attention
        self.ln4 = nn.LayerNorm(embed_dim)  # For FFN

        # Self-attention (causal)
        self.self_attn = CausalSelfAttention(
            embed_dim=embed_dim,
            block_size=block_size,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Cross-attention to text
        self.text_cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Cross-attention to Q0+Q1 (hierarchical conditioning)
        self.prev_quantizers_cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, text_context, prev_quantizers_context, text_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - Q2 token embeddings
            text_context: (B, S_text, text_dim) - Text features
            prev_quantizers_context: (B, T_prev, embed_dim) - Concatenated Q0+Q1 embeddings
            text_mask: (B, S_text) - Optional text mask

        Returns:
            (B, T, embed_dim) - Updated Q2 embeddings
        """
        # Self-attention
        x = x + self.self_attn(self.ln1(x))

        # Cross-attention to text
        x = x + self.text_cross_attn(self.ln2(x), text_context, context_mask=text_mask)

        # Cross-attention to Q0+Q1 (KEY: hierarchical dependency)
        x = x + self.prev_quantizers_cross_attn(self.ln3(x), prev_quantizers_context)

        # Feedforward
        x = x + self.mlp(self.ln4(x))

        return x


class HierarchicalRVQDecoder(nn.Module):
    """
    Complete GPT decoder for hierarchical RVQ prediction.

    For Q0: Standard GPT with cross-attention to text
    For Q1: GPT with cross-attention to BOTH text AND Q0
    For Q2: GPT with cross-attention to BOTH text AND Q0+Q1

    This matches the natural RVQ hierarchy where each quantizer refines
    the representation from previous stages.
    """

    def __init__(self, num_vq, embed_dim, block_size, num_layers, n_head,
                 dropout, quantizer_level=0):
        """
        Args:
            num_vq: Vocabulary size (codebook size, typically 512)
            embed_dim: Embedding dimension
            block_size: Maximum sequence length
            num_layers: Number of transformer blocks
            n_head: Number of attention heads
            dropout: Dropout probability
            quantizer_level: 0 for Q0 (independent), 1 for Q1, 2 for Q2
        """
        super().__init__()

        self.quantizer_level = quantizer_level
        self.block_size = block_size
        self.num_vq = num_vq
        self.eos_token_id = num_vq  # EOS token is index num_vq (e.g., 512 if num_vq=512)

        # Token embedding (num_vq + 1 to include EOS token)
        self.tok_emb = nn.Embedding(num_vq + 1, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.drop = nn.Dropout(dropout)

        # Transformer blocks (type depends on quantizer level)
        if quantizer_level == 0:
            # Q0: Independent decoder (like body decoder in SOKE)
            self.blocks = nn.ModuleList([
                Q0DecoderBlock(embed_dim, block_size, n_head, dropout)
                for _ in range(num_layers)
            ])
        elif quantizer_level == 1:
            # Q1: Conditioned on Q0
            self.blocks = nn.ModuleList([
                Q1DecoderBlock(embed_dim, block_size, n_head, dropout)
                for _ in range(num_layers)
            ])
        elif quantizer_level == 2:
            # Q2: Conditioned on Q0+Q1
            self.blocks = nn.ModuleList([
                Q2DecoderBlock(embed_dim, block_size, n_head, dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"quantizer_level must be 0, 1, or 2, got {quantizer_level}")

        # Output layer (num_vq + 1 to include EOS token prediction)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, text_context, text_mask=None, prev_quantizers_context=None):
        """
        Args:
            idx: (B, T) - Token indices for current quantizer
            text_context: (B, S, text_dim) - Text features
            text_mask: (B, S) - Optional text mask
            prev_quantizers_context: (B, T_prev, embed_dim) - Context from previous quantizers
                                     None for Q0, Q0 embeddings for Q1, Q0+Q1 for Q2

        Returns:
            logits: (B, T, num_vq) - Logits for next token prediction
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        # Embeddings
        token_embeddings = self.tok_emb(idx)  # (B, T, embed_dim)
        position_embeddings = self.pos_emb[:, :T, :]  # (1, T, embed_dim)
        x = self.drop(token_embeddings + position_embeddings)

        # Forward through transformer blocks
        if self.quantizer_level == 0:
            # Q0: Only text conditioning
            for block in self.blocks:
                x = block(x, text_context, text_mask)
        elif self.quantizer_level == 1:
            # Q1: Text + Q0 conditioning
            assert prev_quantizers_context is not None, "Q1 decoder requires Q0 context"
            for block in self.blocks:
                x = block(x, text_context, prev_quantizers_context, text_mask)
        elif self.quantizer_level == 2:
            # Q2: Text + Q0+Q1 conditioning
            assert prev_quantizers_context is not None, "Q2 decoder requires Q0+Q1 context"
            for block in self.blocks:
                x = block(x, text_context, prev_quantizers_context, text_mask)

        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def get_embeddings(self, idx):
        """
        Get token embeddings (used for quantizer-to-quantizer conditioning).

        Args:
            idx: (B, T) - Token indices

        Returns:
            embeddings: (B, T, embed_dim) - Token embeddings with positional encoding
        """
        B, T = idx.shape
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        return token_embeddings + position_embeddings
