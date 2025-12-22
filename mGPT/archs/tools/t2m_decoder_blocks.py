"""
T2M-GPT Style Decoder Blocks for Hierarchical RVQ-GPT

Implements the T2M-GPT architecture where text is PREPENDED to the motion sequence,
allowing all motion tokens to attend to text via standard causal self-attention.

Key difference from cross-attention approach:
- Text embedding is at position 0 of the sequence
- Motion tokens at positions 1, 2, 3, ...
- Causal attention naturally allows motion to attend to text
- Simpler architecture (no separate cross-attention layer)

Reference: T2M-GPT (Zhang et al., 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional


def PE1d_sincos(seq_length: int, dim: int) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional encoding.

    Args:
        seq_length: Maximum sequence length
        dim: Embedding dimension (must be even)

    Returns:
        pe: (seq_length, 1, dim) - Positional encoding
    """
    if dim % 2 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dim})")

    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)  # (seq_length, 1, dim)


class PositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding (fixed, not learned).

    From T2M-GPT: Uses fixed sinusoidal encoding for better generalization
    to different sequence lengths.
    """

    def __init__(self, seq_length: int, dim: int, dropout: float = 0.0, learnable: bool = False):
        super().__init__()
        self.embed = nn.Parameter(
            data=PE1d_sincos(seq_length, dim),
            requires_grad=learnable
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (B, T, D) - Input tensor

        Returns:
            (B, T, D) - Input with positional encoding added
        """
        seq_len = x.shape[1]
        # embed is (seq_length, 1, dim), expand to match batch
        x = x.permute(1, 0, 2) + self.embed[:seq_len].expand(-1, x.shape[0], -1)
        x = self.dropout(x.permute(1, 0, 2))
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention from T2M-GPT.

    Standard multi-head self-attention with causal masking.
    Text at position 0 can be attended to by all motion tokens.
    """

    def __init__(self, embed_dim: int, block_size: int, n_head: int, drop_out_rate: float = 0.1):
        super().__init__()
        assert embed_dim % n_head == 0, f"embed_dim ({embed_dim}) must be divisible by n_head ({n_head})"

        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        # Q, K, V projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) - Input tensor

        Returns:
            (B, T, C) - Output tensor
        """
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values
        y = att @ v  # (B, nh, T, hs)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_drop(self.proj(y))

        return y


class T2MTransformerBlock(nn.Module):
    """
    Transformer block from T2M-GPT.

    Pre-LN architecture:
        x = x + SelfAttn(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, embed_dim: int, block_size: int, n_head: int,
                 drop_out_rate: float = 0.1, fc_rate: int = 4):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            block_size=block_size,
            n_head=n_head,
            drop_out_rate=drop_out_rate
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class T2MTransformerBase(nn.Module):
    """
    Transformer base from T2M-GPT.

    Handles:
    - Token embedding
    - Text conditioning (prepended to sequence)
    - Positional encoding
    - N transformer blocks
    """

    def __init__(self, num_vq: int, embed_dim: int, text_dim: int,
                 block_size: int, num_layers: int, n_head: int,
                 drop_out_rate: float = 0.1, fc_rate: int = 4):
        super().__init__()

        self.block_size = block_size

        # Token embedding: num_vq codes + EOS + PAD
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)

        # Text projection: text_dim -> embed_dim
        self.cond_emb = nn.Linear(text_dim, embed_dim)

        # Positional embedding (sinusoidal)
        # block_size + 1 to account for prepended text token
        self.pos_embed = PositionEmbedding(block_size + 1, embed_dim, dropout=0.0, learnable=False)

        # Dropout
        self.drop = nn.Dropout(drop_out_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            T2MTransformerBlock(embed_dim, block_size + 1, n_head, drop_out_rate, fc_rate)
            for _ in range(num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) - Motion token indices, or empty tensor for first step
            text_feature: (B, text_dim) - Text features from CLIP

        Returns:
            (B, T+1, embed_dim) - Hidden states (position 0 is text, 1: are motion)
        """
        # Handle empty sequence (first generation step)
        if idx.numel() == 0 or (idx.dim() == 2 and idx.shape[1] == 0):
            # Only text embedding
            token_embeddings = self.cond_emb(text_feature).unsqueeze(1)  # (B, 1, embed_dim)
        else:
            B, T = idx.size()
            assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

            # Embed motion tokens
            token_embeddings = self.tok_emb(idx)  # (B, T, embed_dim)

            # Prepend text embedding
            text_emb = self.cond_emb(text_feature).unsqueeze(1)  # (B, 1, embed_dim)
            token_embeddings = torch.cat([text_emb, token_embeddings], dim=1)  # (B, T+1, embed_dim)

        # Add positional encoding
        x = self.pos_embed(token_embeddings)

        # Pass through transformer blocks
        x = self.blocks(x)

        return x


class T2MTransformerHead(nn.Module):
    """
    Transformer head from T2M-GPT.

    Additional transformer blocks + final classification head.
    """

    def __init__(self, num_vq: int, embed_dim: int, block_size: int,
                 num_layers: int, n_head: int, drop_out_rate: float = 0.1, fc_rate: int = 4):
        super().__init__()

        # Additional transformer blocks (can be 0)
        self.blocks = nn.Sequential(*[
            T2MTransformerBlock(embed_dim, block_size + 1, n_head, drop_out_rate, fc_rate)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Classification head: predict num_vq codes + EOS token
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T+1, embed_dim) - Hidden states from base

        Returns:
            (B, T+1, num_vq+1) - Logits for each position
        """
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class T2MStyleQ0Decoder(nn.Module):
    """
    Complete T2M-GPT style decoder for Q0 (coarse quantizer).

    Architecture:
        Text Feature + Motion Tokens
              ↓
        [Prepend text to sequence]
              ↓
        [Sinusoidal Positional Encoding]
              ↓
        [TransformerBase] (N layers)
              ↓
        [TransformerHead] (M layers + classification)
              ↓
        Logits (B, T, num_vq+1)

    Key innovation: Text is prepended to sequence, not cross-attended.
    This allows motion tokens to attend to text via causal self-attention.
    """

    def __init__(self, num_vq: int, embed_dim: int, text_dim: int,
                 block_size: int, num_layers: int, num_layers_head: int = 0,
                 n_head: int = 16, drop_out_rate: float = 0.1, fc_rate: int = 4):
        """
        Args:
            num_vq: Codebook size (e.g., 512)
            embed_dim: Embedding dimension
            text_dim: Text feature dimension (512 for CLIP)
            block_size: Maximum motion sequence length
            num_layers: Number of layers in base transformer
            num_layers_head: Number of additional layers in head (default 0)
            n_head: Number of attention heads
            drop_out_rate: Dropout probability
            fc_rate: FFN expansion factor
        """
        super().__init__()

        self.num_vq = num_vq
        self.block_size = block_size
        self.eos_token_id = num_vq  # EOS token index

        # Base transformer (embedding + N layers)
        self.trans_base = T2MTransformerBase(
            num_vq=num_vq,
            embed_dim=embed_dim,
            text_dim=text_dim,
            block_size=block_size,
            num_layers=num_layers,
            n_head=n_head,
            drop_out_rate=drop_out_rate,
            fc_rate=fc_rate
        )

        # Head transformer (M layers + classification)
        self.trans_head = T2MTransformerHead(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=num_layers_head,
            n_head=n_head,
            drop_out_rate=drop_out_rate,
            fc_rate=fc_rate
        )

    def forward(self, idx: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            idx: (B, T) - Motion token indices
            text_feature: (B, text_dim) - Text features from CLIP

        Returns:
            logits: (B, T, num_vq+1) - Logits for next token prediction
                    Note: Position 0 (text) logits are excluded
        """
        # Get hidden states (B, T+1, embed_dim)
        feat = self.trans_base(idx, text_feature)

        # Get logits (B, T+1, num_vq+1)
        logits = self.trans_head(feat)

        # Exclude text position (position 0), return only motion positions
        # logits[:, 0, :] predicts first motion token
        # logits[:, 1, :] predicts second motion token, etc.
        return logits  # Keep all positions for now, caller will handle

    def get_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get token embeddings (used for conditioning Q1/Q2 decoders).

        Args:
            idx: (B, T) - Token indices

        Returns:
            embeddings: (B, T, embed_dim) - Token embeddings with positional encoding
        """
        if idx.numel() == 0 or (idx.dim() == 2 and idx.shape[1] == 0):
            return torch.zeros((idx.shape[0], 0, self.trans_base.tok_emb.embedding_dim),
                             device=idx.device)

        token_embeddings = self.trans_base.tok_emb(idx)  # (B, T, embed_dim)

        # Add positional encoding (without text position offset)
        # Use positions 1, 2, 3, ... to match the forward pass where text is at position 0
        B, T, D = token_embeddings.shape
        pos_embed = self.trans_base.pos_embed.embed[1:T+1].transpose(0, 1).expand(B, -1, -1)

        return token_embeddings + pos_embed

    @torch.no_grad()
    def sample(self, text_feature: torch.Tensor, max_len: Optional[int] = None,
               temperature: float = 1.0, do_sample: bool = True,
               top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Autoregressive sampling.

        Args:
            text_feature: (B, text_dim) - Text features
            max_len: Maximum generation length (default: block_size)
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            xs: (B, T) - Generated token indices (variable length, stops at EOS)
        """
        if max_len is None:
            max_len = self.block_size

        B = text_feature.shape[0]
        device = text_feature.device

        xs = None

        for k in range(max_len):
            # Prepare input
            if k == 0:
                idx = torch.zeros((B, 0), dtype=torch.long, device=device)
            else:
                idx = xs

            # Forward pass
            logits = self.forward(idx, text_feature)  # (B, k+1, num_vq+1)
            logits = logits[:, -1, :] / temperature  # (B, num_vq+1)

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Check for EOS
            if (idx_next == self.eos_token_id).all():
                break

            # Append to sequence
            if xs is None:
                xs = idx_next
            else:
                xs = torch.cat([xs, idx_next], dim=1)

            # Stop if max length reached
            if k == max_len - 1:
                break

        return xs if xs is not None else torch.zeros((B, 0), dtype=torch.long, device=device)
