"""
Causal self-attention for GPT-style transformers
Adapted from T2M-GPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention
    Prevents attending to future positions (autoregressive)
    """

    def __init__(self, embed_dim=1024, block_size=51, n_head=16, drop_out_rate=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            block_size: Maximum sequence length (for causal mask)
            n_head: Number of attention heads
            drop_out_rate: Dropout probability
        """
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.embed_dim = embed_dim

        # Q, K, V projections for all heads (but in a batch)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (lower triangular)
        # Register as buffer so it moves to device with model
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        #     0  1  2  3
        # 0 [[1, 0, 0, 0],   # Token 0 can only see itself
        # 1  [1, 1, 0, 0],   # Token 1 can see 0, 1
        # 2  [1, 1, 1, 0],   # Token 2 can see 0, 1, 2
        # 3  [1, 1, 1, 1]]   # Token 3 can see all previous

    def forward(self, x):
        """
        Args:
            x: (B, T, embed_dim) input embeddings

        Returns:
            (B, T, embed_dim) output after attention
        """
        B, T, C = x.size()  # Batch, Time, Channels

        # Compute Q, K, V for all heads in batch
        q = self.query(x)  # (B, T, embed_dim)
        k = self.key(x)    # (B, T, embed_dim)
        v = self.value(x)  # (B, T, embed_dim)

        # Reshape to (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Scaled dot-product attention
        # att[i,j] = similarity between query_i and key_j
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)

        # Apply causal mask (prevent attending to future)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Positions with 0 in mask → -∞ → softmax → 0 (no attention)

        # Normalize to probability distribution
        att = F.softmax(att, dim=-1)  # (B, n_head, T, T)

        # Dropout on attention weights
        att = self.attn_drop(att)

        # Weighted sum of values
        y = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

        # Output projection
        y = self.resid_drop(self.proj(y))

        return y
