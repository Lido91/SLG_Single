"""
Cross-attention module for GPT decoder
Allows attending to external context (e.g., CLIP text embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention
    Query from decoder, Key/Value from encoder
    """

    def __init__(self, embed_dim=1024, n_head=16, drop_out_rate=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            n_head: Number of attention heads
            drop_out_rate: Dropout probability
        """
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.embed_dim = embed_dim

        # Q from decoder, K/V from encoder
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context, context_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - decoder hidden states (for Query)
            context: (B, S, embed_dim) - encoder outputs (for Key/Value)
                     S can be 1 (CLIP single embedding) or variable length (BERT words)
            context_mask: (B, S) - attention mask for context (1=valid, 0=padding)
                         Optional, used for variable-length sequences (BERT)

        Returns:
            (B, T, embed_dim) output after cross-attention
        """
        B, T, C = x.size()  # Batch, Target length, Channels
        S = context.size(1)  # Source length

        # Query from decoder
        q = self.query(x)  # (B, T, embed_dim)

        # Key, Value from encoder context
        k = self.key(context)    # (B, S, embed_dim)
        v = self.value(context)  # (B, S, embed_dim)

        # Reshape to (B, n_head, T/S, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, S, head_dim)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, S, head_dim)

        # Scaled dot-product attention
        # att[i,j] = similarity between decoder query_i and encoder key_j
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))  # (B, n_head, T, S)

        # Apply attention mask if provided (for padded sequences)
        if context_mask is not None:
            # context_mask: (B, S) where 1=valid, 0=padding
            # Reshape to (B, 1, 1, S) for broadcasting
            context_mask = context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            # Set attention scores to -inf for padding positions
            att = att.masked_fill(context_mask == 0, float('-inf'))

        # No causal mask needed for cross-attention!
        # Decoder can attend to all encoder positions
        att = F.softmax(att, dim=-1)  # (B, n_head, T, S)

        # Handle NaN from softmax(-inf) if entire row is masked
        att = torch.nan_to_num(att, nan=0.0)

        att = self.attn_drop(att)

        # Apply attention to values
        y = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

        # Output projection
        y = self.proj(y)
        y = self.resid_drop(y)

        return y
