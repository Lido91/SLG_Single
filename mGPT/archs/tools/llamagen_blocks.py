"""
LlamaGen-style Transformer Blocks for Q0 Decoder

Adapted from LlamaGen (https://github.com/FoundationVision/LlamaGen):
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU FFN
- Grouped Query Attention (GQA) with 1D RoPE
- Prefix concatenation for conditioning (no cross-attention)
- KV Cache for fast autoregressive inference

Reference: /data/hwu/workspace/LlamaGen/autoregressive/models/gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def find_multiple(n: int, k: int) -> int:
    """Round up n to the nearest multiple of k."""
    if n % k == 0:
        return n
    return n + k - (n % k)


# ===========================================================================
#  RMSNorm
# ===========================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, faster than LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ===========================================================================
#  SwiGLU Feed-Forward
# ===========================================================================
class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN: w2(SiLU(w1(x)) * w3(x)), hidden_dim = 4*dim*2/3 rounded to multiple of 256."""
    def __init__(self, dim: int, dropout: float = 0.0, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ===========================================================================
#  DropPath (Stochastic Depth)
# ===========================================================================
class DropPath(nn.Module):
    """Stochastic depth per sample (randomly drops entire residual paths during training)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ===========================================================================
#  KV Cache
# ===========================================================================
class KVCache(nn.Module):
    """Stores past K/V for each layer, updated incrementally during inference."""
    def __init__(self, max_batch_size: int, max_seq_length: int, n_kv_head: int, head_dim: int, dtype: torch.dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_kv_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        """
        Args:
            input_pos: (S,) - positions to update
            k_val: (B, n_kv_head, S, head_dim)
            v_val: (B, n_kv_head, S, head_dim)
        Returns:
            k_out, v_out: full cache tensors (B, n_kv_head, max_seq_length, head_dim)
        """
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


# ===========================================================================
#  1D RoPE
# ===========================================================================
def precompute_freqs_cis_1d(seq_len: int, n_elem: int, base: float = 10000.0, cls_token_num: int = 0):
    """
    Precompute 1D RoPE frequencies for sequential (non-spatial) data.

    Conditioning prefix tokens get zero rotation (positions filled with zeros).
    Motion tokens get 1D sequential positions (0, 1, 2, ..., seq_len-1).

    Args:
        seq_len: Maximum motion sequence length
        n_elem: head_dim (dimension per head)
        base: RoPE base frequency
        cls_token_num: Number of conditioning prefix tokens (zero-padded)

    Returns:
        freqs_cis: (cls_token_num + seq_len, n_elem // 2, 2) - [cos, sin] pairs
    """
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (seq_len, head_dim // 2, 2)
    # Prepend zeros for conditioning prefix (no rotation for prefix tokens)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num + seq_len, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to Q or K tensors.

    Args:
        x: (B, seq_len, n_head, head_dim)
        freqs_cis: (seq_len, head_dim // 2, 2)

    Returns:
        x_out: (B, seq_len, n_head, head_dim) with RoPE applied
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (B, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    x_out = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out = x_out.flatten(3)
    return x_out.type_as(x)


# ===========================================================================
#  LlamaGen Attention (GQA + RoPE + KV Cache)
# ===========================================================================
class LlamaGenAttention(nn.Module):
    """
    Grouped Query Attention with RoPE and optional KV cache.

    Uses fused QKV projection, 1D RoPE, and F.scaled_dot_product_attention.
    """
    def __init__(self, dim: int, n_head: int, n_kv_head: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # Fused QKV projection
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, dim)
            freqs_cis: (S, head_dim//2, 2) - RoPE frequencies for current positions
            input_pos: (S,) - position indices (for KV cache)
            mask: (B, 1, S, total_len) or None - attention mask
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # Transpose to (B, n_head, S, head_dim)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # KV cache update
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        # GQA: repeat KV heads to match Q heads
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Scaled dot-product attention
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=(mask is None and self.kv_cache is None),
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        return output


# ===========================================================================
#  LlamaGen Transformer Block
# ===========================================================================
class LlamaGenBlock(nn.Module):
    """
    Single LlamaGen transformer block:
        x -> RMSNorm -> LlamaGenAttention -> +residual(DropPath)
          -> RMSNorm -> SwiGLUFeedForward -> +residual(DropPath)
    """
    def __init__(self, dim: int, n_head: int, n_kv_head: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.attention = LlamaGenAttention(dim, n_head, n_kv_head, dropout)
        self.feed_forward = SwiGLUFeedForward(dim, dropout)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, input_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


# ===========================================================================
#  LlamaGen Q0 Decoder
# ===========================================================================
class LlamaGenQ0Decoder(nn.Module):
    """
    Complete Q0 decoder using LlamaGen-style architecture.

    Uses prefix concatenation for conditioning (no cross-attention):
    - Conditioning embeddings (text/speech) are prepended to the motion token sequence
    - Causal self-attention over the combined [cond_prefix | motion_tokens] sequence
    - RoPE: zeros for conditioning prefix, 1D sequential for motion tokens

    Interface:
    - forward():           Training forward pass
    - get_embeddings():    Returns tok_emb(idx) for Q1/Q2 cross-attention conditioning
    - setup_caches():      Allocate KV cache per layer
    - clear_caches():      Clear KV caches
    - generate_prefill():  Prefill KV cache with conditioning tokens
    - generate_one_token(): Decode one motion token using KV cache
    """

    def __init__(
        self,
        num_vq: int = 512,
        embed_dim: int = 1024,
        block_size: int = 200,
        num_layers: int = 9,
        n_head: int = 16,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            num_vq: Codebook size (e.g. 512 or 2048)
            embed_dim: Token embedding dimension
            block_size: Maximum motion sequence length
            num_layers: Number of transformer blocks
            n_head: Number of query attention heads
            n_kv_head: Number of KV heads for GQA (default = n_head = standard MHA)
            dropout: Dropout probability
            drop_path_rate: Stochastic depth rate (linearly increases per layer)
        """
        super().__init__()

        if n_kv_head is None:
            n_kv_head = n_head

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = embed_dim // n_head
        self.eos_token_id = num_vq

        # Token embedding (num_vq + 1 to include EOS token)
        self.tok_emb = nn.Embedding(num_vq + 1, embed_dim)
        self.tok_dropout = nn.Dropout(dropout)

        # Transformer blocks with linearly increasing drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            LlamaGenBlock(embed_dim, n_head, n_kv_head, dropout, dpr[i])
            for i in range(num_layers)
        ])

        # Output
        self.norm = RMSNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_vq + 1, bias=False)

        # RoPE freqs will be computed lazily based on actual cls_token_num
        self._freqs_cis = None
        self._freqs_cls_token_num = -1
        self._freqs_max_seq_len = -1

        # KV cache state
        self.max_batch_size = -1
        self.max_seq_length = -1

    def _get_freqs_cis(self, cls_token_num: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or recompute RoPE frequencies."""
        if self._freqs_cis is None or self._freqs_cls_token_num != cls_token_num or self._freqs_max_seq_len < seq_len:
            self._freqs_cis = precompute_freqs_cis_1d(
                seq_len=max(seq_len, self.block_size),
                n_elem=self.head_dim,
                cls_token_num=cls_token_num,
            )
            self._freqs_cls_token_num = cls_token_num
            self._freqs_max_seq_len = max(seq_len, self.block_size)
        return self._freqs_cis.to(device)

    def _build_training_mask(
        self,
        B: int,
        cls_token_num: int,
        motion_len: int,
        cond_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build combined attention mask for training: causal + conditioning padding mask.

        The full sequence is [cond_prefix (C tokens) | motion_tokens (T tokens)].
        - Causal: each position can attend to itself and all previous positions
        - Padding: no position should attend to padding positions in the conditioning prefix

        Args:
            B: batch size
            cls_token_num: number of conditioning prefix tokens (C)
            motion_len: number of motion tokens (T)
            cond_mask: (B, C) - 1 for valid, 0 for padding. None means all valid.
            device: torch device
            dtype: torch dtype for the mask

        Returns:
            mask: (B, 1, C+T, C+T) - attention mask (0 = attend, -inf = block)
        """
        total_len = cls_token_num + motion_len

        # Start with causal mask
        causal_mask = torch.tril(torch.ones(total_len, total_len, device=device, dtype=dtype))

        if cond_mask is not None:
            # cond_mask: (B, C), 1=valid, 0=padding
            # We need to block attention to padding columns in the prefix
            # Expand to (B, 1, 1, C) then broadcast to block padding columns
            # For each row, mask out columns corresponding to padding in prefix
            pad_mask = torch.ones(B, 1, 1, total_len, device=device, dtype=dtype)
            # cond_mask is (B, C): set padding positions to 0
            pad_mask[:, :, :, :cls_token_num] = cond_mask.unsqueeze(1).unsqueeze(2).to(dtype)  # (B, 1, 1, C)
            # Combine: causal AND padding
            # causal_mask: (total_len, total_len) -> (1, 1, total_len, total_len)
            mask = causal_mask.unsqueeze(0).unsqueeze(0) * pad_mask  # (B, 1, total_len, total_len)
        else:
            mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len)

        # Convert to additive mask: 0 -> 0.0, 1 stays, but we need 0 -> -inf for blocked
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def forward(
        self,
        idx: torch.Tensor,
        cond_context: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass with prefix concatenation.

        Args:
            idx: (B, T) - Motion token indices
            cond_context: (B, C, embed_dim) - Conditioning embeddings (already projected to embed_dim)
            cond_mask: (B, C) - Conditioning mask (1=valid, 0=padding). None for CLIP (no padding).

        Returns:
            logits: (B, T, num_vq+1) - Logits for motion token positions only
        """
        B, T = idx.shape
        C = cond_context.shape[1]  # Number of conditioning prefix tokens

        # Token embeddings
        token_embeddings = self.tok_emb(idx)  # (B, T, embed_dim)

        # Concatenate: [cond_prefix | motion_tokens]
        h = torch.cat([cond_context, token_embeddings], dim=1)  # (B, C+T, embed_dim)
        h = self.tok_dropout(h)

        # RoPE frequencies
        freqs_cis = self._get_freqs_cis(C, T, h.device)
        freqs_cis = freqs_cis[:C + T]  # (C+T, head_dim//2, 2)

        # Build attention mask
        mask = self._build_training_mask(B, C, T, cond_mask, h.device, h.dtype)

        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos=None, mask=mask)

        # Output: only for motion token positions
        h = self.norm(h)
        h_motion = h[:, C:, :]  # (B, T, embed_dim) - strip conditioning prefix
        logits = self.output(h_motion)  # (B, T, num_vq+1)

        return logits

    def get_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get token embeddings for Q1/Q2 cross-attention conditioning.

        Returns raw token embeddings (no positional encoding, since RoPE is internal
        and Q1/Q2 have their own positional encoding).

        Args:
            idx: (B, T) - Token indices

        Returns:
            embeddings: (B, T, embed_dim)
        """
        return self.tok_emb(idx)

    # ===================================================================
    #  KV Cache methods for inference
    # ===================================================================

    def setup_caches(self, max_batch_size: int, max_seq_length: int, dtype: torch.dtype):
        """Allocate KV cache per layer for inference."""
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        head_dim = self.head_dim
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.n_kv_head, head_dim, dtype
            )
        # Precompute causal mask for KV cache inference
        device = next(self.parameters()).device
        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))
        self._kv_causal_mask = causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1).to(device)

    def clear_caches(self):
        """Clear all KV caches (reset to None)."""
        for layer in self.layers:
            layer.attention.kv_cache = None
        self._kv_causal_mask = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def generate_prefill(
        self,
        cond_context: torch.Tensor,
        input_pos: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill KV cache with conditioning tokens and return first motion token logits.

        Args:
            cond_context: (B, C, embed_dim) - Conditioning prefix embeddings
            input_pos: (C,) - Position indices for conditioning tokens (0, 1, ..., C-1)
            cond_mask: (B, C) - Conditioning mask (1=valid, 0=pad). None if no padding.

        Returns:
            logits: (B, 1, num_vq+1) - Logits for the first motion token (predicted from last cond token)
        """
        B, C, _ = cond_context.shape

        h = self.tok_dropout(cond_context)

        # RoPE: all zeros for conditioning prefix
        freqs_cis = self._get_freqs_cis(C, self.block_size, h.device)
        freqs_cis = freqs_cis[input_pos]  # (C, head_dim//2, 2) — all zeros

        # Attention mask for prefill: causal within prefix + padding mask
        # Use the precomputed causal mask
        mask = self._kv_causal_mask[:B, None, input_pos]  # (B, 1, C, max_seq_len)
        if cond_mask is not None:
            # Block attention to padding positions within the prefix
            pad_column_mask = torch.ones(B, 1, 1, self.max_seq_length, device=h.device, dtype=torch.bool)
            pad_column_mask[:, :, :, :C] = cond_mask.unsqueeze(1).unsqueeze(2).bool()
            mask = mask & pad_column_mask

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        # Take logits from the last conditioning token position
        logits = self.output(h[:, -1:, :])  # (B, 1, num_vq+1)
        return logits

    def generate_one_token(
        self,
        idx: torch.Tensor,
        input_pos: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
        cls_token_num: int = 0,
    ) -> torch.Tensor:
        """
        Decode one motion token using KV cache.

        Args:
            idx: (B, 1) or (B,) - Previous motion token index
            input_pos: (1,) - Position index for this token
            cond_mask: (B, C) - Conditioning mask for prefix padding. None if no padding.
            cls_token_num: Number of conditioning prefix tokens (for padding mask)

        Returns:
            logits: (B, 1, num_vq+1) - Logits for next motion token
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)  # (B,) -> (B, 1)
        B = idx.shape[0]

        token_embeddings = self.tok_emb(idx)  # (B, 1, embed_dim)
        h = self.tok_dropout(token_embeddings)

        # RoPE for this position
        freqs_cis = self._get_freqs_cis(cls_token_num, self.block_size, h.device)
        freqs_cis = freqs_cis[input_pos]  # (1, head_dim//2, 2)

        # Mask: attend to all cached positions up to current
        mask = self._kv_causal_mask[:B, None, input_pos]  # (B, 1, 1, max_seq_len)
        if cond_mask is not None:
            # Block attention to padding positions in prefix
            pad_column_mask = torch.ones(B, 1, 1, self.max_seq_length, device=h.device, dtype=torch.bool)
            pad_column_mask[:, :, :, :cls_token_num] = cond_mask.unsqueeze(1).unsqueeze(2).bool()
            mask = mask & pad_column_mask

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        h = self.norm(h)
        logits = self.output(h)  # (B, 1, num_vq+1)
        return logits
