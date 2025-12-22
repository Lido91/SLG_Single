"""
Text-to-Motion Transformer Architecture
Adapted from UniMuMo's mm_lm.py for MotionGPT

Key Features:
- Per-codebook embeddings (6 codebooks x 512 vocab)
- Per-codebook output heads
- T5 cross-attention conditioning
- Classifier-Free Guidance (CFG) support
- Delayed codebook pattern for RVQ
"""

from dataclasses import dataclass
from functools import partial
import math
import typing as tp
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """Get initialization function for layers."""
    std = 1 / math.sqrt(input_dim)
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(
    m: nn.Module,
    method: str,
    init_depth: tp.Optional[int] = None,
    zero_bias_init: bool = False
):
    """Initialize a layer with the specified method."""
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Embedding with optional separate learning rate."""
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    """Output from the language model."""
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor    # [B, K, T]


class LayerScale(nn.Module):
    """Layer scale from Touvron et al 2021."""
    def __init__(self, channels: int, init: float = 1e-4, device=None, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((channels,), init, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return self.scale * x


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape [B, T, C]."""
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class MultiheadCrossAttention(nn.Module):
    """Cross-attention module for T5 conditioning."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_k, D]
            attn_mask: [B, T_q, T_k] or [B, 1, T_k] - attention mask (0 = attend, -inf = ignore)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_q, T_k]

        if attn_mask is not None:
            # Expand mask for multi-head: [B, 1, T_k] -> [B, 1, 1, T_k] or [B, T_q, T_k] -> [B, 1, T_q, T_k]
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, T_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        out = self.out_proj(out)

        return out


class TransformerLayer(nn.Module):
    """Transformer layer with cross-attention support."""
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, cross_attention: bool = True,
                 layer_scale: tp.Optional[float] = None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, **factory_kwargs
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (for T5 conditioning)
        self.cross_attention: tp.Optional[nn.Module] = None
        if cross_attention:
            self.cross_attention = MultiheadCrossAttention(
                d_model, num_heads, dropout=dropout, **factory_kwargs
            )
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
            self.dropout_cross = nn.Dropout(dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Layer scale
        if layer_scale is not None:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)
            if cross_attention:
                self.layer_scale_cross = LayerScale(d_model, layer_scale, **factory_kwargs)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
            if cross_attention:
                self.layer_scale_cross = nn.Identity()

    def forward(self, x: torch.Tensor,
                src_mask: tp.Optional[torch.Tensor] = None,
                cross_attention_src: tp.Optional[torch.Tensor] = None,
                cross_attn_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask, need_weights=False)
        x = x + self.layer_scale_1(self.dropout1(attn_out))

        # Cross-attention with T5 embeddings
        if self.cross_attention is not None and cross_attention_src is not None:
            x_norm = self.norm_cross(x)
            cross_out = self.cross_attention(x_norm, cross_attention_src, cross_attention_src,
                                             attn_mask=cross_attn_mask)
            x = x + self.layer_scale_cross(self.dropout_cross(cross_out))

        # Feedforward (pre-norm)
        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(x_norm))))
        x = x + self.layer_scale_2(self.dropout3(ff_out))

        return x


class MotionTransformer(nn.Module):
    """Transformer backbone for motion generation."""
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 cross_attention: bool = True, layer_scale: tp.Optional[float] = None,
                 max_period: float = 10000, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                cross_attention=cross_attention,
                layer_scale=layer_scale,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor,
                src_mask: tp.Optional[torch.Tensor] = None,
                cross_attention_src: tp.Optional[torch.Tensor] = None,
                cross_attn_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # Add positional encoding
        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
        x = x + pos_emb

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, cross_attention_src=cross_attention_src,
                     cross_attn_mask=cross_attn_mask)

        x = self.out_norm(x)
        return x


class MotionLM(nn.Module):
    """
    Motion Language Model with per-codebook embeddings and output heads.

    Adapted from UniMuMo for motion-only generation with:
    - 6 codebooks (MotionGPT RVQ)
    - 512 vocabulary size per codebook
    - T5 cross-attention conditioning
    - Classifier-Free Guidance support
    - Delayed codebook pattern
    """
    def __init__(
        self,
        n_q: int = 6,                    # Number of codebooks
        card: int = 512,                 # Codebook vocabulary size
        dim: int = 768,                  # Transformer hidden dimension
        num_heads: int = 12,             # Attention heads
        num_layers: int = 12,            # Transformer layers
        hidden_scale: int = 4,           # FFN hidden dim = dim * hidden_scale
        dropout: float = 0.1,
        cross_attention: bool = True,
        layer_scale: tp.Optional[float] = None,
        emb_lr: tp.Optional[float] = None,
        bias_proj: bool = True,
        weight_init: tp.Optional[str] = None,
        depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False,
        cfg_dropout: float = 0.1,        # CFG dropout probability during training
        cfg_coef: float = 3.0,           # CFG coefficient at inference
        max_period: float = 10000,
        device=None,
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_q = n_q
        self.card = card
        self.dim = dim
        self.cfg_coef = cfg_coef
        self.cfg_dropout = cfg_dropout

        embed_dim = card + 1  # +1 for special token

        # Per-codebook embedding tables
        self.emb = nn.ModuleList([
            ScaledEmbedding(embed_dim, dim, lr=emb_lr, **factory_kwargs)
            for _ in range(n_q)
        ])

        # Transformer backbone
        self.transformer = MotionTransformer(
            d_model=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=int(hidden_scale * dim),
            dropout=dropout,
            cross_attention=cross_attention,
            layer_scale=layer_scale,
            max_period=max_period,
            **factory_kwargs
        )

        # Per-codebook output heads
        self.linears = nn.ModuleList([
            nn.Linear(dim, card, bias=bias_proj, **factory_kwargs)
            for _ in range(n_q)
        ])

        # Initialize weights
        if weight_init is not None:
            self._init_weights(weight_init, depthwise_init, zero_bias_init)

    def _init_weights(self, weight_init: str, depthwise_init: tp.Optional[str],
                      zero_bias_init: bool):
        """Initialize weights."""
        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None,
                      zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth,
                            zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None,
                      zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        """Special token ID (used for padding in delayed pattern)."""
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(
        self,
        codes: torch.LongTensor,
        condition: tp.Optional[torch.Tensor] = None,
        condition_mask: tp.Optional[torch.Tensor] = None,
        src_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the motion language model.

        Args:
            codes: Motion codes [B, K, T] where K is num_codebooks
            condition: T5 condition embeddings [B, L, D] for cross-attention
            condition_mask: Condition attention mask [B, L]
            src_mask: Self-attention mask [T, T]

        Returns:
            logits: [B, K, T, card] logits for each codebook
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

        # Sum embeddings from all codebooks
        input_ = sum([self.emb[k](codes[:, k]) for k in range(K)])  # [B, T, dim]

        # Prepare cross-attention mask if condition is provided
        cross_attn_mask = None
        if condition is not None and condition_mask is not None:
            # Check if any samples have valid conditions (non-zero mask)
            # When CFG dropout nullifies all masks, we should skip cross-attention
            has_valid_condition = condition_mask.sum(dim=-1, keepdim=True) > 0  # [B, 1]

            # Convert mask to attention format: True -> 0, False -> -inf
            # But for samples with no valid conditions, use 0 everywhere to avoid nan
            cross_attn_mask = torch.where(
                condition_mask.unsqueeze(1).bool(),  # [B, 1, L]
                torch.zeros_like(condition_mask.unsqueeze(1).float()),
                torch.full_like(condition_mask.unsqueeze(1).float(), float('-inf'))
            )

            # For samples with no valid conditions, set mask to 0 (will be ignored anyway)
            # This prevents nan from softmax when all values are -inf
            cross_attn_mask = torch.where(
                has_valid_condition.unsqueeze(-1),  # [B, 1, 1]
                cross_attn_mask,
                torch.zeros_like(cross_attn_mask)
            )

        # Apply transformer
        out = self.transformer(
            input_,
            src_mask=src_mask,
            cross_attention_src=condition,
            cross_attn_mask=cross_attn_mask
        )  # [B, T, dim]

        # Per-codebook output projections
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, T, card]

        return logits

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        mask = torch.where(mask, float('-inf'), 0.0)
        return mask

    @torch.no_grad()
    def generate(
        self,
        condition: tp.Optional[torch.Tensor] = None,
        condition_mask: tp.Optional[torch.Tensor] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        device: tp.Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """
        Generate motion codes autoregressively.

        Args:
            condition: T5 embeddings [B, L, D] or [2B, L, D] for CFG
            condition_mask: Attention mask [B, L] or [2B, L]
            max_gen_len: Maximum generation length
            use_sampling: Whether to sample or use greedy decoding
            temp: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            cfg_coef: CFG coefficient (overrides self.cfg_coef)

        Returns:
            codes: Generated motion codes [B, K, T]
        """
        assert not self.training, "Generation should not be used in training mode"

        if device is None:
            device = next(self.parameters()).device

        cfg_coef = cfg_coef if cfg_coef is not None else self.cfg_coef

        # Determine batch size
        if condition is not None:
            B = condition.shape[0]
            # For CFG: condition should be [B + B, L, D] = [cond, uncond]
            if cfg_coef > 1.0:
                B = B // 2
        else:
            B = 1

        K = self.num_codebooks

        # Initialize with special token
        gen_codes = torch.full((B, K, 1), self.special_token_id, dtype=torch.long, device=device)

        for t in tqdm(range(max_gen_len), desc="Generating motion"):
            # Forward pass
            if condition is not None and cfg_coef > 1.0:
                # CFG: duplicate current sequence
                curr_codes = torch.cat([gen_codes, gen_codes], dim=0)
            else:
                curr_codes = gen_codes

            # Get logits
            logits = self.forward(
                codes=curr_codes,
                condition=condition,
                condition_mask=condition_mask,
                src_mask=self.get_causal_mask(curr_codes.shape[-1], device)
            )  # [B or 2B, K, T, card]

            # Get logits for last position
            logits = logits[:, :, -1, :]  # [B or 2B, K, card]

            # Apply CFG
            if condition is not None and cfg_coef > 1.0:
                cond_logits, uncond_logits = logits.split(B, dim=0)
                logits = uncond_logits + cfg_coef * (cond_logits - uncond_logits)

            # Sample next token
            if use_sampling and temp > 0:
                logits = logits / temp
                if top_k > 0:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, :, -1:]] = float('-inf')
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
                    sorted_indices_to_remove[:, :, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.view(B * K, -1), num_samples=1)
                next_token = next_token.view(B, K, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            gen_codes = torch.cat([gen_codes, next_token], dim=-1)

        # Remove initial special token
        gen_codes = gen_codes[:, :, 1:]

        return gen_codes


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample from top-k distribution."""
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs /= probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    return next_token.view(*probs.shape[:-1], 1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample from nucleus (top-p) distribution."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs /= probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    return next_token.view(*probs.shape[:-1], 1)
