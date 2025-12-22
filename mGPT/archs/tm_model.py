"""
Text-to-Motion Model
Complete model integrating transformer, conditioner, and codebook patterns

Key Features:
- Per-codebook embeddings and output heads (6 codebooks x 512 vocab)
- T5 cross-attention conditioning
- Delayed codebook pattern for RVQ
- Classifier-Free Guidance (CFG) support
- Training and inference modes
"""

from dataclasses import dataclass
import typing as tp
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from .tm_transformer import MotionLM, ScaledEmbedding, create_sin_embedding, LMOutput
from .tm_conditioner import T5Conditioner, CFGDropout, ConditionProvider
from .tm_codebook_patterns import (
    Pattern, CodebooksPatternProvider, DelayedPatternProvider, get_pattern_provider
)


@dataclass
class TMModelOutput:
    """Output from the Text-to-Motion model."""
    logits: torch.Tensor     # [B, K, T, card]
    mask: torch.Tensor       # [B, K, T]
    loss: tp.Optional[torch.Tensor] = None
    loss_per_codebook: tp.Optional[tp.List[torch.Tensor]] = None


class TextToMotionLM(nn.Module):
    """
    Complete Text-to-Motion Language Model.

    Combines:
    - T5 text conditioner for cross-attention
    - MotionLM transformer with per-codebook embeddings/heads
    - Delayed codebook pattern for RVQ generation
    - CFG training dropout and inference scaling

    Args:
        n_q: Number of codebooks (default: 6 for MotionGPT)
        card: Codebook vocabulary size (default: 512)
        dim: Transformer hidden dimension
        num_heads: Attention heads
        num_layers: Transformer layers
        hidden_scale: FFN hidden dim = dim * hidden_scale
        dropout: Dropout rate
        t5_name: T5 model name for conditioning
        t5_finetune: Whether to finetune T5
        cfg_dropout: CFG dropout probability during training
        cfg_coef: CFG coefficient at inference
        pattern_type: Codebook pattern type ('delayed', 'parallel', 'flatten')
        delays: Custom delays for delayed pattern
    """
    def __init__(
        self,
        # Codebook settings
        n_q: int = 6,
        card: int = 512,
        # Transformer settings
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_scale: int = 4,
        dropout: float = 0.1,
        layer_scale: tp.Optional[float] = None,
        # T5 conditioner settings
        t5_name: str = "google/flan-t5-base",
        t5_finetune: bool = False,
        # CFG settings
        cfg_dropout: float = 0.1,
        cfg_coef: float = 3.0,
        # Pattern settings
        pattern_type: str = 'delayed',
        delays: tp.Optional[tp.List[int]] = None,
        # Initialization
        weight_init: tp.Optional[str] = 'gaussian',
        depthwise_init: tp.Optional[str] = 'current',
        zero_bias_init: bool = True,
        # Other
        emb_lr: tp.Optional[float] = None,
        device: tp.Optional[str] = None,
        dtype: tp.Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.n_q = n_q
        self.card = card
        self.dim = dim
        self.cfg_coef = cfg_coef
        self.cfg_dropout_prob = cfg_dropout

        # T5 Conditioner
        device_str = device if device is not None else 'cuda'
        self.condition_provider = ConditionProvider(
            t5_name=t5_name,
            output_dim=dim,
            finetune=t5_finetune,
            cfg_dropout=cfg_dropout,
            device=device_str,
        )

        # Motion Language Model (transformer with per-codebook embeddings/heads)
        self.lm = MotionLM(
            n_q=n_q,
            card=card,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_scale=hidden_scale,
            dropout=dropout,
            cross_attention=True,
            layer_scale=layer_scale,
            emb_lr=emb_lr,
            bias_proj=True,
            weight_init=weight_init,
            depthwise_init=depthwise_init,
            zero_bias_init=zero_bias_init,
            cfg_dropout=cfg_dropout,
            cfg_coef=cfg_coef,
            device=device,
            dtype=dtype,
        )

        # Codebook pattern provider
        self.pattern_provider = get_pattern_provider(
            n_q=n_q,
            pattern_type=pattern_type,
            delays=delays,
        )

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        mask = torch.where(mask, float('-inf'), 0.0)
        return mask

    def compute_predictions(
        self,
        codes: torch.LongTensor,
        texts: tp.List[str],
        apply_cfg_dropout: bool = True,
    ) -> TMModelOutput:
        """
        Compute predictions for training.

        Args:
            codes: Motion codes [B, K, T]
            texts: List of text descriptions
            apply_cfg_dropout: Whether to apply CFG dropout

        Returns:
            TMModelOutput with logits and mask
        """
        B, K, T = codes.shape
        device = codes.device

        # Validate codebook count matches model config
        if K != self.n_q:
            raise ValueError(
                f"Input codes have {K} codebooks, but model expects {self.n_q}. "
                f"Check that TM_MODEL.N_Q matches the number of quantizers in your VQ-VAE."
            )

        # Encode text conditions
        condition, condition_mask = self.condition_provider(
            texts, device, apply_cfg_dropout=apply_cfg_dropout
        )

        # Build pattern sequence
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, indexes, code_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )

        # Get causal mask
        S = sequence_codes.shape[-1]
        src_mask = self.get_causal_mask(S, device)

        # Forward through transformer
        logits = self.lm(
            codes=sequence_codes,
            condition=condition,
            condition_mask=condition_mask,
            src_mask=src_mask,
        )  # [B, K, S, card]

        # Revert pattern to original alignment
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [B, K, T]

        return TMModelOutput(logits=logits, mask=logits_mask)

    def compute_loss(
        self,
        codes: torch.LongTensor,
        texts: tp.List[str],
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Compute cross-entropy loss for training.

        Args:
            codes: Motion codes [B, K, T]
            texts: List of text descriptions

        Returns:
            Tuple of (total_loss, loss_per_codebook)
        """
        output = self.compute_predictions(codes, texts, apply_cfg_dropout=True)

        # Compute cross-entropy loss per codebook
        B, K, T = codes.shape
        total_loss = torch.zeros([], device=codes.device)
        loss_per_codebook = []

        for k in range(K):
            logits_k = output.logits[:, k].contiguous().view(-1, self.card)  # [B*T, card]
            targets_k = codes[:, k].contiguous().view(-1)  # [B*T]
            mask_k = output.mask[:, k].contiguous().view(-1)  # [B*T]

            # Only compute loss for valid positions
            valid_logits = logits_k[mask_k]
            valid_targets = targets_k[mask_k]

            if valid_targets.numel() > 0:
                loss_k = F.cross_entropy(valid_logits, valid_targets)
                total_loss = total_loss + loss_k
                loss_per_codebook.append(loss_k.detach())
            else:
                loss_per_codebook.append(torch.tensor(0.0, device=codes.device))

        # Average across codebooks
        total_loss = total_loss / K

        return total_loss, loss_per_codebook

    @torch.no_grad()
    def generate(
        self,
        texts: tp.List[str],
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        show_progress: bool = True,
    ) -> torch.LongTensor:
        """
        Generate motion codes from text descriptions.

        Args:
            texts: List of text descriptions
            max_gen_len: Maximum generation length
            use_sampling: Whether to sample or use greedy decoding
            temp: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            cfg_coef: CFG coefficient (overrides self.cfg_coef)
            show_progress: Show progress bar

        Returns:
            codes: Generated motion codes [B, K, T]
        """
        assert not self.training, "Generation should not be used in training mode"

        device = next(self.parameters()).device
        B = len(texts)
        K = self.num_codebooks
        cfg_coef = cfg_coef if cfg_coef is not None else self.cfg_coef

        # Prepare CFG conditions
        if cfg_coef > 1.0:
            condition, condition_mask = self.condition_provider.prepare_cfg_conditions(
                texts, device
            )
        else:
            condition, condition_mask = self.condition_provider(
                texts, device, apply_cfg_dropout=False
            )

        # Get pattern
        pattern = self.pattern_provider.get_pattern(max_gen_len)

        # Initialize with special token (not -1, to avoid negative indexing in embeddings)
        gen_codes = torch.full((B, K, max_gen_len), self.special_token_id, dtype=torch.long, device=device)

        # Build initial pattern sequence
        gen_sequence, _, gen_mask = pattern.build_pattern_sequence(
            gen_codes, self.special_token_id
        )

        gen_sequence_len = gen_sequence.shape[-1]
        iterator = range(1, gen_sequence_len)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating motion ({B} samples)")

        # Define unknown token for sanity checks (like UniMuMo)
        unknown_token = -1

        for offset in iterator:
            # Get current sequence
            curr_sequence = gen_sequence[..., :offset]

            # Sanity check: ensure no unknown tokens in current sequence (like UniMuMo)
            # This should never happen if build_pattern_sequence works correctly
            assert not (curr_sequence == unknown_token).any(), \
                f"Found unknown token (-1) in sequence at offset {offset}. " \
                f"This indicates a bug in pattern sequence building."

            # Duplicate for CFG if needed
            if cfg_coef > 1.0:
                curr_sequence = torch.cat([curr_sequence, curr_sequence], dim=0)

            # Get causal mask
            src_mask = self.get_causal_mask(offset, device)

            # Forward pass
            logits = self.lm(
                codes=curr_sequence,
                condition=condition,
                condition_mask=condition_mask,
                src_mask=src_mask,
            )  # [B or 2B, K, T, card]

            # Get last position logits
            logits = logits[:, :, -1, :]  # [B or 2B, K, card]

            # Apply CFG
            if cfg_coef > 1.0:
                cond_logits, uncond_logits = logits.split(B, dim=0)
                logits = uncond_logits + cfg_coef * (cond_logits - uncond_logits)

            # Sample next tokens
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
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probs.view(B * K, -1), num_samples=1
                ).view(B, K, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Apply valid mask (for delayed pattern)
            valid_mask = gen_mask[..., offset:offset+1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id

            # Update sequence
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == self.special_token_id,
                next_token,
                gen_sequence[..., offset:offset+1]
            )

        # Revert pattern to get final codes
        # Use -1 as special_token to mark unfilled positions (like UniMuMo's unknown_token)
        unknown_token = -1
        out_codes, _, out_mask = pattern.revert_pattern_sequence(
            gen_sequence, special_token=unknown_token
        )

        # Trim to max_gen_len
        out_codes = out_codes[..., :max_gen_len]
        out_mask = out_mask[..., :max_gen_len]

        # Sanity checks (like UniMuMo):
        # After generation, all positions in the valid range should be filled
        # If not, it indicates a bug in pattern building or generation
        num_unknown = (out_codes == unknown_token).sum().item()
        if num_unknown > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Found {num_unknown} unknown tokens (-1) in generated codes. "
                f"Replacing with 0. This may indicate incomplete generation."
            )

        # Replace any remaining -1 (unfilled positions) with 0 to avoid CUDA indexing errors
        # This is important because the VAE decoder expects valid codebook indices (0 to card-1)
        out_codes = torch.where(out_codes == unknown_token, torch.zeros_like(out_codes), out_codes)

        # Final sanity check: ensure all codes are valid
        assert (out_codes >= 0).all() and (out_codes < self.card).all(), \
            f"Generated codes out of valid range [0, {self.card-1}]"

        return out_codes

    def forward(
        self,
        codes: torch.LongTensor,
        texts: tp.List[str],
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            codes: Motion codes [B, K, T]
            texts: List of text descriptions

        Returns:
            Tuple of (loss, loss_per_codebook)
        """
        return self.compute_loss(codes, texts)


def build_text_to_motion_model(
    n_q: int = 6,
    card: int = 512,
    dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    t5_name: str = "google/flan-t5-base",
    cfg_dropout: float = 0.1,
    cfg_coef: float = 3.0,
    **kwargs
) -> TextToMotionLM:
    """
    Build Text-to-Motion model with default settings.

    Args:
        n_q: Number of codebooks
        card: Codebook vocabulary size
        dim: Transformer dimension
        num_heads: Attention heads
        num_layers: Transformer layers
        t5_name: T5 model for conditioning
        cfg_dropout: CFG dropout rate
        cfg_coef: CFG coefficient

    Returns:
        TextToMotionLM model
    """
    return TextToMotionLM(
        n_q=n_q,
        card=card,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        t5_name=t5_name,
        cfg_dropout=cfg_dropout,
        cfg_coef=cfg_coef,
        **kwargs
    )
