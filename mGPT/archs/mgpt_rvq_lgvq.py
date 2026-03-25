"""
RVQ-VAE with Language-Guided Codebook Learning (LG-VQ) — Lightweight version.

Simplified from the original LG-VQ (NeurIPS 2023) implementation:
- Removed QuantTransformer1D (heavy 2-layer Transformer encoder)
- Replaced CLS token with mean pooling + linear projection for NCE
- Mask Prediction decoder now cross-attends directly to projected motion tokens
- WRS unchanged (already lightweight)

Three language-guided alignment losses on quantized features:
1. NCE Loss: InfoNCE between mean-pooled motion and CLIP text (EOS)
2. Mask Prediction: Predict masked text tokens from projected motion tokens
3. WRS Loss: Align pairwise relation structure between text and motion codes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

import clip

from .mgpt_rvq import RVQVae


# ============================================================================
# Helper modules
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        target = target.long().detach()
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return self.confidence * nll_loss + self.smoothing * smooth_loss


class MlmLayer(nn.Module):
    """MLM prediction head: projects features to CLIP vocabulary logits."""

    def __init__(self, feat_emb_dim: int, word_emb_dim: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x: Tensor, word_embeddings: Tensor) -> Tensor:
        h = self.ln(self.gelu(self.fc(x)))
        logits = torch.matmul(h, word_embeddings.t()) + self.bias
        return logits


class WrsLayer(nn.Module):
    """Projects quantized motion features to text dimension for WRS loss."""

    def __init__(self, quant_dim: int, text_dim: int):
        super().__init__()
        self.fc = nn.Linear(quant_dim, text_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(text_dim)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(self.gelu(self.fc(x)))


# ============================================================================
# Lightweight motion-to-text projector (replaces QuantTransformer1D)
# ============================================================================

class MotionProjector(nn.Module):
    """
    Lightweight projector: mean pooling + linear for global NCE feature.
    Token-level features are passed through directly (code_dim == text_dim).
    """

    def __init__(self, code_dim: int, text_dim: int):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(code_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
        )
        nn.init.normal_(self.global_proj[0].weight, std=0.01)
        nn.init.zeros_(self.global_proj[0].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, code_dim, T'] encoder features

        Returns:
            global_feat: [B, text_dim] mean-pooled + projected (for NCE)
            token_feats: [B, T', code_dim] raw encoder tokens (for WRS/Mask)
        """
        x = x.permute(0, 2, 1)                        # [B, T', code_dim]
        global_feat = self.global_proj(x.mean(dim=1))  # [B, text_dim]
        return global_feat, x


# ============================================================================
# Simplified MaskTransformer (decoder-only, no QuantTransformer encoder)
# ============================================================================

class MaskTransformerDecoderLayer(nn.Module):
    """Single decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model: int, n_head: int, ffn_hidden: int, drop_prob: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=drop_prob, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=drop_prob, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec: Tensor, enc: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        _x = dec
        x, _ = self.self_attn(dec, dec, dec, key_padding_mask=key_padding_mask)
        x = self.norm1(self.dropout1(x) + _x)

        _x = x
        x, _ = self.cross_attn(x, enc, enc)
        x = self.norm2(self.dropout2(x) + _x)

        _x = x
        x = self.ffn(x)
        x = self.norm3(self.dropout3(x) + _x)
        return x


class MaskPredictionDecoder(nn.Module):
    """
    Decoder-only mask prediction module.
    Text tokens cross-attend to projected motion tokens (from MotionProjector).
    """

    def __init__(self, text_dim: int, n_layers: int, n_head: int, drop_prob: float):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskTransformerDecoderLayer(text_dim, n_head, text_dim, drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, text_features: Tensor, motion_tokens: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            text_features: [B, N_text, text_dim] (possibly masked) text token features
            motion_tokens: [B, T', text_dim] projected motion tokens
            key_padding_mask: [B, N_text] True = padded position (to ignore)

        Returns:
            [B, N_text, text_dim] decoded text features
        """
        x = text_features
        for layer in self.layers:
            x = layer(x, motion_tokens, key_padding_mask=key_padding_mask)
        return x


# ============================================================================
# Main class: RVQVaeLGVQ
# ============================================================================

class RVQVaeLGVQ(RVQVae):
    """
    RVQ-VAE with Language-Guided Codebook Learning (LG-VQ) — Lightweight.

    Uses mean pooling + linear projection instead of QuantTransformer1D.

    Three text-guided losses on the quantized features:
    1. NCE: Global contrastive alignment (mean-pooled motion <-> CLIP text EOS)
    2. Mask Prediction: Predict masked CLIP text tokens from motion tokens
    3. WRS: Weighted Relation Supervision (structural alignment)
    """

    def __init__(
        self,
        clip_model_name: str = 'ViT-B/32',
        clip_text_max_len: int = 30,
        mask_transformer_layers: int = 2,
        mask_transformer_heads: int = 4,
        max_motion_len: int = 128,
        drop_prob: float = 0.1,
        nce_weight: float = 0.001,
        mask_weight: float = 0.0,
        wrs_weight: float = 0.0,
        label_smoothing: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.text_dim = 512  # CLIP ViT-B/32 output dim
        self.clip_text_max_len = clip_text_max_len
        self.nce_weight = nce_weight
        self.mask_weight = mask_weight
        self.wrs_weight = wrs_weight

        # Frozen CLIP text encoder
        self.clip_model, _ = clip.load(clip_model_name, device='cpu', jit=False)
        self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # Lightweight motion projector (replaces QuantTransformer1D)
        self.motion_projector = MotionProjector(self.code_dim, self.text_dim)

        # Mask prediction decoder (cross-attends to projected motion tokens)
        self.mask_decoder = MaskPredictionDecoder(
            text_dim=self.text_dim,
            n_layers=mask_transformer_layers,
            n_head=mask_transformer_heads,
            drop_prob=drop_prob,
        )

        # Learnable mask tokens for masked text positions
        self.mask_learned_parameter = nn.Parameter(
            torch.randn(1, clip_text_max_len, self.text_dim) * 0.01
        )

        # MLM prediction head
        vocab_size = self.clip_model.vocab_size
        self.mlm_layer = MlmLayer(self.text_dim, self.text_dim, vocab_size)

        # Label-smoothed cross entropy for MLM
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        # WRS projection layer (code_dim -> text_dim)
        self.wrs_layer = WrsLayer(self.code_dim, self.text_dim)

        # MSE loss for WRS
        self.mse_loss = nn.MSELoss()

    def train(self, mode: bool = True):
        """Keep CLIP frozen in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def _encode_text_clip(self, texts: List[str], device: torch.device):
        """
        Encode texts with frozen CLIP.

        Returns:
            all_text_features: [B, max_len, 512] per-token features
            last_text_feature: [B, 512] global (EOS) text feature
            text_mask: [B, max_len] 1=visible, 0=masked (for MLM)
            mask_padding: [B, max_len] 1=valid token, 0=padding
            text_tokens: [B, max_len] token indices
        """
        text_tokens_full = clip.tokenize(texts, truncate=True).to(device)  # [B, 77]

        with torch.no_grad():
            x = self.clip_model.token_embedding(text_tokens_full).float()  # [B, 77, 512]
            x = x + self.clip_model.positional_embedding.float()
            x = x.permute(1, 0, 2)  # [77, B, 512]
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # [B, 77, 512]
            x = self.clip_model.ln_final(x).float()

            eos_indices = text_tokens_full.argmax(dim=-1)  # [B]
            last_text_feature = x[torch.arange(x.shape[0]), eos_indices]  # [B, 512]

        max_len = self.clip_text_max_len
        text_tokens = text_tokens_full[:, :max_len]
        all_text_features = x[:, :max_len].detach()  # [B, max_len, 512]

        # Ensure EOS token is preserved within the truncated window
        eos_token_id = 49407
        has_eos = (text_tokens == eos_token_id).any(dim=-1)  # [B]
        text_tokens = text_tokens.clone()
        text_tokens[~has_eos, max_len - 1] = eos_token_id

        mask_padding = (text_tokens != 0).float()

        # Random masking for MLM (~15% of valid tokens, skip BOS/EOS)
        text_mask = torch.ones_like(mask_padding)  # 1 = visible, 0 = masked
        if self.training:
            for i in range(text_tokens.shape[0]):
                valid_len = int(mask_padding[i].sum().item())
                if valid_len > 2:
                    maskable = torch.arange(1, valid_len - 1, device=device)
                    n_mask = max(1, int(len(maskable) * 0.15))
                    perm = torch.randperm(len(maskable), device=device)[:n_mask]
                    mask_positions = maskable[perm]
                    text_mask[i, mask_positions] = 0.0

        return all_text_features, last_text_feature, text_mask, mask_padding, text_tokens

    # ---- Three alignment losses ----

    def global_infor_sup(self, vision_global: Tensor, last_text_feature: Tensor,
                         temperature: float = 0.07) -> Tensor:
        """InfoNCE between mean-pooled motion and CLIP text EOS."""
        v = F.normalize(vision_global, dim=-1)
        t = F.normalize(last_text_feature.detach(), dim=-1)
        logits = (v @ t.t()) / temperature  # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        nce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]
        return nce_loss

    def mask_prediction(self, motion_tokens: Tensor, all_text_features: Tensor,
                        text_mask: Tensor, mask_padding: Tensor,
                        text_tokens: Tensor) -> Tensor:
        """
        Masked text prediction from projected motion tokens.

        Args:
            motion_tokens: [B, T', text_dim] projected motion token features
            all_text_features: [B, N_text, text_dim] CLIP token features
            text_mask: [B, N_text] 1=visible, 0=masked
            mask_padding: [B, N_text] 1=valid token, 0=padding
            text_tokens: [B, N_text] ground truth token indices

        Returns:
            mlm_loss: scalar masked language modeling loss
        """
        B = motion_tokens.shape[0]

        # Replace masked positions with learnable mask tokens
        mask_expanded = text_mask.unsqueeze(-1).expand_as(all_text_features)
        mask_param = self.mask_learned_parameter.expand(B, -1, -1)
        masked_text = torch.where(mask_expanded == 0, mask_param, all_text_features)
        masked_text = masked_text * mask_padding.unsqueeze(-1)

        # key_padding_mask: True = padded (to ignore)
        key_padding_mask = mask_padding == 0  # [B, N_text]

        # Decoder: text cross-attends to motion tokens
        output = self.mask_decoder(masked_text, motion_tokens, key_padding_mask)
        # output: [B, N_text, text_dim]

        # MLM loss on masked positions only
        mask_pre = (~text_mask.bool()) & mask_padding.bool()
        mask_pre = mask_pre.float()  # [B, N_text]

        word_embeddings = self.clip_model.token_embedding.weight.data.detach().float()
        logits = self.mlm_layer(output, word_embeddings)  # [B, N_text, vocab_size]

        N_text = text_tokens.shape[1]
        loss = self.criterion(
            logits.reshape(B * N_text, -1),
            text_tokens.reshape(B * N_text)
        )
        loss = loss.reshape(B, N_text)
        mlm_loss = (loss * mask_pre).sum() / (mask_pre.sum() + 1e-5)

        return mlm_loss

    def wrs_relation_sup(self, motion_tokens: Tensor, x_encoder: Tensor,
                         all_text_features: Tensor, mask_padding: Tensor) -> Tensor:
        """
        Weighted Relation Supervision loss.
        Aligns pairwise relation structure between text tokens and motion code tokens.
        """
        B = x_encoder.shape[0]

        # Project encoder features to text space
        x = x_encoder.permute(0, 2, 1)  # [B, T', code_dim]
        quant_proj = self.wrs_layer(x)      # [B, T', text_dim]

        # Normalize
        text_norm = F.normalize(all_text_features, p=2, dim=-1)
        vision_norm = F.normalize(motion_tokens, p=2, dim=-1)

        # Find nearest motion token for each text token
        sim = torch.matmul(text_norm, vision_norm.permute(0, 2, 1))  # [B, N_text, T']
        values, indices = sim.max(dim=-1)  # [B, N_text]

        # Gather matched code tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.text_dim)
        text_to_quant = torch.gather(quant_proj, 1, indices_expanded)  # [B, N_text, text_dim]

        # Text relation matrix (ground truth)
        text_masked = text_norm * mask_padding.unsqueeze(-1)
        q = torch.matmul(text_masked, text_masked.permute(0, 2, 1))

        # Code relation matrix (weighted by similarity)
        text_to_quant = text_to_quant * values.unsqueeze(-1)
        text_to_quant = F.normalize(text_to_quant, p=2, dim=-1)
        text_to_quant = text_to_quant * mask_padding.unsqueeze(-1)
        p = torch.matmul(text_to_quant, text_to_quant.permute(0, 2, 1))

        return self.mse_loss(p, q.detach())

    def text_supervise(self, x_encoder: Tensor, all_text_features: Tensor,
                       last_text_feature: Tensor, text_mask: Tensor,
                       mask_padding: Tensor, text_tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Orchestrate all three text supervision losses (skips disabled losses)."""
        device = x_encoder.device
        # Project encoder features (before quantization, for cleaner gradients)
        global_feat, motion_tokens = self.motion_projector(x_encoder)

        # 1. NCE: mean-pooled motion vs CLIP text
        nce_loss = (self.global_infor_sup(global_feat, last_text_feature).mean()
                    if self.nce_weight > 0 else torch.tensor(0.0, device=device))

        # 2. Mask prediction: decoder cross-attends to motion tokens
        mask_loss = (self.mask_prediction(
            motion_tokens, all_text_features, text_mask, mask_padding, text_tokens
        ) if self.mask_weight > 0 else torch.tensor(0.0, device=device))

        # 3. WRS: structural alignment
        wrs_loss = (self.wrs_relation_sup(
            motion_tokens, x_encoder, all_text_features, mask_padding
        ) if self.wrs_weight > 0 else torch.tensor(0.0, device=device))

        return nce_loss, mask_loss, wrs_loss

    def forward(
        self,
        features: Tensor,
        texts: Optional[List[str]] = None,
        clip_text_features: Optional[Tensor] = None,
    ):
        """
        Forward pass through LG-VQ RVQ-VAE.

        Args:
            features: Input motion features [B, T, D]
            texts: Optional list of text descriptions (fallback for CLIP encoding)
            clip_text_features: Optional precomputed CLIP text features [B, 512]

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss_commit: Commitment loss from RVQ
            perplexity: Codebook usage metric
            motion_emb: [B, 512] L2-normed motion embedding (or None)
            text_emb: [B, 512] L2-normed text embedding (or None)
        """
        # Standard RVQ forward
        x_in = self.preprocess(features)          # [B, D, T]
        x_encoder = self.encoder(x_in)            # [B, code_dim, T']
        x_quantized, all_indices, loss_commit, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)     # [B, D, T]
        x_out = self.postprocess(x_decoder)       # [B, T, D]

        device = features.device
        motion_emb = None
        text_emb = None

        if not self.training or (self.nce_weight == 0 and self.mask_weight == 0 and self.wrs_weight == 0):
            return x_out, loss_commit, perplexity, motion_emb, text_emb

        # Get text embedding: precomputed or CLIP forward
        if clip_text_features is not None:
            text_emb = F.normalize(clip_text_features.to(device).float(), dim=-1)
        elif texts is not None and len(texts) > 1:
            _, last_text_feature, _, _, _ = self._encode_text_clip(texts, device)
            text_emb = F.normalize(last_text_feature.detach(), dim=-1)
        else:
            return x_out, loss_commit, perplexity, motion_emb, text_emb

        # Get motion embedding
        global_feat, _ = self.motion_projector(x_encoder)
        motion_emb = F.normalize(global_feat, dim=-1)

        return x_out, loss_commit, perplexity, motion_emb, text_emb
