"""
Masked Transformer for Text-to-Motion Generation

Adapted from MoMask's MaskTransformer with CLIP text encoding.
Uses iterative masked prediction with cosine scheduling.

Architecture:
    Text → CLIP (frozen) → Text Features
    Motion Tokens → Token Embeddings → Transformer → Logits

    Training: Random masking with BERT-style strategy
    Inference: Iterative refinement with classifier-free guidance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from typing import List, Optional
from functools import partial
from torch.distributions.categorical import Categorical


# ============================================================================
# Helper Functions
# ============================================================================

def cosine_schedule(t):
    """Cosine noise schedule for masking."""
    return torch.cos(t * np.pi * 0.5)


def uniform(shape, device):
    """Sample uniform random values."""
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def lengths_to_mask(lengths, max_len):
    """
    Convert lengths to boolean mask.

    Args:
        lengths: (B,) sequence lengths
        max_len: maximum sequence length

    Returns:
        mask: (B, max_len) where True = valid position
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask


def get_mask_subset_prob(mask, prob):
    """
    Randomly mask a subset of True positions with given probability.

    Args:
        mask: (B, T) boolean mask
        prob: probability of masking

    Returns:
        subset_mask: (B, T) boolean mask
    """
    batch_size, seq_len = mask.shape
    rand = torch.rand(batch_size, seq_len, device=mask.device)
    subset_mask = mask & (rand < prob)
    return subset_mask


def top_k(logits, thres=0.9, dim=-1):
    """
    Top-k filtering for logits.

    Args:
        logits: (..., vocab_size)
        thres: keep top-k tokens with cumulative probability >= thres
        dim: dimension to apply top-k

    Returns:
        filtered_logits: (..., vocab_size)
    """
    k = int((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs


def gumbel_sample(logits, temperature=1.0, dim=-1):
    """Gumbel-softmax sampling."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return ((logits / temperature) + gumbel_noise).argmax(dim=dim)


def cal_performance(logits, labels, ignore_index=-100):
    """
    Calculate cross-entropy loss and accuracy.

    Args:
        logits: (B, vocab_size, T)
        labels: (B, T)
        ignore_index: index to ignore in loss

    Returns:
        loss: scalar
        pred_id: (B, T) predicted token ids
        acc: scalar accuracy
    """
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

    pred_id = logits.argmax(dim=1)  # (B, T)
    mask = (labels != ignore_index)

    if mask.sum() > 0:
        n_correct = (pred_id[mask] == labels[mask]).sum().item()
        n_total = mask.sum().item()
        acc = n_correct / n_total
    else:
        acc = 0.0

    return loss, pred_id, acc


def eval_decorator(fn):
    """Decorator to set model to eval mode during generation."""
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


# ============================================================================
# Model Components
# ============================================================================

class InputProcess(nn.Module):
    """Process token embeddings to latent space."""

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_feats)
        Returns:
            x: (T, B, latent_dim)
        """
        x = x.permute(1, 0, 2)  # (T, B, input_feats)
        x = self.poseEmbedding(x)  # (T, B, latent_dim)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (T, B, d_model)
        Returns:
            x: (T, B, d_model)
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess(nn.Module):
    """Output projection with BERT-style transformation."""

    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (T, B, latent_dim)
        Returns:
            output: (B, out_feats, T)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # (T, B, out_feats)
        output = output.permute(1, 2, 0)  # (B, out_feats, T)
        return output


# ============================================================================
# Main Model
# ============================================================================

class MaskedTransformerT2M(nn.Module):
    """
    Masked Transformer for Text-to-Motion Generation.

    Uses CLIP for text encoding and iterative masked prediction for generation.
    """

    def __init__(
        self,
        num_tokens=512,           # Codebook size
        code_dim=512,             # Token embedding dimension
        latent_dim=256,           # Transformer hidden dimension
        ff_size=1024,             # Feedforward dimension
        num_layers=8,             # Number of transformer layers
        num_heads=4,              # Number of attention heads
        dropout=0.1,              # Dropout rate
        cond_drop_prob=0.1,       # Classifier-free guidance dropout
        clip_dim=512,             # CLIP output dimension
        clip_version='ViT-B/32', # CLIP model version
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.cond_drop_prob = cond_drop_prob

        # Special token IDs
        self.mask_id = num_tokens      # Mask token
        self.pad_id = num_tokens + 1   # Padding token

        print(f"\n{'='*70}")
        print(f"{'Masked Transformer T2M':^70}")
        print(f"{'='*70}")
        print(f"{'Num Tokens:':<25} {num_tokens}")
        print(f"{'Code Dim:':<25} {code_dim}")
        print(f"{'Latent Dim:':<25} {latent_dim}")
        print(f"{'FF Size:':<25} {ff_size}")
        print(f"{'Num Layers:':<25} {num_layers}")
        print(f"{'Num Heads:':<25} {num_heads}")
        print(f"{'Dropout:':<25} {dropout}")
        print(f"{'Cond Drop Prob:':<25} {cond_drop_prob}")
        print(f"{'CLIP Version:':<25} {clip_version}")
        print(f"{'='*70}\n")

        # Token embeddings (includes mask and pad tokens)
        _num_tokens = num_tokens + 2
        self.token_emb = nn.Embedding(_num_tokens, code_dim)

        # Input/output processing
        self.input_process = InputProcess(code_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.output_process = OutputProcess(num_tokens, latent_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (T, B, D) format
        )
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Text conditioning via CLIP (frozen)
        self.cond_emb = nn.Linear(self.clip_dim, latent_dim)

        # Cosine noise schedule for masking
        self.noise_schedule = cosine_schedule

        # Initialize weights BEFORE loading CLIP (so CLIP pretrained weights are preserved)
        self.apply(self._init_weights)

        # Load CLIP after weight init
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_and_freeze_clip(self, clip_version):
        """Load and freeze CLIP model."""
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        """Encode text using CLIP.

        Args:
            raw_text: List[str] of text descriptions

        Returns:
            feat_clip_text: (B, clip_dim)
        """
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def parameters_wo_clip(self):
        """Return parameters excluding CLIP."""
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_token_emb(self, codebook):
        """
        Initialize token embeddings from VQ-VAE codebook.

        Args:
            codebook: (num_tokens, code_dim) codebook embeddings
        """
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape

        # Add two dummy tokens (mask and pad) with zero embeddings
        token_weights = torch.cat([
            codebook,
            torch.zeros(size=(2, d), device=codebook.device)
        ], dim=0)

        self.token_emb.weight = nn.Parameter(token_weights)
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized and frozen!")

    def mask_cond(self, cond, force_mask=False):
        """
        Apply classifier-free guidance masking to conditioning.

        Args:
            cond: (B, D) conditioning features
            force_mask: if True, always mask (for unconditional generation)

        Returns:
            masked_cond: (B, D)
        """
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            ).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        """
        Transformer forward pass.

        Args:
            motion_ids: (B, T) token indices
            cond: (B, clip_dim) text features
            padding_mask: (B, T) True for padded positions
            force_mask: if True, use unconditional generation

        Returns:
            logits: (B, num_tokens, T)
        """
        # Apply classifier-free guidance masking
        cond = self.mask_cond(cond, force_mask=force_mask)

        # Embed tokens
        x = self.token_emb(motion_ids)  # (B, T, code_dim)
        x = self.input_process(x)  # (T, B, latent_dim)

        # Conditioning
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, B, latent_dim)

        # Add positional encoding
        x = self.position_enc(x)

        # Prepend conditioning
        xseq = torch.cat([cond, x], dim=0)  # (T+1, B, latent_dim)

        # Extend padding mask for the condition token (condition is never padded)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)  # (B, T+1)

        # Transformer encoding with bidirectional attention + padding mask
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[1:]  # (T, B, latent_dim)

        # Output projection
        logits = self.output_process(output)  # (B, num_tokens, T)

        return logits

    def forward(self, ids, texts, m_lens):
        """
        Training forward pass with random masking.

        Args:
            ids: (B, T) motion token indices
            texts: List[str] of text descriptions
            m_lens: (B,) actual sequence lengths

        Returns:
            ce_loss: cross-entropy loss
            pred_id: (B, T) predicted token ids
            acc: accuracy
        """
        bs, ntokens = ids.shape
        device = ids.device

        # Create padding mask (True for padded positions)
        non_pad_mask = lengths_to_mask(m_lens, ntokens)  # (B, T)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        # Encode text (no grad for frozen CLIP)
        with torch.no_grad():
            text_features = self.encode_text(texts)  # (B, 512)

        # Random masking schedule
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        # Random positions to mask
        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # Only mask non-padded positions
        mask &= non_pad_mask

        # Create labels (what we want to predict)
        labels = torch.where(mask, ids, self.mask_id)

        # Create corrupted input (BERT-style masking)
        x_ids = ids.clone()

        # 10% replace with random incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)

        # 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        # Forward pass
        logits = self.trans_forward(x_ids, text_features, ~non_pad_mask, force_mask=False)

        # Calculate loss and accuracy
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(
        self, motion_ids, cond_vector, padding_mask, cond_scale=3, force_mask=False
    ):
        """
        Forward pass with classifier-free guidance scaling.

        Args:
            motion_ids: (B, T) token indices
            cond_vector: (B, D) conditioning features
            padding_mask: (B, T) padding mask
            cond_scale: classifier-free guidance scale
            force_mask: if True, unconditional generation

        Returns:
            scaled_logits: (B, num_tokens, T)
        """
        if force_mask or cond_scale == 1:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=force_mask)

        # Conditional logits
        logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=False)

        # Unconditional logits
        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        # Classifier-free guidance: unconditional + scale * (conditional - unconditional)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts,
        m_lens,
        timesteps=18,
        cond_scale=3,
        temperature=1.0,
        topk_filter_thres=0.9,
        gsample=False,
        force_mask=False
    ):
        """
        Iterative masked generation.

        Args:
            texts: List[str] text descriptions
            m_lens: (B,) or List[int] target lengths (in tokens)
            timesteps: number of iterative refinement steps
            cond_scale: classifier-free guidance scale
            temperature: sampling temperature
            topk_filter_thres: top-k filtering threshold
            gsample: use Gumbel sampling if True
            force_mask: unconditional generation if True

        Returns:
            ids: (B, max_len) generated token ids
        """
        device = next(self.parameters()).device

        # Convert m_lens to tensor if needed
        if isinstance(m_lens, list):
            m_lens = torch.tensor(m_lens, device=device)

        batch_size = len(texts)
        seq_len = max(m_lens)

        # Encode text
        cond_vector = self.encode_text(texts)  # (B, 512)

        # Create padding mask
        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all masked tokens
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)

        starting_temperature = temperature

        # Iterative refinement
        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = self.noise_schedule(timestep)

            # Number of tokens to mask at this step
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)

            # Select lowest-confidence tokens to mask
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            # Predict tokens
            logits = self.forward_with_cond_scale(
                ids, cond_vector, padding_mask, cond_scale, force_mask
            )
            logits = logits.permute(0, 2, 1)  # (B, T, num_tokens)

            # Filter low-probability tokens
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            # Sample predictions
            temperature_curr = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature_curr, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature_curr, dim=-1)
                pred_ids = Categorical(probs).sample()

            # Update ids with predictions
            ids = torch.where(is_mask, pred_ids, ids)

            # Update confidence scores
            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None, **kwargs):
        """
        Generation interface compatible with MotionGPT.

        Args:
            texts: List[str] text descriptions
            lengths: List[int] or Tensor - target lengths in frames
            stage: 'test' or 'val'
            tasks: not used

        Returns:
            List[Tensor]: List of (T_i,) token sequences
        """
        device = next(self.parameters()).device

        # Convert frame lengths to token lengths (divide by 4)
        if isinstance(lengths, list):
            token_lengths = [l // 4 for l in lengths]
        else:
            token_lengths = (lengths // 4).tolist()

        # Generate tokens
        generated_ids = self.generate(
            texts=texts,
            m_lens=token_lengths,
            timesteps=18,
            cond_scale=3,
            temperature=1.0,
            topk_filter_thres=0.9,
            gsample=False,
            force_mask=False
        )  # (B, T)

        # Convert to list of variable-length tensors
        outputs_tokens = []
        for i in range(generated_ids.shape[0]):
            actual_len = token_lengths[i]
            outputs_tokens.append(generated_ids[i, :actual_len])

        return outputs_tokens
