"""
Hierarchical RVQ-GPT for Motion Generation

Implements hierarchical prediction scheme for Residual Vector Quantized codes:
    P(Q0, Q1, Q2 | text) = ∏_t P(Q0^t | Q0^{<t}, text) ×
                                P(Q1^t | Q1^{<t}, Q0^{≤t}, text) ×
                                P(Q2^t | Q2^{<t}, Q0^{≤t}, Q1^{≤t}, text)

Key Innovation:
- Uses first 3 quantizers (Q0, Q1, Q2) from a 6-quantizer RVQ-VAE
- Q0 (coarse) predicted first at each timestep
- Q1 (medium) explicitly conditioned on Q0
- Q2 (fine) explicitly conditioned on Q0+Q1
- Creates hierarchical dependency matching RVQ's natural coarse-to-fine structure

Architecture:
    Text → Text Encoder
              ↓
    Step 1: Q0 Decoder → Q0^t (coarse code)
              ↓
    Step 2: Q1 Decoder (conditioned on Q0^t) → Q1^t (medium refinement)
              ↓
    Step 3: Q2 Decoder (conditioned on Q0^t + Q1^t) → Q2^t (fine refinement)

Adapted from SOKE's hierarchical sign language generation architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .tools.rvq_hierarchical_blocks import HierarchicalRVQDecoder
from .tools.t2m_decoder_blocks import T2MStyleQ0Decoder
from .tools.llamagen_blocks import LlamaGenQ0Decoder
from .speech_encoder import SpeechEncoder, SPEECH_ENCODER_CONFIGS
from .pos_encoding import PositionEmbedding
import clip

# Text encoder configurations
TEXT_ENCODER_CONFIGS = {
    'clip': {'dim': 512, 'model': 'ViT-B/32'},
    'bert': {'dim': 768, 'model': 'bert-base-uncased'},
    'bert-large': {'dim': 1024, 'model': 'bert-large-uncased'},
}


class TextEncoder(nn.Module):
    """
    Switchable text encoder supporting CLIP and BERT.

    Returns token-level embeddings for fine-grained text-motion alignment.

    Usage:
        encoder = TextEncoder(encoder_type='clip')  # or 'bert', 'bert-large'
        text_features, attention_mask = encoder(texts)
        # BERT: (B, seq_len, 768), (B, seq_len) - token-level
        # CLIP: (B, 1, 512), None - sentence-level
    """

    def __init__(self, encoder_type: str = 'clip', freeze: bool = True):
        """
        Args:
            encoder_type: One of 'clip', 'bert', 'bert-large'
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        if encoder_type not in TEXT_ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. "
                           f"Choose from {list(TEXT_ENCODER_CONFIGS.keys())}")

        self.encoder_type = encoder_type
        self.config = TEXT_ENCODER_CONFIGS[encoder_type]
        self.text_dim = self.config['dim']

        if encoder_type == 'clip':
            self.model, _ = clip.load(self.config['model'], device='cpu', jit=False)
        else:
            # BERT variants
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.config['model'])
            self.model = BertModel.from_pretrained(self.config['model'])

        # Freeze encoder
        self._freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        """Override to keep frozen encoder in eval mode."""
        if self._freeze:
            return super().train(False)
        return super().train(mode)

    @property
    def output_dim(self) -> int:
        return self.text_dim

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text to features.

        Args:
            texts: List of text strings

        Returns:
            text_features: (B, seq_len, text_dim) for BERT (token-level), (B, 1, text_dim) for CLIP
            attention_mask: (B, seq_len) for masking padding tokens, or None for CLIP
        """
        device = next(self.model.parameters()).device

        if self.encoder_type == 'clip':
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens).float()
            # Add sequence dimension for compatibility: (B, 512) -> (B, 1, 512)
            text_features = text_features.unsqueeze(1)
            attention_mask = None  # CLIP doesn't need mask (single token)
        else:
            # BERT variants - Return ALL token embeddings (token-level)
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,  # Match CLIP's max length
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use ALL hidden states (token-level), not just CLS!
                text_features = outputs.last_hidden_state.float()  # (B, seq_len, 768)
            attention_mask = inputs.attention_mask  # (B, seq_len) - mask padding

        return text_features, attention_mask


class HierarchicalRVQGPT(nn.Module):
    """
    Hierarchical RVQ-GPT with explicit Q0→Q1→Q2 dependency.

    Three decoders with hierarchical generation:
    1. Q0 decoder: P(Q0 | text) - coarse motion codes
    2. Q1 decoder: P(Q1 | Q0, text) - medium refinement
    3. Q2 decoder: P(Q2 | Q0, Q1, text) - fine refinement

    During generation:
    - Predict Q0 token first (coarse)
    - Use Q0 token embedding to condition Q1 prediction (medium)
    - Use Q0+Q1 embeddings to condition Q2 prediction (fine)
    - Ensures refinement codes are coherent with coarse codes
    """

    def __init__(
        self,
        num_vq=512,
        embed_dim=1024,
        block_size=200,
        num_layers=9,
        num_layers_q0=None,  # NEW: Separate layer count for Q0 decoder (default: num_layers)
        num_layers_q1=None,  # NEW: Separate layer count for Q1 decoder (default: num_layers)
        num_layers_q2=None,  # NEW: Separate layer count for Q2 decoder (default: num_layers)
        n_head=16,
        dropout=0.1,
        text_dim=512,
        text_encoder_type='clip',  # 'clip', 'bert', or 'bert-large'
        speech_encoder_type=None,  # NEW: 'hubert-large', 'wavlm-large', etc.
        use_speech=False,  # NEW: Whether to use speech instead of text
        n_kv_head=None,  # GQA KV heads for Q0 LlamaGen decoder (default = n_head = standard MHA)
        drop_path_rate=0.0,  # DropPath rate for Q0 LlamaGen decoder
        pkeep=1.0,  # Probability of keeping GT tokens for cross-decoder conditioning
        contrastive_speech_ckpt=None,  # Path to contrastive speech_head_best.pt
        contrastive_proj_dim=512,  # Projection dim used in contrastive training
        contrastive_num_layers=4,  # Number of transformer layers in contrastive projection head
        contrastive_nhead=8,  # Number of attention heads in contrastive projection head
        contrastive_ff_dim=None,  # Feedforward dim in contrastive projection head (default: proj_dim * 4)
        contrastive_max_seq_len=2000,  # Max sequence length for contrastive positional encoding
        freeze_contrastive_proj=True,  # Whether to freeze contrastive projection weights
        q0_decoder_type='llamagen',  # 'llamagen' (prefix-concat) or 'transformer' (cross-attention)
        q1_use_cond=True,  # Whether Q1 cross-attends to conditioning features (speech/text)
        q2_use_cond=True,  # Whether Q2 cross-attends to conditioning features (speech/text)
        # Per-time corrupted RVQ augmentation (T2M-HiFiGPT paper Sec 3.4)
        corrupt_ratio=0.5,  # τ: fraction of timesteps to corrupt (0.0 = disabled)
    ):
        """
        Args:
            num_vq: Codebook size per quantizer (typically 512)
            embed_dim: Token embedding dimension
            block_size: Maximum sequence length
            num_layers: Number of transformer layers per decoder (default for all)
            num_layers_q0: Number of layers for Q0 decoder (if None, uses num_layers)
            num_layers_q1: Number of layers for Q1 decoder (if None, uses num_layers)
            num_layers_q2: Number of layers for Q2 decoder (if None, uses num_layers)
            n_head: Number of attention heads
            dropout: Dropout probability
            text_dim: Text feature dimension (auto-set based on text_encoder_type if not specified)
            text_encoder_type: Type of text encoder ('clip', 'bert', 'bert-large')
            speech_encoder_type: Type of speech encoder ('hubert-large', 'wavlm-large', 'whisper-large-v3', etc.)
            use_speech: If True, use speech encoder instead of text encoder
            pkeep: Probability of keeping GT tokens for hierarchical conditioning
                   1.0 = full teacher forcing (100% GT)
                   0.5 = scheduled sampling (50% GT, 50% predicted)
                   0.0 = no teacher forcing (100% predicted)
        """
        super().__init__()

        # Set decoder-specific layer counts (default to num_layers if not specified)
        self.num_layers_q0 = num_layers_q0 if num_layers_q0 is not None else num_layers
        self.num_layers_q1 = num_layers_q1 if num_layers_q1 is not None else num_layers
        self.num_layers_q2 = num_layers_q2 if num_layers_q2 is not None else num_layers

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_layers = num_layers  # Keep for backward compatibility
        self.num_quantizers_used = 3  # Use first 3 of 6 quantizers
        self.pkeep = pkeep
        self.eos_token_id = num_vq  # EOS token is index num_vq (e.g., 512)
        self.text_encoder_type = text_encoder_type
        self.speech_encoder_type = speech_encoder_type
        self.use_speech = use_speech
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.drop_path_rate = drop_path_rate
        self.q0_decoder_type = q0_decoder_type
        self.q1_use_cond = q1_use_cond
        self.q2_use_cond = q2_use_cond
        self.corrupt_ratio = corrupt_ratio

        # Initialize encoders based on modality
        self.use_contrastive_proj = contrastive_speech_ckpt is not None
        if use_speech:
            # Speech-driven mode
            assert speech_encoder_type is not None, "speech_encoder_type must be specified when use_speech=True"
            print(f"\n{'='*70}")
            print(f"{'Speech-Driven Hierarchical RVQ-GPT':^70}")
            print(f"{'='*70}")

            self.speech_encoder = SpeechEncoder(encoder_type=speech_encoder_type, freeze=True)
            audio_dim = self.speech_encoder.output_dim

            if contrastive_speech_ckpt is not None:
                # Contrastive-pretrained projection head (sequence output for cross-attention)
                # Architecture matches ModalityProjectionHead from contrastive training
                self.contrastive_input_proj = nn.Linear(audio_dim, contrastive_proj_dim)
                self.contrastive_pos_enc = PositionEmbedding(contrastive_max_seq_len, contrastive_proj_dim, dropout)
                _ff_dim = contrastive_ff_dim if contrastive_ff_dim is not None else contrastive_proj_dim * 4
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=contrastive_proj_dim,
                    nhead=contrastive_nhead,
                    dim_feedforward=_ff_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                self.contrastive_transformer = nn.TransformerEncoder(encoder_layer, num_layers=contrastive_num_layers)

                # Load pretrained weights
                import os
                if os.path.exists(contrastive_speech_ckpt):
                    ckpt = torch.load(contrastive_speech_ckpt, map_location='cpu')
                    # Map ModalityProjectionHead keys → our contrastive_ prefixed names
                    mapped_state = {}
                    for k, v in ckpt.items():
                        mapped_state['contrastive_' + k] = v
                    # Handle pos_enc size mismatch: checkpoint may have shorter embedding
                    pos_key = 'contrastive_pos_enc.embed'
                    if pos_key in mapped_state:
                        ckpt_len = mapped_state[pos_key].shape[0]
                        model_len = self.contrastive_pos_enc.embed.shape[0]
                        if ckpt_len < model_len:
                            padded = self.contrastive_pos_enc.embed.data.clone()
                            padded[:ckpt_len] = mapped_state[pos_key]
                            mapped_state[pos_key] = padded
                    missing, unexpected = self.load_state_dict(mapped_state, strict=False)
                    print(f"[Contrastive] Loaded {len(mapped_state)} keys from {contrastive_speech_ckpt}")
                    contrastive_missing = [k for k in missing if 'contrastive_' in k]
                    if contrastive_missing:
                        print(f"[Contrastive] Missing keys: {contrastive_missing}")
                else:
                    print(f"[WARNING] Contrastive checkpoint not found: {contrastive_speech_ckpt}")

                # Optionally freeze contrastive projection
                if freeze_contrastive_proj:
                    for name, p in self.named_parameters():
                        if 'contrastive_' in name:
                            p.requires_grad = False
                    print(f"[Contrastive] Projection head frozen")

                # Bridge: identity if proj_dim == embed_dim, otherwise linear
                if contrastive_proj_dim == embed_dim:
                    self.audio_proj = nn.Identity()
                    conditioning_modality = "Speech (contrastive-pretrained, direct)"
                    conditioning_dim = f"{audio_dim}D → {contrastive_proj_dim}D (contrastive) = {embed_dim}D (no bridge)"
                else:
                    self.audio_proj = nn.Linear(contrastive_proj_dim, embed_dim)
                    conditioning_modality = "Speech (contrastive-pretrained)"
                    conditioning_dim = f"{audio_dim}D → {contrastive_proj_dim}D (contrastive) → {embed_dim}D"
            else:
                # Simple linear projection (no contrastive pretraining)
                self.audio_proj = nn.Linear(audio_dim, embed_dim)

                conditioning_modality = "Speech"
                conditioning_dim = f"{audio_dim}D → {embed_dim}D"


            # Disable text encoder
            self.text_encoder = None
            self.text_proj = None

            conditioning_encoder = speech_encoder_type
        else:
            # Text-driven mode (original)
            print(f"\n{'='*70}")
            print(f"{'Text-Driven Hierarchical RVQ-GPT':^70}")
            print(f"{'='*70}")

            self.text_encoder = TextEncoder(encoder_type=text_encoder_type, freeze=True)
            text_dim = self.text_encoder.output_dim

            # Text projection: text_dim → embed_dim
            self.text_proj = nn.Linear(text_dim, embed_dim)

            # Disable speech encoder
            self.speech_encoder = None
            self.audio_proj = None

            conditioning_modality = "Text"
            conditioning_encoder = text_encoder_type
            conditioning_dim = f"{text_dim}D → {embed_dim}D"

        print(f"{'Architecture:':<20} Hierarchical Q0 → Q1 → Q2")
        print(f"{'Quantizers Used:':<20} 3 (first 3 of 6 from RVQ-VAE)")
        print(f"{'Generation Order:':<20} Coarse (Q0) → Medium (Q1) → Fine (Q2)")
        print(f"{'Conditioning:':<20} {conditioning_modality}")
        print(f"{'Encoder Type:':<20} {conditioning_encoder}")
        print(f"{'Conditioning Dim:':<20} {conditioning_dim}")
        print(f"{'Num Layers (Q0):':<20} {self.num_layers_q0}")
        print(f"{'Num Layers (Q1):':<20} {self.num_layers_q1}")
        print(f"{'Num Layers (Q2):':<20} {self.num_layers_q2}")
        print(f"{'Embed Dim:':<20} {embed_dim}")
        print(f"{'Num Heads:':<20} {n_head}")
        if q0_decoder_type == 'llamagen':
            print(f"{'Num KV Heads (Q0):':<20} {self.n_kv_head}")
            print(f"{'Q0 Architecture:':<20} LlamaGen (RMSNorm, GQA+RoPE, SwiGLU, prefix-concat)")
            print(f"{'DropPath Rate (Q0):':<20} {drop_path_rate}")
        else:
            print(f"{'Q0 Architecture:':<20} Transformer Decoder (LayerNorm, MHA, GELU, cross-attention)")
        print(f"{'Block Size:':<20} {block_size}")
        print(f"{'Codebook Size:':<20} {num_vq}")
        print(f"{'EOS Token ID:':<20} {num_vq}")
        print(f"{'PKeep:':<20} {pkeep}")
        print(f"{'Corrupt Ratio:':<20} {corrupt_ratio}" + (" (per-time RVQ augmentation)" if corrupt_ratio > 0 else " (disabled)"))
        print(f"{'='*70}\n")

        # === Three Hierarchical Decoders ===

        # Q0 Decoder: switchable between LlamaGen and standard Transformer Decoder
        if q0_decoder_type == 'llamagen':
            self.q0_decoder = LlamaGenQ0Decoder(
                num_vq=num_vq,
                embed_dim=embed_dim,
                block_size=block_size,
                num_layers=self.num_layers_q0,
                n_head=n_head,
                n_kv_head=self.n_kv_head,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
            )
        elif q0_decoder_type == 'transformer':
            self.q0_decoder = HierarchicalRVQDecoder(
                num_vq=num_vq,
                embed_dim=embed_dim,
                block_size=block_size,
                num_layers=self.num_layers_q0,
                n_head=n_head,
                dropout=dropout,
                quantizer_level=0,
            )
        else:
            raise ValueError(f"Unknown q0_decoder_type: {q0_decoder_type}. Use 'llamagen' or 'transformer'.")

        # Q1 Decoder: Conditioned on Q0 - MEDIUM REFINEMENT
        self.q1_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=self.num_layers_q1,  # Use Q1-specific layer count
            n_head=n_head,
            dropout=dropout,
            quantizer_level=1  # Conditioned on Q0
        )

        # Q2 Decoder: Conditioned on Q0+Q1 - FINE REFINEMENT
        self.q2_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=self.num_layers_q2,  # Use Q2-specific layer count
            n_head=n_head,
            dropout=dropout,
            quantizer_level=2  # Conditioned on Q0+Q1
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        q0_params = sum(p.numel() for p in self.q0_decoder.parameters())
        q1_params = sum(p.numel() for p in self.q1_decoder.parameters())
        q2_params = sum(p.numel() for p in self.q2_decoder.parameters())
        total_params = q0_params + q1_params + q2_params
        print(f"Parameters: Q0={q0_params/1e6:.1f}M, Q1={q1_params/1e6:.1f}M, "
              f"Q2={q2_params/1e6:.1f}M, Total={total_params/1e6:.1f}M\n")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _corrupt_per_time(self, motion_codes: torch.Tensor) -> torch.Tensor:
        """
        Per-time corrupted RVQ augmentation (T2M-HiFiGPT paper Sec 3.4).

        Randomly select corrupt_ratio fraction of timesteps and replace the
        ENTIRE row (all quantizers) with random codes. This forces the decoder
        to learn to recover from corrupted autoregressive inputs, reducing
        exposure bias.

        Args:
            motion_codes: (B, T, num_q) - GT motion token indices

        Returns:
            corrupted_codes: (B, T, num_q) - Corrupted motion codes
        """
        B, T, num_q = motion_codes.shape
        device = motion_codes.device

        # Per-sample, per-timestep mask: True = corrupt this timestep
        corrupt_mask = torch.rand(B, T, device=device) < self.corrupt_ratio  # (B, T)

        # Generate random codes for corrupted positions
        random_codes = torch.randint(0, self.num_vq, (B, T, num_q), device=device)

        # Expand mask to cover all quantizers: (B, T) -> (B, T, num_q)
        corrupt_mask = corrupt_mask.unsqueeze(-1).expand_as(motion_codes)

        # Replace corrupted timesteps with random codes
        corrupted = torch.where(corrupt_mask, random_codes, motion_codes)
        return corrupted

    def forward(
        self,
        motion_codes: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pkeep: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            motion_codes: (B, T, 3) or (B, T, 6) - Motion token indices
                         If shape is (B, T, 6), only first 3 quantizers are used
            text_features: (B, S, text_dim) - Text features
            text_mask: (B, S) - Optional text mask
            pkeep: Override self.pkeep - probability of keeping GT tokens for conditioning
                   1.0 = full teacher forcing, 0.0 = no teacher forcing

        Returns:
            logits_q0: (B, T, num_vq+1) - Logits for Q0
            logits_q1: (B, T, num_vq+1) - Logits for Q1
            logits_q2: (B, T, num_vq+1) - Logits for Q2
        """
        # Use instance default if not overridden
        if pkeep is None:
            pkeep = self.pkeep

        # Extract first 3 quantizers if input has 6
        if motion_codes.dim() == 3 and motion_codes.shape[-1] == 6:
            motion_codes = motion_codes[:, :, :3]  # (B, T, 3)

        # Per-time corrupted RVQ augmentation (training only)
        # Corrupt the INPUT tokens but keep GT targets for loss computation
        if self.training and self.corrupt_ratio > 0:
            input_codes = self._corrupt_per_time(motion_codes)  # corrupted copy
        else:
            input_codes = motion_codes

        # GT targets (always clean, for loss computation)
        target_q0 = motion_codes[:, :, 0]  # (B, T)
        target_q1 = motion_codes[:, :, 1]  # (B, T)
        target_q2 = motion_codes[:, :, 2]  # (B, T)

        # Input tokens (possibly corrupted, fed to decoders)
        motion_q0 = input_codes[:, :, 0]  # (B, T)
        motion_q1 = input_codes[:, :, 1]  # (B, T)
        motion_q2 = input_codes[:, :, 2]  # (B, T)

        B, T = motion_q0.shape
        device = motion_q0.device

        # Same-level scheduled sampling: replace some input tokens with random codes
        # This reduces exposure bias within each decoder's autoregressive input
        if self.training and pkeep < 1.0:
            mask_q0 = torch.bernoulli(pkeep * torch.ones_like(motion_q0, dtype=torch.float)).long()
            motion_q0 = mask_q0 * motion_q0 + (1 - mask_q0) * torch.randint_like(motion_q0, self.num_vq)

            mask_q1 = torch.bernoulli(pkeep * torch.ones_like(motion_q1, dtype=torch.float)).long()
            motion_q1 = mask_q1 * motion_q1 + (1 - mask_q1) * torch.randint_like(motion_q1, self.num_vq)

            mask_q2 = torch.bernoulli(pkeep * torch.ones_like(motion_q2, dtype=torch.float)).long()
            motion_q2 = mask_q2 * motion_q2 + (1 - mask_q2) * torch.randint_like(motion_q2, self.num_vq)

        # Project conditioning features to decoder dimension
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # (B, text_dim) → (B, 1, text_dim)
        if self.text_proj is not None:
            text_context = self.text_proj(text_features)  # (B, S, embed_dim)
        else:
            # Speech mode: features already projected by audio_proj in __call__
            text_context = text_features

        # Hierarchical forward: Q0 → Q1 → Q2

        # 1. Q0 forward
        if self.q0_decoder_type == 'llamagen':
            logits_q0 = self.q0_decoder(
                idx=motion_q0,
                cond_context=text_context,
                cond_mask=text_mask,
            )
        else:
            logits_q0 = self.q0_decoder(
                idx=motion_q0,
                text_context=text_context,
                text_mask=text_mask,
            )

        # Determine Q0 tokens for conditioning Q1
        # pkeep controls the probability of using GT vs predicted tokens
        # Note: always use clean GT (target_q0) for the GT branch, not corrupted input
        if self.training and pkeep < 1.0:
            # Scheduled sampling: randomly mix GT and predicted tokens (per-sample)
            pred_q0 = logits_q0.argmax(dim=-1)  # (B, T)
            mask = torch.rand(B, device=device) < pkeep  # (B,)
            mask = mask.unsqueeze(1).expand(-1, T)  # (B, T)
            q0_for_conditioning = torch.where(mask, target_q0, pred_q0)
        else:
            # Full teacher forcing (pkeep=1.0): use GT tokens
            q0_for_conditioning = target_q0

        q0_embeddings = self.q0_decoder.get_embeddings(q0_for_conditioning)

        # 2. Q1 forward (conditioned on Q0, optionally on speech/text)
        logits_q1 = self.q1_decoder(
            idx=motion_q1,
            text_context=text_context if self.q1_use_cond else None,
            text_mask=text_mask if self.q1_use_cond else None,
            prev_quantizers_context=q0_embeddings
        )

        # Determine Q1 tokens for conditioning Q2
        # pkeep controls the probability of using GT vs predicted tokens
        # Note: always use clean GT (target_q1) for the GT branch, not corrupted input
        if self.training and pkeep < 1.0:
            # Scheduled sampling: randomly mix GT and predicted tokens (per-sample)
            pred_q1 = logits_q1.argmax(dim=-1)  # (B, T)
            mask = torch.rand(B, device=device) < pkeep  # (B,)
            mask = mask.unsqueeze(1).expand(-1, T)  # (B, T)
            q1_for_conditioning = torch.where(mask, target_q1, pred_q1)
        else:
            # Full teacher forcing (pkeep=1.0): use GT tokens
            q1_for_conditioning = target_q1

        q1_embeddings = self.q1_decoder.get_embeddings(q1_for_conditioning)

        # 3. Q2 forward (conditioned on Q0+Q1, optionally on speech/text)
        q0_q1_embeddings = torch.cat([q0_embeddings, q1_embeddings], dim=1)
        logits_q2 = self.q2_decoder(
            idx=motion_q2,
            text_context=text_context if self.q2_use_cond else None,
            text_mask=text_mask if self.q2_use_cond else None,
            prev_quantizers_context=q0_q1_embeddings
        )

        return logits_q0, logits_q1, logits_q2

    @torch.no_grad()
    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        audio_waveforms: Optional[torch.Tensor] = None,
        max_len: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        High-level generation method that handles both text and speech input.

        Args:
            texts: List[str] - Text descriptions (for text mode)
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz (for speech mode)
            max_len: Maximum generation length
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated_codes: (B, actual_len, 3) - Generated Q0, Q1, Q2 codes (EOS tokens removed)
            lengths: List[int] - Actual generated lengths per sample
        """
        # Encode conditioning modality
        if self.use_speech:
            if audio_waveforms is None:
                raise ValueError("audio_waveforms must be provided in speech mode")
            audio_features, audio_mask = self.encode_audio(audio_waveforms)
            conditioning_features = self.project_audio(audio_features, audio_mask)
            conditioning_mask = audio_mask
        else:
            if texts is None:
                raise ValueError("texts must be provided in text mode")
            text_features, text_mask = self.encode_text(texts)
            conditioning_features = text_features
            conditioning_mask = text_mask

        # Call the low-level generate method
        return self.generate(conditioning_features, conditioning_mask, max_len, do_sample, temperature, top_k, top_p)

    @torch.no_grad()
    def generate(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Low-level autoregressive generation with KV cache for Q0 and hierarchical conditioning.

        Q0 uses LlamaGen-style KV cache for efficient generation:
        1. Prefill: process conditioning prefix once, cache K/V
        2. Decode: generate one Q0 token at a time using cached K/V

        Q1/Q2 remain unchanged (full recomputation each step).

        Args:
            text_features: (B, S, text_dim) - Text features
            text_mask: (B, S) - Optional text mask
            max_len: Maximum generation length
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated_codes: (B, actual_len, 3) - Generated Q0, Q1, Q2 codes (EOS tokens removed)
            lengths: List[int] - Actual generated lengths per sample
        """
        B = text_features.shape[0]
        device = text_features.device
        dtype = text_features.dtype

        # Project conditioning features
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if self.text_proj is not None:
            text_context = self.text_proj(text_features)
        else:
            # Speech mode: features already projected by audio_proj in generate_conditional
            text_context = text_features

        # Clamp max_len to block_size (following SOKE's approach)
        max_len = min(max_len, self.block_size)

        # Initialize sequences
        q0_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q1_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q2_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)

        # Track which samples have finished (EOS predicted)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        if self.q0_decoder_type == 'llamagen':
            # === LlamaGen: KV cache inference ===
            cls_token_num = text_context.shape[1]
            self.q0_decoder.setup_caches(B, cls_token_num + max_len, dtype)

            # Prefill: process conditioning prefix, cache K/V
            cond_input_pos = torch.arange(cls_token_num, device=device)
            q0_prefill_logits = self.q0_decoder.generate_prefill(
                cond_context=text_context,
                input_pos=cond_input_pos,
                cond_mask=text_mask,
            )
            logits_q0_t = q0_prefill_logits[:, -1, :] / temperature
            if do_sample:
                q0_t = self._sample(logits_q0_t, top_k=top_k, top_p=top_p)
            else:
                q0_t = logits_q0_t.argmax(dim=-1)

            for t in range(max_len):
                if t > 0:
                    input_pos = torch.tensor([cls_token_num + t], device=device)
                    q0_logits = self.q0_decoder.generate_one_token(
                        idx=q0_t, input_pos=input_pos,
                        cond_mask=text_mask, cls_token_num=cls_token_num,
                    )
                    logits_q0_t = q0_logits[:, -1, :] / temperature
                    if do_sample:
                        q0_t = self._sample(logits_q0_t, top_k=top_k, top_p=top_p)
                    else:
                        q0_t = logits_q0_t.argmax(dim=-1)
                else:
                    input_pos = torch.tensor([cls_token_num], device=device)
                    self.q0_decoder.generate_one_token(
                        idx=q0_t, input_pos=input_pos,
                        cond_mask=text_mask, cls_token_num=cls_token_num,
                    )

                # Check EOS
                eos_mask = (q0_t == self.eos_token_id) & ~finished
                lengths[eos_mask] = t
                finished = finished | eos_mask
                if finished.all():
                    break

                q0_t = torch.where(finished, torch.zeros_like(q0_t), q0_t)
                q0_tokens = torch.cat([q0_tokens, q0_t.unsqueeze(1)], dim=1)
                q0_emb = self.q0_decoder.get_embeddings(q0_tokens)

                # Q1
                logits_q1 = self.q1_decoder(
                    idx=q1_tokens if q1_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                    text_context=text_context if self.q1_use_cond else None,
                    text_mask=text_mask if self.q1_use_cond else None,
                    prev_quantizers_context=q0_emb,
                )
                logits_q1_t = logits_q1[:, -1, :] / temperature
                q1_t = self._sample(logits_q1_t, top_k=top_k, top_p=top_p) if do_sample else logits_q1_t.argmax(dim=-1)
                q1_t = torch.where(finished, torch.zeros_like(q1_t), q1_t)
                q1_tokens = torch.cat([q1_tokens, q1_t.unsqueeze(1)], dim=1)
                q1_emb = self.q1_decoder.get_embeddings(q1_tokens)

                # Q2
                q0_q1_emb = torch.cat([q0_emb, q1_emb], dim=1)
                logits_q2 = self.q2_decoder(
                    idx=q2_tokens if q2_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                    text_context=text_context if self.q2_use_cond else None,
                    text_mask=text_mask if self.q2_use_cond else None,
                    prev_quantizers_context=q0_q1_emb,
                )
                logits_q2_t = logits_q2[:, -1, :] / temperature
                q2_t = self._sample(logits_q2_t, top_k=top_k, top_p=top_p) if do_sample else logits_q2_t.argmax(dim=-1)
                q2_t = torch.where(finished, torch.zeros_like(q2_t), q2_t)
                q2_tokens = torch.cat([q2_tokens, q2_t.unsqueeze(1)], dim=1)

            self.q0_decoder.clear_caches()

        else:
            # === Transformer Decoder: full-forward inference (same as Q1/Q2) ===
            for t in range(max_len):
                # Q0: full forward each step (no KV cache)
                q0_input = q0_tokens if q0_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device)
                logits_q0 = self.q0_decoder(
                    idx=q0_input,
                    text_context=text_context,
                    text_mask=text_mask,
                )
                logits_q0_t = logits_q0[:, -1, :] / temperature
                q0_t = self._sample(logits_q0_t, top_k=top_k, top_p=top_p) if do_sample else logits_q0_t.argmax(dim=-1)

                # Check EOS
                eos_mask = (q0_t == self.eos_token_id) & ~finished
                lengths[eos_mask] = t
                finished = finished | eos_mask
                if finished.all():
                    break

                q0_t = torch.where(finished, torch.zeros_like(q0_t), q0_t)
                q0_tokens = torch.cat([q0_tokens, q0_t.unsqueeze(1)], dim=1)
                q0_emb = self.q0_decoder.get_embeddings(q0_tokens)

                # Q1
                logits_q1 = self.q1_decoder(
                    idx=q1_tokens if q1_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                    text_context=text_context if self.q1_use_cond else None,
                    text_mask=text_mask if self.q1_use_cond else None,
                    prev_quantizers_context=q0_emb,
                )
                logits_q1_t = logits_q1[:, -1, :] / temperature
                q1_t = self._sample(logits_q1_t, top_k=top_k, top_p=top_p) if do_sample else logits_q1_t.argmax(dim=-1)
                q1_t = torch.where(finished, torch.zeros_like(q1_t), q1_t)
                q1_tokens = torch.cat([q1_tokens, q1_t.unsqueeze(1)], dim=1)
                q1_emb = self.q1_decoder.get_embeddings(q1_tokens)

                # Q2
                q0_q1_emb = torch.cat([q0_emb, q1_emb], dim=1)
                logits_q2 = self.q2_decoder(
                    idx=q2_tokens if q2_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                    text_context=text_context if self.q2_use_cond else None,
                    text_mask=text_mask if self.q2_use_cond else None,
                    prev_quantizers_context=q0_q1_emb,
                )
                logits_q2_t = logits_q2[:, -1, :] / temperature
                q2_t = self._sample(logits_q2_t, top_k=top_k, top_p=top_p) if do_sample else logits_q2_t.argmax(dim=-1)
                q2_t = torch.where(finished, torch.zeros_like(q2_t), q2_t)
                q2_tokens = torch.cat([q2_tokens, q2_t.unsqueeze(1)], dim=1)

        # Set lengths for samples that didn't hit EOS (reached max_len)
        lengths[~finished] = q0_tokens.shape[1]

        # Stack all quantizers: (B, T, 3)
        if q0_tokens.shape[1] > 0:
            generated_codes = torch.stack([q0_tokens, q1_tokens, q2_tokens], dim=-1)
        else:
            # Edge case: no tokens generated
            generated_codes = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
            lengths = torch.ones(B, dtype=torch.long, device=device)

        return generated_codes, lengths.tolist()

    def _sample(self, logits, top_k=None, top_p=None):
        """
        Sample from logits with optional top-k and nucleus filtering.

        Args:
            logits: (B, vocab_size)
            top_k: Keep top k tokens
            top_p: Keep top tokens with cumulative probability >= p

        Returns:
            sampled: (B,) - Sampled token indices
        """
        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Nucleus (top-p) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)

        return sampled

    # ========================================================================
    # Compatibility methods for MotionGPT training interface
    # ========================================================================

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text using the configured text encoder (CLIP/BERT).

        Args:
            texts: List[str] - Text descriptions

        Returns:
            text_features: (B, seq_len, text_dim) - Token-level text features (BERT) or (B, 1, text_dim) (CLIP)
            attention_mask: (B, seq_len) - Attention mask for padding tokens, or None for CLIP
        """
        if self.text_encoder is None:
            raise ValueError("Text encoder not initialized. Set use_speech=False to use text mode.")
        return self.text_encoder(texts)

    def encode_audio(self, audio_waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio waveforms using the configured speech encoder (HuBERT/WavLM/Whisper).

        Args:
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz

        Returns:
            audio_features: (B, seq_len, audio_dim) - Frame-level audio features
            attention_mask: (B, seq_len) - Attention mask for valid frames
        """
        if self.speech_encoder is None:
            raise ValueError("Speech encoder not initialized. Set use_speech=True to use speech mode.")
        return self.speech_encoder(audio_waveforms)

    def project_audio(self, audio_features: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to decoder dimension, optionally through contrastive projection head.

        Args:
            audio_features: (B, seq_len, audio_dim) from encode_audio
            audio_mask: (B, seq_len) attention mask

        Returns:
            conditioning_features: (B, seq_len, embed_dim)
        """
        if self.use_contrastive_proj:
            x = self.contrastive_input_proj(audio_features)
            x = self.contrastive_pos_enc(x)
            pad_mask = ~audio_mask.bool() if audio_mask is not None else None
            x = self.contrastive_transformer(x, src_key_padding_mask=pad_mask)
            return self.audio_proj(x)
        else:
            return self.audio_proj(audio_features)

    def __call__(self, texts=None, audio_waveforms=None, motion_tokens=None, lengths=None, tasks=None,
                 speech_feats=None, speech_mask=None):
        """
        Training forward pass compatible with MotionGPT interface.
        This method is called by train_lm_forward() in mgpt.py.

        Args:
            texts: List[str] - Text descriptions (for text mode)
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz (for speech mode)
            motion_tokens: (B, T, num_quantizers) - Ground truth motion codes
            lengths: List[int] - Sequence lengths (in tokens, not frames)
            tasks: Dict - Task information (not used)
            speech_feats: (B, T_s, D) - Precomputed speech encoder features (skip encode_audio)
            speech_mask: (B, T_s) - Mask for precomputed speech features

        Returns:
            Object with .loss attribute containing the total cross-entropy loss
        """
        # Encode conditioning modality (text or speech)
        if self.use_speech:
            if speech_feats is not None:
                # Precomputed speech features: skip encode_audio, go directly to project_audio
                conditioning_features = self.project_audio(speech_feats, speech_mask)
                conditioning_mask = speech_mask
            elif audio_waveforms is not None:
                # Raw audio waveform mode
                audio_features, audio_mask = self.encode_audio(audio_waveforms)
                conditioning_features = self.project_audio(audio_features, audio_mask)
                conditioning_mask = audio_mask
            else:
                raise ValueError("speech_feats or audio_waveforms must be provided in speech mode")
        else:
            # Text mode (original)
            if texts is None:
                raise ValueError("texts must be provided in text mode")
            text_features, text_mask = self.encode_text(texts)  # (B, seq_len, text_dim), (B, seq_len) or None
            conditioning_features = text_features
            conditioning_mask = text_mask

        # Extract first 3 quantizers if input has 6
        if motion_tokens.dim() == 3 and motion_tokens.shape[-1] == 6:
            motion_tokens = motion_tokens[:, :, :3]  # (B, T, 3)

        B, T, _ = motion_tokens.shape
        device = motion_tokens.device

        # Append EOS token at the end of each sequence based on actual lengths
        # lengths are in tokens (not frames)
        # Create targets with EOS at the correct position
        # Target: [tok_1, tok_2, ..., tok_L, EOS, PAD, PAD, ...]

        # Create padded targets with EOS
        targets_q0 = torch.full((B, T), -1, dtype=torch.long, device=device)  # -1 for ignore
        targets_q1 = torch.full((B, T), -1, dtype=torch.long, device=device)
        targets_q2 = torch.full((B, T), -1, dtype=torch.long, device=device)

        for i in range(B):
            seq_len = min(lengths[i], T - 1) if lengths is not None else T - 1
            # Copy GT tokens shifted by 1 (next token prediction)
            targets_q0[i, :seq_len] = motion_tokens[i, 1:seq_len+1, 0]
            targets_q1[i, :seq_len] = motion_tokens[i, 1:seq_len+1, 1]
            targets_q2[i, :seq_len] = motion_tokens[i, 1:seq_len+1, 2]
            # Add EOS token at the end of valid sequence
            if seq_len < T:
                targets_q0[i, seq_len] = self.eos_token_id
                targets_q1[i, seq_len] = self.eos_token_id
                targets_q2[i, seq_len] = self.eos_token_id

        # Call the main forward method with conditioning features
        logits_q0, logits_q1, logits_q2 = self.forward(motion_tokens, conditioning_features, text_mask=conditioning_mask)

        # Logits are for positions [0:T-1] predicting [1:T]
        logits_q0 = logits_q0[:, :-1, :]  # (B, T-1, num_vq+1)
        logits_q1 = logits_q1[:, :-1, :]  # (B, T-1, num_vq+1)
        logits_q2 = logits_q2[:, :-1, :]  # (B, T-1, num_vq+1)

        # Trim targets to match logits
        targets_q0 = targets_q0[:, :T-1]
        targets_q1 = targets_q1[:, :T-1]
        targets_q2 = targets_q2[:, :T-1]

        # Compute cross-entropy loss for each quantizer
        loss_q0 = F.cross_entropy(
            logits_q0.reshape(-1, logits_q0.size(-1)),
            targets_q0.reshape(-1),
            ignore_index=-1  # Ignore padding
        )
        loss_q1 = F.cross_entropy(
            logits_q1.reshape(-1, logits_q1.size(-1)),
            targets_q1.reshape(-1),
            ignore_index=-1
        )
        loss_q2 = F.cross_entropy(
            logits_q2.reshape(-1, logits_q2.size(-1)),
            targets_q2.reshape(-1),
            ignore_index=-1
        )

        # Total loss (weighted: Q0 most important, Q2 least)
        # total_loss = loss_q0 + 0.5 * loss_q1 + 0.25 * loss_q2

        # Equal weighting
        total_loss = loss_q0 + loss_q1 + loss_q2

        # Compute token accuracy for each quantizer
        with torch.no_grad():
            # Get predictions
            pred_q0 = logits_q0.argmax(dim=-1)  # (B, T-1)
            pred_q1 = logits_q1.argmax(dim=-1)
            pred_q2 = logits_q2.argmax(dim=-1)

            # Create mask for valid positions (not -1)
            mask_q0 = targets_q0 != -1
            mask_q1 = targets_q1 != -1
            mask_q2 = targets_q2 != -1

            # Compute accuracy (only on valid positions)
            acc_q0 = (pred_q0[mask_q0] == targets_q0[mask_q0]).float().mean() if mask_q0.any() else torch.tensor(0.0, device=device)
            acc_q1 = (pred_q1[mask_q1] == targets_q1[mask_q1]).float().mean() if mask_q1.any() else torch.tensor(0.0, device=device)
            acc_q2 = (pred_q2[mask_q2] == targets_q2[mask_q2]).float().mean() if mask_q2.any() else torch.tensor(0.0, device=device)

        # Return object with .loss, per-decoder losses, and accuracies for logging
        class LossOutput:
            def __init__(self, loss, loss_q0, loss_q1, loss_q2, acc_q0, acc_q1, acc_q2):
                self.loss = loss
                self.loss_q0 = loss_q0
                self.loss_q1 = loss_q1
                self.loss_q2 = loss_q2
                self.acc_q0 = acc_q0
                self.acc_q1 = acc_q1
                self.acc_q2 = acc_q2

        return LossOutput(total_loss, loss_q0, loss_q1, loss_q2, acc_q0, acc_q1, acc_q2)

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None,
                             audio_waveforms=None, **kwargs):
        """
        Generation method compatible with MotionGPT interface.
        This method is called by val_t2m_forward() in mgpt.py.

        Args:
            texts: List[str] - Text descriptions (for text mode)
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz (for speech mode)
            lengths: List[int] or Tensor - Target generation lengths (used as max_len hint)
            stage: str - 'test' or 'val'
            tasks: Dict - Task information (not used)

        Returns:
            List[Tensor] - List of (T_i, 3) tensors, one per sample (variable length due to EOS)
        """
        # Encode conditioning modality (text or speech)
        if self.use_speech:
            if audio_waveforms is None:
                raise ValueError("audio_waveforms must be provided in speech mode")
            audio_features, audio_mask = self.encode_audio(audio_waveforms)
            conditioning_features = self.project_audio(audio_features, audio_mask)
            conditioning_mask = audio_mask
        else:
            if texts is None:
                raise ValueError("texts must be provided in text mode")
            text_features, text_mask = self.encode_text(texts)
            conditioning_features = text_features
            conditioning_mask = text_mask

        # Determine max generation length
        # NOTE: lengths are in frames, need to convert to tokens (divide by unit_length=4)
        if lengths is not None:
            if isinstance(lengths, list):
                max_len = max(lengths) // 4  # Convert frames to tokens
            else:
                max_len = int(lengths.max().item()) // 4  # Convert frames to tokens
        else:
            max_len = self.block_size

        # Clamp to block_size to avoid exceeding positional embedding range
        max_len = min(max_len, self.block_size)

        # Generate using the main generate method
        generated_codes, gen_lengths = self.generate(
            text_features=conditioning_features,
            text_mask=conditioning_mask,
            max_len=max_len,
            do_sample=True,
            temperature=1.0
        )

        # Convert (B, T, 3) tensor to list of (T_i, 3) tensors
        # Trim each sample to its actual length (determined by EOS or max_len)
        outputs_tokens = []
        for i in range(generated_codes.shape[0]):
            actual_len = gen_lengths[i]
            if actual_len > 0:
                outputs_tokens.append(generated_codes[i, :actual_len, :])
            else:
                # Edge case: at least 1 token
                outputs_tokens.append(generated_codes[i, :1, :])

        return outputs_tokens


class HierarchicalRVQGPT_6_layer(nn.Module):
    """
    Hierarchical RVQ-GPT with 6 decoders (Q0 → Q1 → Q2 → Q3 → Q4 → Q5).

    Supports two conditioning modes (controlled by `conditioning_mode` parameter):

    "chain" mode - each decoder depends on the LAST quantizer + text only:
    1. Q0 decoder: P(Q0 | text)
    2. Q1 decoder: P(Q1 | Q0, text)
    3. Q2 decoder: P(Q2 | Q1, text)
    4. Q3 decoder: P(Q3 | Q2, text)
    5. Q4 decoder: P(Q4 | Q3, text)
    6. Q5 decoder: P(Q5 | Q4, text)

    "full" mode - each decoder depends on ALL previous quantizers + text:
    1. Q0 decoder: P(Q0 | text)
    2. Q1 decoder: P(Q1 | Q0, text)
    3. Q2 decoder: P(Q2 | Q0, Q1, text)
    4. Q3 decoder: P(Q3 | Q0, Q1, Q2, text)
    5. Q4 decoder: P(Q4 | Q0, Q1, Q2, Q3, text)
    6. Q5 decoder: P(Q5 | Q0, Q1, Q2, Q3, Q4, text)
    """

    def __init__(
        self,
        num_vq=512,
        embed_dim=1024,
        block_size=200,
        num_layers=9,
        num_layers_q0=None,  # Per-decoder layer counts (default to num_layers)
        num_layers_q1=None,
        num_layers_q2=None,
        num_layers_q3=None,
        num_layers_q4=None,
        num_layers_q5=None,
        n_head=16,
        n_kv_head=None,  # GQA KV heads for LlamaGen Q0 (default = n_head = standard MHA)
        drop_path_rate=0.0,  # DropPath rate for LlamaGen Q0
        dropout=0.1,
        text_dim=512,
        text_encoder_type='clip',  # 'clip', 'bert', or 'bert-large'
        pkeep=1.0,
        conditioning_mode='chain',  # 'chain' or 'full'
        q0_decoder_type='transformer',  # 'transformer' or 'llamagen'
        corrupt_ratio=0.0,  # Per-time RVQ corruption fraction (T2M-HiFiGPT Sec 3.4)
        use_speech=False,  # Accepted for compatibility with MotionGPT wrapper; must be False
        speech_encoder_type=None,  # Accepted for compatibility; not supported here
    ):
        super().__init__()

        assert conditioning_mode in ['chain', 'full'], f"conditioning_mode must be 'chain' or 'full', got {conditioning_mode}"
        assert q0_decoder_type in ['transformer', 'llamagen'], f"q0_decoder_type must be 'transformer' or 'llamagen', got {q0_decoder_type}"
        assert not use_speech, "HierarchicalRVQGPT_6_layer is text-only; set condition='text' in the top-level config."

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_layers = num_layers
        # Per-decoder layer counts (default to num_layers if not specified)
        self.num_layers_per_q = [
            num_layers_q0 if num_layers_q0 is not None else num_layers,
            num_layers_q1 if num_layers_q1 is not None else num_layers,
            num_layers_q2 if num_layers_q2 is not None else num_layers,
            num_layers_q3 if num_layers_q3 is not None else num_layers,
            num_layers_q4 if num_layers_q4 is not None else num_layers,
            num_layers_q5 if num_layers_q5 is not None else num_layers,
        ]
        self.num_quantizers_used = 6  # Use all 6 quantizers
        self.pkeep = pkeep
        self.eos_token_id = num_vq
        self.conditioning_mode = conditioning_mode
        self.text_encoder_type = text_encoder_type
        self.q0_decoder_type = q0_decoder_type
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.drop_path_rate = drop_path_rate
        self.corrupt_ratio = corrupt_ratio

        # Initialize text encoder
        self.text_encoder = TextEncoder(encoder_type=text_encoder_type, freeze=True)
        # Override text_dim with actual encoder output dim
        text_dim = self.text_encoder.output_dim

        print(f"\n{'='*70}")
        print(f"{'Hierarchical RVQ-GPT (6-Layer)':^70}")
        print(f"{'='*70}")
        print(f"{'Architecture:':<20} Q0 → Q1 → Q2 → Q3 → Q4 → Q5")
        print(f"{'Conditioning Mode:':<20} {conditioning_mode}")
        if conditioning_mode == 'chain':
            print(f"{'Conditioning:':<20} Each Qi depends on Q(i-1) + text only")
        else:
            print(f"{'Conditioning:':<20} Each Qi depends on Q0..Q(i-1) + text")
        print(f"{'Quantizers Used:':<20} 6 (all from RVQ-VAE)")
        print(f"{'Num Layers:':<20} {num_layers} (default)")
        print(f"{'Layers per Q:':<20} {self.num_layers_per_q}")
        print(f"{'Embed Dim:':<20} {embed_dim}")
        print(f"{'Num Heads:':<20} {n_head}")
        if q0_decoder_type == 'llamagen':
            print(f"{'Num KV Heads (Q0):':<20} {self.n_kv_head}")
            print(f"{'Q0 Architecture:':<20} LlamaGen (RMSNorm, GQA+RoPE, SwiGLU, prefix-concat)")
            print(f"{'DropPath Rate (Q0):':<20} {drop_path_rate}")
        else:
            print(f"{'Q0 Architecture:':<20} Transformer Decoder (LayerNorm, MHA, GELU, cross-attention)")
        print(f"{'Block Size:':<20} {block_size}")
        print(f"{'Codebook Size:':<20} {num_vq}")
        print(f"{'Text Encoder:':<20} {text_encoder_type}")
        print(f"{'Text Dim:':<20} {text_dim}D")
        print(f"{'Corrupt Ratio:':<20} {corrupt_ratio}" + (" (per-time RVQ augmentation)" if corrupt_ratio > 0 else " (disabled)"))
        print(f"{'='*70}\n")

        # Text projection
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 6 Hierarchical Decoders
        # Q0: text-only conditioning (level=0) — switchable between LlamaGen and Transformer
        # Q1+: conditioning depends on mode
        #   - chain mode: level=1 (text + prev Q only)
        #   - full mode: level=2 for Q2+ (text + all prev Qs)
        if conditioning_mode == 'chain':
            decoder_levels = [min(i, 1) for i in range(6)]  # [0, 1, 1, 1, 1, 1]
        else:  # full mode
            decoder_levels = [min(i, 2) for i in range(6)]  # [0, 1, 2, 2, 2, 2]

        # Build Q0 separately (may be LlamaGen), Q1..Q5 are always HierarchicalRVQDecoder
        if q0_decoder_type == 'llamagen':
            q0_decoder = LlamaGenQ0Decoder(
                num_vq=num_vq,
                embed_dim=embed_dim,
                block_size=block_size,
                num_layers=self.num_layers_per_q[0],
                n_head=n_head,
                n_kv_head=self.n_kv_head,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
            )
        else:
            q0_decoder = HierarchicalRVQDecoder(
                num_vq=num_vq,
                embed_dim=embed_dim,
                block_size=block_size,
                num_layers=self.num_layers_per_q[0],
                n_head=n_head,
                dropout=dropout,
                quantizer_level=0,
            )

        refinement_decoders = [
            HierarchicalRVQDecoder(
                num_vq=num_vq,
                embed_dim=embed_dim,
                block_size=block_size,
                num_layers=self.num_layers_per_q[i],
                n_head=n_head,
                dropout=dropout,
                quantizer_level=decoder_levels[i],
            )
            for i in range(1, 6)
        ]
        self.decoders = nn.ModuleList([q0_decoder] + refinement_decoders)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params/1e6:.1f}M\n")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _corrupt_per_time(self, motion_codes: torch.Tensor) -> torch.Tensor:
        """Per-time corrupted RVQ augmentation (T2M-HiFiGPT Sec 3.4).

        Randomly select corrupt_ratio fraction of timesteps and replace the
        entire row (all quantizers) with random codes.
        """
        B, T, num_q = motion_codes.shape
        device = motion_codes.device
        corrupt_mask = torch.rand(B, T, device=device) < self.corrupt_ratio
        random_codes = torch.randint(0, self.num_vq, (B, T, num_q), device=device)
        corrupt_mask = corrupt_mask.unsqueeze(-1).expand_as(motion_codes)
        return torch.where(corrupt_mask, random_codes, motion_codes)

    def _q0_forward(self, idx, text_context, text_mask):
        """Q0 forward pass dispatching on decoder type."""
        if self.q0_decoder_type == 'llamagen':
            return self.decoders[0](
                idx=idx,
                cond_context=text_context,
                cond_mask=text_mask,
            )
        return self.decoders[0](
            idx=idx,
            text_context=text_context,
            text_mask=text_mask,
        )

    def forward(
        self,
        motion_codes: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pkeep: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for training.

        Args:
            motion_codes: (B, T, 6) - Motion token indices for all 6 quantizers
            text_features: (B, S, text_dim) - Text features
            text_mask: (B, S) - Optional text mask
            pkeep: Override self.pkeep

        Returns:
            Tuple of 6 logits tensors: (logits_q0, logits_q1, ..., logits_q5)
        """
        if pkeep is None:
            pkeep = self.pkeep

        # Per-time corrupted RVQ augmentation: corrupt INPUT, keep GT targets clean for loss
        if self.training and self.corrupt_ratio > 0:
            input_codes = self._corrupt_per_time(motion_codes)
        else:
            input_codes = motion_codes

        B, T, num_q = motion_codes.shape
        device = motion_codes.device

        # Project text
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        text_context = self.text_proj(text_features)

        all_logits = []
        prev_embeddings = None

        for q_idx in range(6):
            motion_q = input_codes[:, :, q_idx]        # fed to decoder (possibly corrupted)
            target_q = motion_codes[:, :, q_idx]       # clean GT for next-decoder conditioning

            # Forward through decoder
            if q_idx == 0:
                logits_q = self._q0_forward(motion_q, text_context, text_mask)
            else:
                logits_q = self.decoders[q_idx](
                    idx=motion_q,
                    text_context=text_context,
                    text_mask=text_mask,
                    prev_quantizers_context=prev_embeddings,
                )
            all_logits.append(logits_q)

            # Prepare embeddings for next decoder (scheduled sampling on clean GT)
            if self.training and pkeep < 1.0:
                pred_q = logits_q.argmax(dim=-1)
                mask = torch.rand(B, device=device) < pkeep
                mask = mask.unsqueeze(1).expand(-1, T)
                q_for_conditioning = torch.where(mask, target_q, pred_q)
            else:
                q_for_conditioning = target_q

            q_embeddings = self.decoders[q_idx].get_embeddings(q_for_conditioning)

            # Update prev_embeddings based on conditioning mode
            if self.conditioning_mode == 'chain':
                prev_embeddings = q_embeddings
            else:  # full mode: accumulate all previous quantizers' embeddings
                if prev_embeddings is None:
                    prev_embeddings = q_embeddings
                else:
                    prev_embeddings = torch.cat([prev_embeddings, q_embeddings], dim=1)

        return tuple(all_logits)

    @torch.no_grad()
    def generate(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Autoregressive generation with configurable conditioning mode.

        Returns:
            generated_codes: (B, actual_len, 6)
            lengths: List[int] - Actual lengths per sample
        """
        B = text_features.shape[0]
        device = text_features.device

        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        text_context = self.text_proj(text_features)

        max_len = min(max_len, self.block_size)

        # Initialize sequences for all 6 quantizers
        q_tokens = [torch.zeros((B, 0), dtype=torch.long, device=device) for _ in range(6)]

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(max_len):
            prev_embeddings = None

            for q_idx in range(6):
                # Build context based on conditioning mode
                if q_idx > 0:
                    if self.conditioning_mode == 'chain':
                        # Chain: only use previous quantizer's embeddings
                        prev_embeddings = self.decoders[q_idx - 1].get_embeddings(q_tokens[q_idx - 1])
                    else:  # full mode
                        # Full: use all previous quantizers' embeddings
                        emb_list = []
                        for prev_q in range(q_idx):
                            emb = self.decoders[prev_q].get_embeddings(q_tokens[prev_q])
                            emb_list.append(emb)
                        prev_embeddings = torch.cat(emb_list, dim=1)

                # Generate token for this quantizer
                q_idx_input = q_tokens[q_idx] if q_tokens[q_idx].shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device)
                if q_idx == 0 and self.q0_decoder_type == 'llamagen':
                    # LlamaGen Q0: full-forward each step (no KV cache in this path for simplicity)
                    logits = self.decoders[0](
                        idx=q_idx_input,
                        cond_context=text_context,
                        cond_mask=text_mask,
                    )
                else:
                    logits = self.decoders[q_idx](
                        idx=q_idx_input,
                        text_context=text_context,
                        text_mask=text_mask,
                        prev_quantizers_context=prev_embeddings,
                    )
                logits_t = logits[:, -1, :] / temperature

                if do_sample:
                    q_t = self._sample(logits_t, top_k=top_k, top_p=top_p)
                else:
                    q_t = logits_t.argmax(dim=-1)

                # Check EOS only for Q0 (primary stopping criterion)
                if q_idx == 0:
                    eos_mask = (q_t == self.eos_token_id) & ~finished
                    lengths[eos_mask] = t
                    finished = finished | eos_mask

                # Replace with placeholder for finished samples
                q_t = torch.where(finished, torch.zeros_like(q_t), q_t)
                q_tokens[q_idx] = torch.cat([q_tokens[q_idx], q_t.unsqueeze(1)], dim=1)

            if finished.all():
                break

        lengths[~finished] = q_tokens[0].shape[1]

        # Stack all quantizers
        if q_tokens[0].shape[1] > 0:
            generated_codes = torch.stack(q_tokens, dim=-1)
        else:
            generated_codes = torch.zeros((B, 1, 6), dtype=torch.long, device=device)
            lengths = torch.ones(B, dtype=torch.long, device=device)

        return generated_codes, lengths.tolist()

    def _sample(self, logits, top_k=None, top_p=None):
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode text using the configured text encoder (CLIP/BERT)."""
        return self.text_encoder(texts)

    def __call__(self, texts, motion_tokens, lengths, tasks=None,
                 audio_waveforms=None, speech_feats=None, speech_mask=None):
        """Training forward pass compatible with MotionGPT interface (text-only; speech kwargs ignored)."""
        assert audio_waveforms is None and speech_feats is None, (
            "HierarchicalRVQGPT_6_layer is text-only; got speech inputs."
        )
        text_features, text_mask = self.encode_text(texts)

        B, T, _ = motion_tokens.shape
        device = motion_tokens.device

        # Create targets with EOS
        targets = [torch.full((B, T), -1, dtype=torch.long, device=device) for _ in range(6)]

        for i in range(B):
            seq_len = min(lengths[i], T - 1) if lengths is not None else T - 1
            for q_idx in range(6):
                targets[q_idx][i, :seq_len] = motion_tokens[i, 1:seq_len+1, q_idx]
                if seq_len < T:
                    targets[q_idx][i, seq_len] = self.eos_token_id

        all_logits = self.forward(motion_tokens, text_features, text_mask=text_mask)

        # Compute loss with decreasing weights: Q0=1.0, Q1=0.5, Q2=0.25, ...
        total_loss = 0.0
        losses = []
        accuracies = []
        weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]

        for q_idx in range(6):
            logits = all_logits[q_idx][:, :-1, :]
            target = targets[q_idx][:, :T-1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-1
            )
            losses.append(loss)
            total_loss = total_loss + weights[q_idx] * loss

            # Compute token accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = target != -1
                acc = (pred[mask] == target[mask]).float().mean() if mask.any() else torch.tensor(0.0, device=device)
                accuracies.append(acc)

        class LossOutput:
            def __init__(self, loss, losses, accuracies):
                self.loss = loss
                for i, l in enumerate(losses):
                    setattr(self, f'loss_q{i}', l)
                for i, a in enumerate(accuracies):
                    setattr(self, f'acc_q{i}', a)

        return LossOutput(total_loss, losses, accuracies)

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None, **kwargs):
        """Generation method compatible with MotionGPT interface."""
        text_features, text_mask = self.encode_text(texts)

        if lengths is not None:
            if isinstance(lengths, list):
                max_len = max(lengths) // 4
            else:
                max_len = int(lengths.max().item()) // 4
        else:
            max_len = self.block_size

        max_len = min(max_len, self.block_size)

        generated_codes, gen_lengths = self.generate(
            text_features=text_features,
            text_mask=text_mask,
            max_len=max_len,
            do_sample=True,
            temperature=1.0
        )

        outputs_tokens = []
        for i in range(generated_codes.shape[0]):
            actual_len = gen_lengths[i]
            if actual_len > 0:
                outputs_tokens.append(generated_codes[i, :actual_len, :])
            else:
                outputs_tokens.append(generated_codes[i, :1, :])

        return outputs_tokens


class HierarchicalRVQGPT_T2M(nn.Module):
    """
    Hierarchical RVQ-GPT with T2M-GPT style Q0 decoder (3-layer version).

    Key difference from HierarchicalRVQGPT:
    - Q0 uses T2M-style decoder where text is PREPENDED to the motion sequence
    - Q1, Q2 use standard HierarchicalRVQDecoder with cross-attention

    T2M-GPT Style Q0:
    - Text embedding prepended to motion sequence (position 0)
    - All motion tokens can attend to text via causal self-attention
    - No separate cross-attention layer needed
    - Sinusoidal positional encoding (fixed)

    Architecture:
        Text → CLIP Encoder → [prepend to sequence] → Q0 Decoder (T2M-style)
                                                              ↓
        Q0 embeddings → Q1 Decoder (cross-attention style)
                                                              ↓
        Q0+Q1 embeddings → Q2 Decoder (cross-attention style)

    This combines:
    - T2M-GPT's proven text-prepending approach for the coarse Q0 decoder
    - Cross-attention conditioning for Q1/Q2 refinement decoders
    """

    def __init__(
        self,
        num_vq=512,
        embed_dim=1024,
        block_size=200,
        num_layers=9,
        num_layers_head=0,  # Additional head layers for T2M-style Q0
        num_layers_q1q2=3,  # Number of layers for Q1/Q2 decoders (smaller than Q0)
        n_head=16,
        dropout=0.1,
        text_dim=512,
        text_encoder_type='clip',  # 'clip', 'bert', or 'bert-large'
        pkeep=1.0,
        scheduled_sampling='t2m',  # Scheduled sampling strategy
    ):
        """
        Args:
            num_vq: Codebook size per quantizer (typically 512)
            embed_dim: Token embedding dimension
            block_size: Maximum sequence length
            num_layers: Number of transformer layers for Q0 decoder
            num_layers_head: Additional layers in T2M-style head (default 0)
            num_layers_q1q2: Number of layers for Q1/Q2 decoders (default 3)
            n_head: Number of attention heads
            dropout: Dropout probability
            text_dim: Text feature dimension (auto-set based on text_encoder_type)
            text_encoder_type: Type of text encoder ('clip', 'bert', 'bert-large')
            pkeep: Probability of keeping GT tokens for hierarchical conditioning
            scheduled_sampling: Scheduled sampling strategy, one of:
                - 't2m': T2M-GPT style (per-token mask + random replacement)
                - 'sample': Per-sample mask + model predictions
                - 'none': No scheduled sampling (always teacher forcing)
        """
        super().__init__()

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_layers_q1q2 = num_layers_q1q2
        self.num_quantizers_used = 3  # Use first 3 of 6 quantizers
        self.pkeep = pkeep
        self.eos_token_id = num_vq
        self.scheduled_sampling = scheduled_sampling
        self.text_encoder_type = text_encoder_type

        # Initialize text encoder
        self.text_encoder = TextEncoder(encoder_type=text_encoder_type, freeze=True)
        # Override text_dim with actual encoder output dim
        text_dim = self.text_encoder.output_dim
        self.text_dim = text_dim

        # Validate scheduled_sampling
        valid_strategies = ['t2m', 'sample', 'none']
        if scheduled_sampling not in valid_strategies:
            raise ValueError(f"scheduled_sampling must be one of {valid_strategies}, got '{scheduled_sampling}'")

        print(f"\n{'='*70}")
        print(f"{'Hierarchical RVQ-GPT (T2M-Style Q0)':^70}")
        print(f"{'='*70}")
        print(f"{'Architecture:':<20} T2M-Q0 → CrossAttn-Q1 → CrossAttn-Q2")
        print(f"{'Q0 Decoder:':<20} T2M-GPT style (text prepended)")
        print(f"{'Q1/Q2 Decoders:':<20} Cross-attention style")
        print(f"{'Quantizers Used:':<20} 3 (first 3 of 6 from RVQ-VAE)")
        print(f"{'Q0 Layers:':<20} {num_layers} (+ {num_layers_head} head layers)")
        print(f"{'Q1/Q2 Layers:':<20} {num_layers_q1q2} each")
        print(f"{'Embed Dim:':<20} {embed_dim}")
        print(f"{'Num Heads:':<20} {n_head}")
        print(f"{'Block Size:':<20} {block_size}")
        print(f"{'Codebook Size:':<20} {num_vq}")
        print(f"{'Text Encoder:':<20} {text_encoder_type}")
        print(f"{'Text Dim:':<20} {text_dim}D")
        print(f"{'PKeep:':<20} {pkeep}")
        print(f"{'Scheduled Sampling:':<20} {scheduled_sampling}")
        print(f"{'='*70}\n")

        # Q0 Decoder: T2M-GPT style (text prepended to sequence)
        self.q0_decoder = T2MStyleQ0Decoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            text_dim=text_dim,
            block_size=block_size,
            num_layers=num_layers,
            num_layers_head=num_layers_head,
            n_head=n_head,
            drop_out_rate=dropout,
        )

        # Text projection for Q1/Q2 decoders (they use cross-attention)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # Q1 Decoder: Cross-attention style (conditioned on Q0)
        self.q1_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=num_layers_q1q2,  # Use smaller number of layers
            n_head=n_head,
            dropout=dropout,
            quantizer_level=1  # Conditioned on Q0
        )

        # Q2 Decoder: Cross-attention style (conditioned on Q0+Q1)
        self.q2_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=num_layers_q1q2,  # Use smaller number of layers
            n_head=n_head,
            dropout=dropout,
            quantizer_level=2  # Conditioned on Q0+Q1
        )

        # Initialize weights (skip Q0 decoder - it has its own init)
        self.text_proj.apply(self._init_weights)
        self.q1_decoder.apply(self._init_weights)
        self.q2_decoder.apply(self._init_weights)

        # Count parameters
        q0_params = sum(p.numel() for p in self.q0_decoder.parameters())
        q1_params = sum(p.numel() for p in self.q1_decoder.parameters())
        q2_params = sum(p.numel() for p in self.q2_decoder.parameters())
        total_params = q0_params + q1_params + q2_params
        print(f"Parameters: Q0(T2M)={q0_params/1e6:.1f}M, Q1={q1_params/1e6:.1f}M, "
              f"Q2={q2_params/1e6:.1f}M, Total={total_params/1e6:.1f}M\n")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        motion_codes: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pkeep: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Supports multiple scheduled sampling strategies (controlled by self.scheduled_sampling):
        - 't2m': T2M-GPT style (per-token mask + random replacement)
        - 'sample': Per-sample mask + model predictions
        - 'none': No scheduled sampling (always teacher forcing, ignores pkeep)

        Args:
            motion_codes: (B, T, 3) or (B, T, 6) - Motion token indices
            text_features: (B, text_dim) - CLIP text features
            text_mask: Not used for T2M-style Q0
            pkeep: Override self.pkeep

        Returns:
            logits_q0: (B, T, num_vq+1) - Logits for Q0
            logits_q1: (B, T, num_vq+1) - Logits for Q1
            logits_q2: (B, T, num_vq+1) - Logits for Q2
        """
        if pkeep is None:
            pkeep = self.pkeep

        # Extract first 3 quantizers if input has 6
        if motion_codes.dim() == 3 and motion_codes.shape[-1] == 6:
            motion_codes = motion_codes[:, :, :3]

        # Split into Q0, Q1, Q2
        motion_q0 = motion_codes[:, :, 0]  # (B, T)
        motion_q1 = motion_codes[:, :, 1]  # (B, T)
        motion_q2 = motion_codes[:, :, 2]  # (B, T)

        B, T = motion_q0.shape
        device = motion_q0.device

        # Ensure text_features is 2D for T2M-style Q0
        if text_features.dim() == 3:
            text_features = text_features.squeeze(1)  # (B, 1, D) → (B, D)

        # === Scheduled Sampling Strategies ===
        use_scheduled_sampling = (
            self.training and
            pkeep < 1.0 and
            self.scheduled_sampling != 'none'
        )

        if use_scheduled_sampling and self.scheduled_sampling == 't2m':
            # ===== T2M-GPT Style =====
            # Per-token masking with random replacement (exactly like original T2M-GPT)

            # Q0: Apply scheduled sampling to input (shift by 1 for autoregressive)
            input_q0 = motion_q0[:, :-1]  # (B, T-1) - input tokens
            mask_q0 = torch.bernoulli(pkeep * torch.ones(input_q0.shape, device=device))
            mask_q0 = mask_q0.round().to(dtype=torch.int64)
            random_q0 = torch.randint_like(input_q0, self.num_vq)
            input_q0_masked = mask_q0 * input_q0 + (1 - mask_q0) * random_q0

            # Q1: Apply scheduled sampling
            input_q1 = motion_q1[:, :-1]
            mask_q1 = torch.bernoulli(pkeep * torch.ones(input_q1.shape, device=device))
            mask_q1 = mask_q1.round().to(dtype=torch.int64)
            random_q1 = torch.randint_like(input_q1, self.num_vq)
            input_q1_masked = mask_q1 * input_q1 + (1 - mask_q1) * random_q1

            # Q2: Apply scheduled sampling
            input_q2 = motion_q2[:, :-1]
            mask_q2 = torch.bernoulli(pkeep * torch.ones(input_q2.shape, device=device))
            mask_q2 = mask_q2.round().to(dtype=torch.int64)
            random_q2 = torch.randint_like(input_q2, self.num_vq)
            input_q2_masked = mask_q2 * input_q2 + (1 - mask_q2) * random_q2

            # Use GT for conditioning context in t2m mode
            q0_for_cond = motion_q0
            q1_for_cond = motion_q1

        elif use_scheduled_sampling and self.scheduled_sampling == 'sample':
            # ===== Per-Sample Style =====
            # Per-sample masking with model predictions (used for hierarchical conditioning)

            # First, do forward pass with GT to get predictions
            input_q0_masked = motion_q0[:, :-1]
            input_q1_masked = motion_q1[:, :-1]
            input_q2_masked = motion_q2[:, :-1]

            # Q0 forward to get predictions
            logits_q0_temp = self.q0_decoder(input_q0_masked, text_features)
            pred_q0 = logits_q0_temp.argmax(dim=-1)  # (B, T)

            # Per-sample mask for Q0 conditioning
            mask_q0 = torch.rand(B, device=device) < pkeep
            mask_q0 = mask_q0.unsqueeze(1).expand(-1, T)
            q0_for_cond = torch.where(mask_q0, motion_q0, pred_q0[:, :T] if pred_q0.shape[1] >= T else
                                      torch.cat([pred_q0, motion_q0[:, pred_q0.shape[1]:]], dim=1))

            # Q1 forward to get predictions (needs Q0 embeddings)
            q0_embeddings_temp = self.q0_decoder.get_embeddings(q0_for_cond)
            text_context_temp = self.text_proj(text_features).unsqueeze(1)
            logits_q1_temp = self.q1_decoder(
                idx=input_q1_masked,
                text_context=text_context_temp,
                text_mask=text_mask,
                prev_quantizers_context=q0_embeddings_temp
            )
            pred_q1 = logits_q1_temp.argmax(dim=-1)

            # Per-sample mask for Q1 conditioning
            mask_q1 = torch.rand(B, device=device) < pkeep
            mask_q1 = mask_q1.unsqueeze(1).expand(-1, T)
            q1_for_cond = torch.where(mask_q1, motion_q1, pred_q1[:, :T] if pred_q1.shape[1] >= T else
                                      torch.cat([pred_q1, motion_q1[:, pred_q1.shape[1]:]], dim=1))

        else:
            # ===== No Scheduled Sampling (Teacher Forcing) =====
            # Full teacher forcing: use GT tokens (shifted)
            input_q0_masked = motion_q0[:, :-1]
            input_q1_masked = motion_q1[:, :-1]
            input_q2_masked = motion_q2[:, :-1]
            q0_for_cond = motion_q0
            q1_for_cond = motion_q1

        # === Q0: T2M-style forward (text prepended) ===
        # Input is shifted tokens, output predicts next token at each position
        logits_q0_full = self.q0_decoder(input_q0_masked, text_features)  # (B, T, num_vq+1)
        # Position 0 predicts first token, position T-1 predicts T-th token
        # Keep all positions (including text position which predicts first motion token)
        logits_q0 = logits_q0_full  # (B, T, num_vq+1) - includes text position

        # Get Q0 embeddings for Q1 conditioning
        q0_embeddings = self.q0_decoder.get_embeddings(q0_for_cond)

        # Project text for Q1/Q2 cross-attention decoders
        text_context = self.text_proj(text_features).unsqueeze(1)  # (B, 1, embed_dim)

        # === Q1: Cross-attention forward (conditioned on Q0) ===
        logits_q1 = self.q1_decoder(
            idx=input_q1_masked,
            text_context=text_context,
            text_mask=text_mask,
            prev_quantizers_context=q0_embeddings
        )

        # Get Q1 embeddings for Q2 conditioning
        q1_embeddings = self.q1_decoder.get_embeddings(q1_for_cond)

        # === Q2: Cross-attention forward (conditioned on Q0+Q1) ===
        q0_q1_embeddings = torch.cat([q0_embeddings, q1_embeddings], dim=1)
        logits_q2 = self.q2_decoder(
            idx=input_q2_masked,
            text_context=text_context,
            text_mask=text_mask,
            prev_quantizers_context=q0_q1_embeddings
        )

        return logits_q0, logits_q1, logits_q2

    @torch.no_grad()
    def generate(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Autoregressive generation with hierarchical conditioning.

        Args:
            text_features: (B, text_dim) - CLIP text features
            text_mask: Not used
            max_len: Maximum generation length
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated_codes: (B, actual_len, 3) - Generated Q0, Q1, Q2 codes
            lengths: List[int] - Actual generated lengths per sample
        """
        B = text_features.shape[0]
        device = text_features.device

        # Ensure text_features is 2D
        if text_features.dim() == 3:
            text_features = text_features.squeeze(1)

        # Text context for Q1/Q2
        text_context = self.text_proj(text_features).unsqueeze(1)  # (B, 1, embed_dim)

        max_len = min(max_len, self.block_size)

        # Initialize sequences
        q0_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q1_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q2_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(max_len):
            # === Step 1: Generate Q0 token (T2M-style) ===
            logits_q0_full = self.q0_decoder(q0_tokens, text_features)  # (B, t+1, num_vq+1)
            logits_q0_t = logits_q0_full[:, -1, :] / temperature  # (B, num_vq+1)

            if do_sample:
                q0_t = self._sample(logits_q0_t, top_k=top_k, top_p=top_p)
            else:
                q0_t = logits_q0_t.argmax(dim=-1)

            # Check for EOS in Q0
            eos_mask = (q0_t == self.eos_token_id) & ~finished
            lengths[eos_mask] = t
            finished = finished | eos_mask

            if finished.all():
                break

            # Replace EOS with placeholder for finished samples
            q0_t = torch.where(finished, torch.zeros_like(q0_t), q0_t)
            q0_tokens = torch.cat([q0_tokens, q0_t.unsqueeze(1)], dim=1)

            # Get Q0 embeddings for Q1
            q0_emb = self.q0_decoder.get_embeddings(q0_tokens)

            # === Step 2: Generate Q1 token (cross-attention style) ===
            logits_q1 = self.q1_decoder(
                idx=q1_tokens if q1_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                text_context=text_context,
                text_mask=text_mask,
                prev_quantizers_context=q0_emb
            )
            logits_q1_t = logits_q1[:, -1, :] / temperature

            if do_sample:
                q1_t = self._sample(logits_q1_t, top_k=top_k, top_p=top_p)
            else:
                q1_t = logits_q1_t.argmax(dim=-1)

            q1_t = torch.where(finished, torch.zeros_like(q1_t), q1_t)
            q1_tokens = torch.cat([q1_tokens, q1_t.unsqueeze(1)], dim=1)

            # Get Q1 embeddings for Q2
            q1_emb = self.q1_decoder.get_embeddings(q1_tokens)

            # === Step 3: Generate Q2 token (cross-attention style) ===
            q0_q1_emb = torch.cat([q0_emb, q1_emb], dim=1)
            logits_q2 = self.q2_decoder(
                idx=q2_tokens if q2_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device),
                text_context=text_context,
                text_mask=text_mask,
                prev_quantizers_context=q0_q1_emb
            )
            logits_q2_t = logits_q2[:, -1, :] / temperature

            if do_sample:
                q2_t = self._sample(logits_q2_t, top_k=top_k, top_p=top_p)
            else:
                q2_t = logits_q2_t.argmax(dim=-1)

            q2_t = torch.where(finished, torch.zeros_like(q2_t), q2_t)
            q2_tokens = torch.cat([q2_tokens, q2_t.unsqueeze(1)], dim=1)

        # Set lengths for samples that didn't hit EOS
        lengths[~finished] = q0_tokens.shape[1]

        # Stack all quantizers
        if q0_tokens.shape[1] > 0:
            generated_codes = torch.stack([q0_tokens, q1_tokens, q2_tokens], dim=-1)
        else:
            generated_codes = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
            lengths = torch.ones(B, dtype=torch.long, device=device)

        return generated_codes, lengths.tolist()

    def _sample(self, logits, top_k=None, top_p=None):
        """Sample from logits with optional top-k and nucleus filtering."""
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    # ========================================================================
    # Compatibility methods for MotionGPT training interface
    # ========================================================================

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using the configured text encoder (CLIP/BERT)."""
        return self.text_encoder(texts)

    def __call__(self, texts, motion_tokens, lengths, tasks=None):
        """
        Training forward pass compatible with MotionGPT interface.

        Follows T2M-GPT training exactly:
        - Input: tokens[:, :-1] (all except last)
        - Target: tokens (full sequence, predict next token)
        - Per-token scheduled sampling with random replacement
        """
        # Encode text with CLIP
        text_features = self.encode_text(texts)  # (B, 512)

        # Extract first 3 quantizers if input has 6
        if motion_tokens.dim() == 3 and motion_tokens.shape[-1] == 6:
            motion_tokens = motion_tokens[:, :, :3]

        B, T, _ = motion_tokens.shape
        device = motion_tokens.device

        # === T2M-GPT style target creation ===
        # Target is the full sequence - we predict each token from previous tokens
        # Input (handled in forward): motion_tokens[:, :-1]
        # Target: motion_tokens (shifted by 1, so target[i] = input[i+1])
        targets_q0 = motion_tokens[:, :, 0].clone()  # (B, T)
        targets_q1 = motion_tokens[:, :, 1].clone()  # (B, T)
        targets_q2 = motion_tokens[:, :, 2].clone()  # (B, T)

        # Add EOS token at the end of each sequence
        for i in range(B):
            seq_len = min(lengths[i], T) if lengths is not None else T
            if seq_len < T:
                targets_q0[i, seq_len] = self.eos_token_id
                targets_q1[i, seq_len] = self.eos_token_id
                targets_q2[i, seq_len] = self.eos_token_id
                # Mark positions after EOS as ignored
                targets_q0[i, seq_len+1:] = -1
                targets_q1[i, seq_len+1:] = -1
                targets_q2[i, seq_len+1:] = -1

        # Forward pass (handles input shifting and scheduled sampling internally)
        logits_q0, logits_q1, logits_q2 = self.forward(motion_tokens, text_features)

        # === T2M-GPT style loss computation ===
        # logits_q0: (B, T, num_vq+1) - position i predicts token i
        # targets: (B, T) - token i is the target for position i-1's prediction
        #
        # For Q0 (T2M-style): logits include text position at 0
        #   logits[:, 0] predicts targets[:, 0] (first motion token)
        #   logits[:, T-1] predicts targets[:, T-1] (last/EOS token)

        # Compute losses (similar to T2M-GPT per-sample loss averaging)
        loss_q0 = F.cross_entropy(
            logits_q0.reshape(-1, logits_q0.size(-1)),
            targets_q0.reshape(-1),
            ignore_index=-1
        )
        loss_q1 = F.cross_entropy(
            logits_q1.reshape(-1, logits_q1.size(-1)),
            targets_q1.reshape(-1),
            ignore_index=-1
        )
        loss_q2 = F.cross_entropy(
            logits_q2.reshape(-1, logits_q2.size(-1)),
            targets_q2.reshape(-1),
            ignore_index=-1
        )

        # Total loss (equal weighting)
        total_loss = loss_q0 + loss_q1 + loss_q2

        # Compute accuracies
        with torch.no_grad():
            pred_q0 = logits_q0.argmax(dim=-1)
            pred_q1 = logits_q1.argmax(dim=-1)
            pred_q2 = logits_q2.argmax(dim=-1)

            mask_q0 = targets_q0 != -1
            mask_q1 = targets_q1 != -1
            mask_q2 = targets_q2 != -1

            acc_q0 = (pred_q0[mask_q0] == targets_q0[mask_q0]).float().mean() if mask_q0.any() else torch.tensor(0.0, device=device)
            acc_q1 = (pred_q1[mask_q1] == targets_q1[mask_q1]).float().mean() if mask_q1.any() else torch.tensor(0.0, device=device)
            acc_q2 = (pred_q2[mask_q2] == targets_q2[mask_q2]).float().mean() if mask_q2.any() else torch.tensor(0.0, device=device)

        class LossOutput:
            def __init__(self, loss, loss_q0, loss_q1, loss_q2, acc_q0, acc_q1, acc_q2):
                self.loss = loss
                self.loss_q0 = loss_q0
                self.loss_q1 = loss_q1
                self.loss_q2 = loss_q2
                self.acc_q0 = acc_q0
                self.acc_q1 = acc_q1
                self.acc_q2 = acc_q2

        return LossOutput(total_loss, loss_q0, loss_q1, loss_q2, acc_q0, acc_q1, acc_q2)

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None, **kwargs):
        """Generation method compatible with MotionGPT interface."""
        text_features, text_mask = self.encode_text(texts)

        if lengths is not None:
            if isinstance(lengths, list):
                max_len = max(lengths) // 4
            else:
                max_len = int(lengths.max().item()) // 4
        else:
            max_len = self.block_size

        max_len = min(max_len, self.block_size)

        generated_codes, gen_lengths = self.generate(
            text_features=text_features,
            text_mask=text_mask,
            max_len=max_len,
            do_sample=True,
            temperature=1.0
        )

        outputs_tokens = []
        for i in range(generated_codes.shape[0]):
            actual_len = gen_lengths[i]
            if actual_len > 0:
                outputs_tokens.append(generated_codes[i, :actual_len, :])
            else:
                outputs_tokens.append(generated_codes[i, :1, :])

        return outputs_tokens
