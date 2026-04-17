"""
Mogo-Style Hierarchical RVQ-GPT for Motion Generation

Adapted from Mogo's Transformotion architecture with key innovations:
1. Shared token embedding + shared output head across all quantizer levels
2. Quantizer-level embedding (one-hot → linear) to distinguish levels
3. Additive summation of previous level tokens for inter-level conditioning
4. Per-level transformer decoders with different depth/heads
5. Classifier-Free Guidance (CFG) via condition dropout
6. Speech encoder support (our extension over Mogo)

Mogo's core idea: at each quantizer level i, the input is
    sum(tok_emb(Q0), tok_emb(Q1), ..., tok_emb(Qi)) + quant_emb(i)
This aligns with RVQ's residual structure where each level refines the previous.

Architecture:
    Text/Speech → Encoder → Project to embed_dim
                                ↓
    For each quantizer level i (0..N-1):
        input = sum(tok_emb(levels 0..i))
        cond = text/speech_proj + quant_emb(i)
        output = TransformerDecoder_i([cond | input])
        logits_i = shared_head(output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from functools import partial

from .tools.attention import CausalSelfAttention
from .tools.cross_attention import CrossAttention
from .speech_encoder import SpeechEncoder
import clip


# ===========================================================================
#  Mogo-Style Decoder Block (causal self-attn + cross-attn to conditioning)
# ===========================================================================
class MogoDecoderBlock(nn.Module):
    """
    Single transformer decoder block for Mogo-style generation.

    Architecture:
        x → LayerNorm → CausalSelfAttention → +residual
          → LayerNorm → CrossAttention(to conditioning) → +residual
          → LayerNorm → FFN → +residual
    """

    def __init__(self, embed_dim=1024, block_size=200, n_head=16,
                 dropout=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

        self.self_attn = CausalSelfAttention(
            embed_dim=embed_dim,
            block_size=block_size,
            n_head=n_head,
            drop_out_rate=dropout
        )

        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            drop_out_rate=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_context, cond_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - token sequence (possibly with cond prefix)
            cond_context: (B, S, embed_dim) - conditioning features for cross-attn
            cond_mask: (B, S) - mask for conditioning (1=valid, 0=pad)
        """
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), cond_context, context_mask=cond_mask)
        x = x + self.mlp(self.ln3(x))
        return x


# ===========================================================================
#  Mogo-Style Per-Level Decoder
# ===========================================================================
class MogoLevelDecoder(nn.Module):
    """
    Transformer decoder for a single quantizer level.

    Takes pre-embedded tokens (from shared tok_emb) and conditioning,
    outputs hidden states (to be passed through shared output head).
    """

    def __init__(self, embed_dim, block_size, num_layers, n_head, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            MogoDecoderBlock(embed_dim, block_size, n_head, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x, cond_context, cond_mask=None):
        """
        Args:
            x: (B, T, embed_dim) - input embeddings
            cond_context: (B, S, embed_dim) - conditioning for cross-attention
            cond_mask: (B, S) - conditioning mask
        Returns:
            (B, T, embed_dim)
        """
        for block in self.blocks:
            x = block(x, cond_context, cond_mask)
        return self.ln_f(x)


# ===========================================================================
#  Text Encoder (reused from mgpt_rvq_hierarchical)
# ===========================================================================
TEXT_ENCODER_CONFIGS = {
    'clip': {'dim': 512, 'model': 'ViT-B/32'},
    'bert': {'dim': 768, 'model': 'bert-base-uncased'},
    'bert-large': {'dim': 1024, 'model': 'bert-large-uncased'},
}


class TextEncoder(nn.Module):
    """Switchable text encoder supporting CLIP and BERT."""

    def __init__(self, encoder_type: str = 'clip', freeze: bool = True):
        super().__init__()
        if encoder_type not in TEXT_ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.config = TEXT_ENCODER_CONFIGS[encoder_type]
        self.text_dim = self.config['dim']

        if encoder_type == 'clip':
            self.model, _ = clip.load(self.config['model'], device='cpu', jit=False)
        else:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.config['model'])
            self.model = BertModel.from_pretrained(self.config['model'])

        self._freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        if self._freeze:
            return super().train(False)
        return super().train(mode)

    @property
    def output_dim(self) -> int:
        return self.text_dim

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.model.parameters()).device
        if self.encoder_type == 'clip':
            text_tokens = clip.tokenize(texts, truncate=True).to(device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens).float()
            text_features = text_features.unsqueeze(1)  # (B, 1, 512)
            return text_features, None
        else:
            inputs = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=77, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_features = outputs.last_hidden_state.float()
            return text_features, inputs.attention_mask


# ===========================================================================
#  Main Model: MogoHierarchicalGPT
# ===========================================================================
class MogoHierarchicalGPT(nn.Module):
    """
    Mogo-Style Hierarchical RVQ-GPT.

    Key differences from HierarchicalRVQGPT:
    1. Shared token embedding and output head across all quantizer levels
    2. Quantizer-level embedding (one-hot → linear)
    3. Additive summation for inter-level conditioning (not cross-attention)
    4. CFG via condition dropout

    Generation at each quantizer level i:
        tokens_sum = sum(tok_emb(Q_0), ..., tok_emb(Q_i))
        cond = speech/text_proj + quant_emb(i)
        logits_i = head(decoder_i(tokens_sum, cond))
    """

    def __init__(
        self,
        num_vq=512,
        embed_dim=1024,
        block_size=200,
        num_quantizers=3,
        # Per-level layer counts: list of ints, e.g. [9, 6, 4]
        layers_per_level=None,
        # Per-level head counts: list of ints, e.g. [16, 16, 8]
        heads_per_level=None,
        # Fallback if per-level not specified
        num_layers=9,
        n_head=16,
        dropout=0.1,
        # Conditioning
        text_encoder_type='clip',
        speech_encoder_type=None,
        use_speech=False,
        # Training
        cond_drop_prob=0.1,  # CFG: probability of dropping conditioning
        pkeep=1.0,  # Probability of keeping GT tokens (scheduled sampling)
        corrupt_ratio=0.0,  # Per-time RVQ augmentation
    ):
        super().__init__()

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_quantizers = num_quantizers
        self.cond_drop_prob = cond_drop_prob
        self.pkeep = pkeep
        self.corrupt_ratio = corrupt_ratio
        self.eos_token_id = num_vq
        self.use_speech = use_speech
        self.text_encoder_type = text_encoder_type
        self.speech_encoder_type = speech_encoder_type

        # Per-level config
        if layers_per_level is None:
            layers_per_level = [num_layers] * num_quantizers
        if heads_per_level is None:
            heads_per_level = [n_head] * num_quantizers
        assert len(layers_per_level) == num_quantizers
        assert len(heads_per_level) == num_quantizers
        self.layers_per_level = layers_per_level
        self.heads_per_level = heads_per_level

        # === Shared token embedding (Mogo-style) ===
        # All quantizer levels share the same embedding and output head
        self.tok_emb = nn.Embedding(num_vq + 1, embed_dim)  # +1 for EOS
        self.tok_dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        # Shared output head
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)

        # === Quantizer-level embedding (Mogo-style) ===
        # One-hot quantizer index → embed_dim
        self.encode_quant = partial(F.one_hot, num_classes=num_quantizers)
        self.quant_emb = nn.Linear(num_quantizers, embed_dim)

        # === Conditioning encoder ===
        if use_speech:
            assert speech_encoder_type is not None
            self.speech_encoder = SpeechEncoder(encoder_type=speech_encoder_type, freeze=True)
            audio_dim = self.speech_encoder.output_dim
            self.cond_proj = nn.Linear(audio_dim, embed_dim)
            self.text_encoder = None
            cond_info = f"Speech ({speech_encoder_type}), {audio_dim}D → {embed_dim}D"
        else:
            self.text_encoder = TextEncoder(encoder_type=text_encoder_type, freeze=True)
            text_dim = self.text_encoder.output_dim
            self.cond_proj = nn.Linear(text_dim, embed_dim)
            self.speech_encoder = None
            cond_info = f"Text ({text_encoder_type}), {text_dim}D → {embed_dim}D"

        # === Per-level transformer decoders ===
        self.decoders = nn.ModuleList()
        for i in range(num_quantizers):
            decoder = MogoLevelDecoder(
                embed_dim=embed_dim,
                block_size=block_size + 1,  # +1 for cond prefix token
                num_layers=layers_per_level[i],
                n_head=heads_per_level[i],
                dropout=dropout,
            )
            self.decoders.append(decoder)

        # Initialize weights
        self.apply(self._init_weights)

        # Print summary
        print(f"\n{'='*70}")
        print(f"{'Mogo-Style Hierarchical RVQ-GPT':^70}")
        print(f"{'='*70}")
        print(f"{'Quantizer Levels:':<25} {num_quantizers}")
        print(f"{'Conditioning:':<25} {cond_info}")
        print(f"{'Codebook Size:':<25} {num_vq}")
        print(f"{'Embed Dim:':<25} {embed_dim}")
        print(f"{'Block Size:':<25} {block_size}")
        print(f"{'CFG Drop Prob:':<25} {cond_drop_prob}")
        print(f"{'PKeep:':<25} {pkeep}")
        print(f"{'Corrupt Ratio:':<25} {corrupt_ratio}")
        print(f"{'Shared Embedding:':<25} Yes (Mogo-style)")
        print(f"{'Shared Output Head:':<25} Yes (Mogo-style)")
        print(f"{'Inter-level Cond:':<25} Additive Summation (Mogo-style)")
        for i in range(num_quantizers):
            print(f"{'  Level ' + str(i) + ':':<25} {layers_per_level[i]} layers, {heads_per_level[i]} heads")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_params = [sum(p.numel() for p in d.parameters()) for d in self.decoders]
        print(f"{'Decoder Params:':<25} {', '.join(f'L{i}={p/1e6:.1f}M' for i,p in enumerate(decoder_params))}")
        print(f"{'Total Trainable:':<25} {total_params/1e6:.1f}M")
        print(f"{'='*70}\n")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ===================================================================
    #  Conditioning
    # ===================================================================

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.text_encoder is None:
            raise ValueError("Text encoder not initialized. Set use_speech=False.")
        return self.text_encoder(texts)

    def encode_audio(self, audio_waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.speech_encoder is None:
            raise ValueError("Speech encoder not initialized. Set use_speech=True.")
        return self.speech_encoder(audio_waveforms)

    def _get_conditioning(self, text_features: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        """
        Project conditioning features to embed_dim.

        Args:
            text_features: (B, S, cond_dim) - raw encoder output
            text_mask: (B, S) - mask

        Returns:
            cond_context: (B, S, embed_dim) - projected features for cross-attention
        """
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        return self.cond_proj(text_features)

    def _mask_conditioning(self, cond_context: torch.Tensor, force_mask=False):
        """
        Classifier-Free Guidance: randomly zero out conditioning during training.

        Args:
            cond_context: (B, S, embed_dim)
            force_mask: If True, always mask (for unconditional generation)

        Returns:
            cond_context: (B, S, embed_dim) with some samples zeroed out
        """
        if force_mask:
            return torch.zeros_like(cond_context)
        if self.training and self.cond_drop_prob > 0:
            B = cond_context.shape[0]
            mask = torch.bernoulli(
                torch.ones(B, device=cond_context.device) * self.cond_drop_prob
            ).view(B, 1, 1)
            return cond_context * (1.0 - mask)
        return cond_context

    # ===================================================================
    #  Per-time corruption (from our original architecture)
    # ===================================================================

    def _corrupt_per_time(self, motion_codes: torch.Tensor) -> torch.Tensor:
        """Per-time corrupted RVQ augmentation."""
        B, T, num_q = motion_codes.shape
        device = motion_codes.device
        corrupt_mask = torch.rand(B, T, device=device) < self.corrupt_ratio
        random_codes = torch.randint(0, self.num_vq, (B, T, num_q), device=device)
        corrupt_mask = corrupt_mask.unsqueeze(-1).expand_as(motion_codes)
        return torch.where(corrupt_mask, random_codes, motion_codes)

    # ===================================================================
    #  Forward (Training)
    # ===================================================================

    def forward(
        self,
        motion_codes: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pkeep: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Mogo-style forward pass for training.

        At each quantizer level i:
        1. Sum token embeddings from levels 0..i (Mogo's additive conditioning)
        2. Create conditioning token: cond_proj(speech/text) + quant_emb(i)
        3. Run through level i's decoder with cross-attention to conditioning
        4. Compute logits via shared output head

        Args:
            motion_codes: (B, T, num_q) - Motion token indices
            text_features: (B, S, cond_dim) - Conditioning features
            text_mask: (B, S) - Conditioning mask
            pkeep: Override scheduled sampling probability

        Returns:
            Tuple of logits: (logits_q0, logits_q1, ..., logits_qN)
                Each: (B, T, num_vq+1)
        """
        if pkeep is None:
            pkeep = self.pkeep

        # Truncate to num_quantizers if needed
        if motion_codes.dim() == 3 and motion_codes.shape[-1] > self.num_quantizers:
            motion_codes = motion_codes[:, :, :self.num_quantizers]

        B, T, num_q = motion_codes.shape
        device = motion_codes.device

        # Per-time corruption (training augmentation)
        if self.training and self.corrupt_ratio > 0:
            input_codes = self._corrupt_per_time(motion_codes)
        else:
            input_codes = motion_codes

        # Project conditioning features
        cond_context = self._get_conditioning(text_features, text_mask)
        # Apply CFG dropout
        cond_context = self._mask_conditioning(cond_context)

        # Embed all quantizer levels at once using shared tok_emb
        # input_codes: (B, T, num_q) → token_embs: (B, T, num_q, embed_dim)
        token_embs = self.tok_emb(input_codes)  # (B, T, num_q, embed_dim)

        # Generate logits for each quantizer level
        all_logits = []
        for i, decoder in enumerate(self.decoders):
            # Mogo's additive summation: sum embeddings from levels 0..i
            # (B, T, i+1, embed_dim) → sum → (B, T, embed_dim)
            summed_tokens = token_embs[:, :, :i + 1, :].sum(dim=2)  # (B, T, embed_dim)

            # Add positional embedding
            pos_emb = self.pos_emb[:, :T, :]
            stage_input = self.tok_dropout(summed_tokens + pos_emb)  # (B, T, embed_dim)

            # Quantizer-level embedding added to conditioning
            qids = torch.full((B,), i, dtype=torch.long, device=device)
            q_onehot = self.encode_quant(qids).float()
            q_emb = self.quant_emb(q_onehot).unsqueeze(1)  # (B, 1, embed_dim)
            level_cond = cond_context + q_emb  # broadcast: (B, S, embed_dim)

            # Forward through level decoder with cross-attention to conditioning
            hidden = decoder(stage_input, level_cond, cond_mask=text_mask)  # (B, T, embed_dim)

            # Shared output head
            logits = self.head(hidden)  # (B, T, num_vq+1)
            all_logits.append(logits)

        return tuple(all_logits)

    # ===================================================================
    #  __call__ — MotionGPT training interface
    # ===================================================================

    def __call__(self, texts=None, audio_waveforms=None, motion_tokens=None, lengths=None, tasks=None,
                 speech_feats=None, speech_mask=None):
        """
        Training forward pass compatible with MotionGPT interface.
        Called by train_lm_forward() in mgpt.py.

        Args:
            texts: List[str] - Text descriptions (for text mode)
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz (for speech mode)
            motion_tokens: (B, T, num_quantizers) - Ground truth motion codes
            lengths: List[int] - Sequence lengths (in tokens, not frames)
            tasks: Dict - Task information (not used)
            speech_feats: (B, T_s, D) - Precomputed speech encoder features
            speech_mask: (B, T_s) - Mask for precomputed speech features

        Returns:
            Object with .loss, .loss_q0..qN, .acc_q0..qN attributes
        """
        # Encode conditioning modality
        if self.use_speech:
            if speech_feats is not None:
                conditioning_features = self.project_audio(speech_feats, speech_mask)
                conditioning_mask = speech_mask
            elif audio_waveforms is not None:
                audio_features, audio_mask = self.encode_audio(audio_waveforms)
                conditioning_features = self.project_audio(audio_features, audio_mask)
                conditioning_mask = audio_mask
            else:
                raise ValueError("speech_feats or audio_waveforms must be provided in speech mode")
        else:
            if texts is None:
                raise ValueError("texts must be provided in text mode")
            text_features, text_mask = self.encode_text(texts)
            conditioning_features = text_features
            conditioning_mask = text_mask

        # Truncate to num_quantizers
        if motion_tokens.dim() == 3 and motion_tokens.shape[-1] > self.num_quantizers:
            motion_tokens = motion_tokens[:, :, :self.num_quantizers]

        B, T, _ = motion_tokens.shape
        device = motion_tokens.device

        # Build targets with EOS (next-token prediction)
        targets = []
        for q in range(self.num_quantizers):
            tgt = torch.full((B, T), -1, dtype=torch.long, device=device)
            for i in range(B):
                seq_len = min(lengths[i], T - 1) if lengths is not None else T - 1
                tgt[i, :seq_len] = motion_tokens[i, 1:seq_len+1, q]
                if seq_len < T:
                    tgt[i, seq_len] = self.eos_token_id
            targets.append(tgt)

        # Forward through Mogo-style hierarchical model
        all_logits = self.forward(motion_tokens, conditioning_features, text_mask=conditioning_mask)

        # Compute losses and accuracies
        losses = []
        accs = []
        for q in range(self.num_quantizers):
            logits_q = all_logits[q][:, :-1, :]  # (B, T-1, num_vq+1)
            target_q = targets[q][:, :T-1]

            loss_q = F.cross_entropy(
                logits_q.reshape(-1, logits_q.size(-1)),
                target_q.reshape(-1),
                ignore_index=-1
            )
            losses.append(loss_q)

            with torch.no_grad():
                pred_q = logits_q.argmax(dim=-1)
                mask_q = target_q != -1
                acc_q = (pred_q[mask_q] == target_q[mask_q]).float().mean() if mask_q.any() else torch.tensor(0.0, device=device)
                accs.append(acc_q)

        total_loss = sum(losses)

        # Return compatible output object
        class LossOutput:
            pass

        output = LossOutput()
        output.loss = total_loss
        for q in range(self.num_quantizers):
            setattr(output, f'loss_q{q}', losses[q])
            setattr(output, f'acc_q{q}', accs[q])

        return output

    # ===================================================================
    #  Generation (Inference)
    # ===================================================================

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None,
                             audio_waveforms=None, **kwargs):
        """
        Generation method compatible with MotionGPT interface.
        Called by val_t2m_forward() in mgpt.py.

        Args:
            texts: List[str] - Text descriptions (for text mode)
            audio_waveforms: (B, num_samples) - Raw audio at 16kHz (for speech mode)
            lengths: List[int] or Tensor - Target generation lengths
            stage: str - 'test' or 'val'
            tasks: Dict - Task information (not used)

        Returns:
            List[Tensor] - List of (T_i, num_quantizers) tensors, one per sample
        """
        # Encode conditioning
        if self.use_speech:
            if audio_waveforms is None:
                raise ValueError("audio_waveforms must be provided in speech mode")
            audio_features, audio_mask = self.encode_audio(audio_waveforms)
            cond_features = self._get_conditioning(audio_features, audio_mask)
            cond_mask = audio_mask
        else:
            if texts is None:
                raise ValueError("texts must be provided in text mode")
            text_features, text_mask = self.encode_text(texts)
            cond_features = self._get_conditioning(text_features, text_mask)
            cond_mask = text_mask

        # Determine max generation length
        if lengths is not None:
            if isinstance(lengths, list):
                max_len = max(lengths) // 4
            else:
                max_len = int(lengths.max().item()) // 4
        else:
            max_len = self.block_size
        max_len = min(max_len, self.block_size)

        # Generate
        generated_codes, gen_lengths = self.generate(
            cond_features, cond_mask, max_len,
            do_sample=True, temperature=1.0
        )

        # Convert (B, T, num_q) tensor to list of (T_i, num_q) tensors
        outputs_tokens = []
        for i in range(generated_codes.shape[0]):
            actual_len = gen_lengths[i]
            if actual_len > 0:
                outputs_tokens.append(generated_codes[i, :actual_len, :])
            else:
                outputs_tokens.append(generated_codes[i, :1, :])

        return outputs_tokens

    @torch.no_grad()
    def generate(
        self,
        cond_features: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Autoregressive generation, Mogo-style.

        At each timestep t, for each quantizer level i:
        1. Sum token embeddings from levels 0..i (using previously generated tokens)
        2. Run through decoder_i
        3. Sample from logits via shared head

        Args:
            cond_features: (B, S, embed_dim) - projected conditioning features
            cond_mask: (B, S) - conditioning mask
            max_len: Maximum generation length
            do_sample: Sample or greedy
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            cfg_scale: CFG scale (>1.0 amplifies conditioning)
        """
        B = cond_features.shape[0]
        device = cond_features.device
        max_len = min(max_len, self.block_size)

        # For CFG: prepare unconditional features
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            uncond_features = torch.zeros_like(cond_features)

        # Initialize per-level generated token sequences
        # level_tokens[i]: (B, 0) initially, grows to (B, T)
        level_tokens = [torch.zeros((B, 0), dtype=torch.long, device=device)
                        for _ in range(self.num_quantizers)]

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(max_len):
            for i in range(self.num_quantizers):
                # Build token embeddings for levels 0..i using generated tokens so far
                # For level 0 at first timestep, we have no tokens yet
                if level_tokens[0].shape[1] == 0 and i == 0:
                    # No motion tokens yet — use a dummy zero input
                    dummy_input = torch.zeros((B, 1, self.embed_dim), device=device)
                    pos_emb = self.pos_emb[:, :1, :]
                    stage_input = self.tok_dropout(dummy_input + pos_emb)
                else:
                    # For levels 0..i, get embeddings and sum them
                    T_cur = level_tokens[0].shape[1]
                    if i == 0 and T_cur == 0:
                        # Edge case: first level, no tokens generated yet
                        dummy_input = torch.zeros((B, 1, self.embed_dim), device=device)
                        pos_emb = self.pos_emb[:, :1, :]
                        stage_input = self.tok_dropout(dummy_input + pos_emb)
                    else:
                        # Sum embeddings from levels 0..i
                        summed = torch.zeros((B, T_cur, self.embed_dim), device=device)
                        for j in range(i + 1):
                            if level_tokens[j].shape[1] == T_cur:
                                summed = summed + self.tok_emb(level_tokens[j])
                            # If level j doesn't have T_cur tokens yet (hasn't been predicted
                            # at this timestep), use only what's available
                            elif level_tokens[j].shape[1] > 0:
                                emb_j = self.tok_emb(level_tokens[j])
                                summed[:, :emb_j.shape[1], :] = summed[:, :emb_j.shape[1], :] + emb_j

                        pos_emb = self.pos_emb[:, :T_cur, :]
                        stage_input = self.tok_dropout(summed + pos_emb)

                # Quantizer-level embedding
                qids = torch.full((B,), i, dtype=torch.long, device=device)
                q_onehot = self.encode_quant(qids).float()
                q_emb = self.quant_emb(q_onehot).unsqueeze(1)
                level_cond = cond_features + q_emb

                # Forward through decoder
                hidden = self.decoders[i](stage_input, level_cond, cond_mask)
                logits = self.head(hidden[:, -1, :])  # (B, num_vq+1)

                # CFG: classifier-free guidance
                if use_cfg:
                    uncond_level_cond = uncond_features + q_emb
                    hidden_uncond = self.decoders[i](stage_input, uncond_level_cond, cond_mask)
                    logits_uncond = self.head(hidden_uncond[:, -1, :])
                    logits = logits_uncond + cfg_scale * (logits - logits_uncond)

                logits = logits / temperature

                # Sample
                if do_sample:
                    token = self._sample(logits, top_k=top_k, top_p=top_p)
                else:
                    token = logits.argmax(dim=-1)

                # Check EOS (only on level 0 — controls overall length)
                if i == 0:
                    eos_mask = (token == self.eos_token_id) & ~finished
                    lengths[eos_mask] = t
                    finished = finished | eos_mask
                    if finished.all():
                        break

                # Mask finished samples
                token = torch.where(finished, torch.zeros_like(token), token)
                level_tokens[i] = torch.cat([level_tokens[i], token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Set lengths for samples that didn't hit EOS
        lengths[~finished] = level_tokens[0].shape[1]

        # Stack all levels: (B, T, num_quantizers)
        if level_tokens[0].shape[1] > 0:
            generated_codes = torch.stack(level_tokens, dim=-1)
        else:
            generated_codes = torch.zeros((B, 1, self.num_quantizers),
                                          dtype=torch.long, device=device)
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
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    # ===================================================================
    #  Compatibility with MotionGPT training interface
    # ===================================================================

    def project_audio(self, audio_features: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """Project audio features for use as conditioning."""
        return self._get_conditioning(audio_features, audio_mask)
