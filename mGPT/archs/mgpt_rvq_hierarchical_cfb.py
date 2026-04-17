"""
Hierarchical RVQ-GPT with Condition Fusion Blocks (CFB)

A refactor of HierarchicalRVQGPT where Q1 and Q2 decoders are decomposed into:
    1. FusionStack (CFB x M): pre-fuses previous-level Q embeddings with the
       conditioning modality (speech/text) into a condition-aware context.
    2. DecoderStack: standard Q0-style decoder (causal self-attn + single
       cross-attn + FFN) whose cross-attention K/V is the fused context.

This makes Q0/Q1/Q2 structurally symmetric (all use Q0DecoderBlock internally)
and isolates the hierarchical conditioning into an explicit, reusable module.

Additive only: the original HierarchicalRVQGPT and its blocks are untouched.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mgpt_rvq_hierarchical import TextEncoder
from .tools.cfb_block import CFBStack
from .tools.rvq_hierarchical_blocks import HierarchicalRVQDecoder
from .tools.llamagen_blocks import LlamaGenQ0Decoder
from .speech_encoder import SpeechEncoder
from .pos_encoding import PositionEmbedding


class HierarchicalRVQGPTCFB(nn.Module):
    """
    Hierarchical RVQ-GPT with CFB-based pre-fusion.

    Generation order is unchanged: Q0 (coarse) -> Q1 (medium) -> Q2 (fine).
    The difference is that Q1/Q2 decoders no longer have a dedicated
    cross-attention to Q0/Q1 inside each block. Instead, a CFBStack pre-fuses
    the previous-level embeddings with the speech/text context once, and the
    decoder consumes that fused context.

    Q0 decoder: standard (LlamaGen or Transformer) cross-attending to cond_ctx.
    Q1 decoder: Q0_emb --CFB--> fused_ctx ; decoder cross-attends to fused_ctx.
    Q2 decoder: cat(Q0_emb, Q1_emb) --CFB--> fused_ctx ; decoder cross-attends.
    """

    def __init__(
        self,
        num_vq: int = 512,
        embed_dim: int = 1024,
        block_size: int = 200,
        num_layers: int = 9,
        num_layers_q0: Optional[int] = None,
        num_layers_q1: Optional[int] = None,
        num_layers_q2: Optional[int] = None,
        num_cfb_layers_q1: int = 3,
        num_cfb_layers_q2: int = 3,
        cfb_use_gate: bool = True,
        n_head: int = 16,
        dropout: float = 0.1,
        text_encoder_type: str = 'clip',
        speech_encoder_type: Optional[str] = None,
        use_speech: bool = False,
        n_kv_head: Optional[int] = None,
        drop_path_rate: float = 0.0,
        pkeep: float = 1.0,
        contrastive_speech_ckpt: Optional[str] = None,
        contrastive_proj_dim: int = 512,
        contrastive_num_layers: int = 4,
        contrastive_nhead: int = 8,
        contrastive_ff_dim: Optional[int] = None,
        contrastive_max_seq_len: int = 2000,
        freeze_contrastive_proj: bool = True,
        q0_decoder_type: str = 'llamagen',
        q1_use_cond: bool = True,
        q2_use_cond: bool = True,
        corrupt_ratio: float = 0.5,
    ):
        super().__init__()

        self.num_layers_q0 = num_layers_q0 if num_layers_q0 is not None else num_layers
        self.num_layers_q1 = num_layers_q1 if num_layers_q1 is not None else num_layers
        self.num_layers_q2 = num_layers_q2 if num_layers_q2 is not None else num_layers
        self.num_cfb_layers_q1 = num_cfb_layers_q1
        self.num_cfb_layers_q2 = num_cfb_layers_q2
        self.cfb_use_gate = cfb_use_gate

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_quantizers_used = 3
        self.pkeep = pkeep
        self.eos_token_id = num_vq
        self.text_encoder_type = text_encoder_type
        self.speech_encoder_type = speech_encoder_type
        self.use_speech = use_speech
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.drop_path_rate = drop_path_rate
        self.q0_decoder_type = q0_decoder_type
        self.q1_use_cond = q1_use_cond
        self.q2_use_cond = q2_use_cond
        self.corrupt_ratio = corrupt_ratio

        # ------------------------------------------------------------------
        # Conditioning encoder (text or speech), same logic as HierarchicalRVQGPT
        # ------------------------------------------------------------------
        self.use_contrastive_proj = contrastive_speech_ckpt is not None
        if use_speech:
            assert speech_encoder_type is not None, \
                "speech_encoder_type must be specified when use_speech=True"
            print(f"\n{'='*70}")
            print(f"{'Speech-Driven Hierarchical RVQ-GPT + CFB':^70}")
            print(f"{'='*70}")

            self.speech_encoder = SpeechEncoder(
                encoder_type=speech_encoder_type, freeze=True,
            )
            audio_dim = self.speech_encoder.output_dim

            if contrastive_speech_ckpt is not None:
                self.contrastive_input_proj = nn.Linear(audio_dim, contrastive_proj_dim)
                self.contrastive_pos_enc = PositionEmbedding(
                    contrastive_max_seq_len, contrastive_proj_dim, dropout,
                )
                _ff_dim = contrastive_ff_dim if contrastive_ff_dim is not None \
                    else contrastive_proj_dim * 4
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=contrastive_proj_dim,
                    nhead=contrastive_nhead,
                    dim_feedforward=_ff_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                self.contrastive_transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=contrastive_num_layers,
                )

                import os
                if os.path.exists(contrastive_speech_ckpt):
                    ckpt = torch.load(contrastive_speech_ckpt, map_location='cpu')
                    mapped_state = {}
                    for k, v in ckpt.items():
                        mapped_state['contrastive_' + k] = v
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
                else:
                    print(f"[WARNING] Contrastive checkpoint not found: {contrastive_speech_ckpt}")

                if freeze_contrastive_proj:
                    for name, p in self.named_parameters():
                        if 'contrastive_' in name:
                            p.requires_grad = False
                    print(f"[Contrastive] Projection head frozen")

                if contrastive_proj_dim == embed_dim:
                    self.audio_proj = nn.Identity()
                    conditioning_modality = "Speech (contrastive-pretrained, direct)"
                    conditioning_dim = f"{audio_dim}D -> {contrastive_proj_dim}D (contrastive) = {embed_dim}D"
                else:
                    self.audio_proj = nn.Linear(contrastive_proj_dim, embed_dim)
                    conditioning_modality = "Speech (contrastive-pretrained)"
                    conditioning_dim = f"{audio_dim}D -> {contrastive_proj_dim}D -> {embed_dim}D"
            else:
                self.audio_proj = nn.Linear(audio_dim, embed_dim)
                conditioning_modality = "Speech"
                conditioning_dim = f"{audio_dim}D -> {embed_dim}D"

            self.text_encoder = None
            self.text_proj = None
            conditioning_encoder = speech_encoder_type
        else:
            print(f"\n{'='*70}")
            print(f"{'Text-Driven Hierarchical RVQ-GPT + CFB':^70}")
            print(f"{'='*70}")

            self.text_encoder = TextEncoder(encoder_type=text_encoder_type, freeze=True)
            text_dim = self.text_encoder.output_dim
            self.text_proj = nn.Linear(text_dim, embed_dim)

            self.speech_encoder = None
            self.audio_proj = None

            conditioning_modality = "Text"
            conditioning_encoder = text_encoder_type
            conditioning_dim = f"{text_dim}D -> {embed_dim}D"

        print(f"{'Architecture:':<22} Q0 -> [CFB] -> Q1 -> [CFB] -> Q2")
        print(f"{'Conditioning:':<22} {conditioning_modality}")
        print(f"{'Encoder Type:':<22} {conditioning_encoder}")
        print(f"{'Conditioning Dim:':<22} {conditioning_dim}")
        print(f"{'Layers (Q0/Q1/Q2):':<22} {self.num_layers_q0}/{self.num_layers_q1}/{self.num_layers_q2}")
        print(f"{'CFB Layers (Q1/Q2):':<22} {self.num_cfb_layers_q1}/{self.num_cfb_layers_q2}")
        print(f"{'CFB Gated FFN:':<22} {self.cfb_use_gate}")
        print(f"{'Embed Dim:':<22} {embed_dim}")
        print(f"{'Num Heads:':<22} {n_head}")
        print(f"{'Q0 Architecture:':<22} {q0_decoder_type}")
        print(f"{'Block Size:':<22} {block_size}")
        print(f"{'Codebook Size:':<22} {num_vq}")
        print(f"{'PKeep:':<22} {pkeep}")
        print(f"{'Corrupt Ratio:':<22} {corrupt_ratio}")
        print(f"{'='*70}\n")

        # ------------------------------------------------------------------
        # Q0 Decoder (unchanged relative to HierarchicalRVQGPT)
        # ------------------------------------------------------------------
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
            raise ValueError(
                f"Unknown q0_decoder_type: {q0_decoder_type}. Use 'llamagen' or 'transformer'."
            )

        # ------------------------------------------------------------------
        # Q1 decoder: Q0-style (self-attn + single cross-attn + FFN)
        # The cross-attn K/V will be the CFB-fused Q0 context.
        # ------------------------------------------------------------------
        self.q1_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=self.num_layers_q1,
            n_head=n_head,
            dropout=dropout,
            quantizer_level=0,  # <-- reuse Q0 block shape
        )

        # ------------------------------------------------------------------
        # Q2 decoder: same shape as Q1
        # ------------------------------------------------------------------
        self.q2_decoder = HierarchicalRVQDecoder(
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=self.num_layers_q2,
            n_head=n_head,
            dropout=dropout,
            quantizer_level=0,
        )

        # ------------------------------------------------------------------
        # CFB stacks: pre-fuse previous-Q embeddings with cond context
        # ------------------------------------------------------------------
        self.cfb_q1 = CFBStack(
            num_layers=num_cfb_layers_q1,
            embed_dim=embed_dim,
            n_head=n_head,
            dropout=dropout,
            use_gate=cfb_use_gate,
        )
        self.cfb_q2 = CFBStack(
            num_layers=num_cfb_layers_q2,
            embed_dim=embed_dim,
            n_head=n_head,
            dropout=dropout,
            use_gate=cfb_use_gate,
        )

        self.apply(self._init_weights)

        q0_params = sum(p.numel() for p in self.q0_decoder.parameters())
        q1_params = sum(p.numel() for p in self.q1_decoder.parameters())
        q2_params = sum(p.numel() for p in self.q2_decoder.parameters())
        cfb_q1_params = sum(p.numel() for p in self.cfb_q1.parameters())
        cfb_q2_params = sum(p.numel() for p in self.cfb_q2.parameters())
        total_params = q0_params + q1_params + q2_params + cfb_q1_params + cfb_q2_params
        print(
            f"Parameters: Q0={q0_params/1e6:.1f}M, "
            f"Q1={q1_params/1e6:.1f}M, Q2={q2_params/1e6:.1f}M, "
            f"CFB_Q1={cfb_q1_params/1e6:.1f}M, CFB_Q2={cfb_q2_params/1e6:.1f}M, "
            f"Total={total_params/1e6:.1f}M\n"
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _corrupt_per_time(self, motion_codes: torch.Tensor) -> torch.Tensor:
        B, T, num_q = motion_codes.shape
        device = motion_codes.device
        corrupt_mask = torch.rand(B, T, device=device) < self.corrupt_ratio
        random_codes = torch.randint(0, self.num_vq, (B, T, num_q), device=device)
        corrupt_mask = corrupt_mask.unsqueeze(-1).expand_as(motion_codes)
        return torch.where(corrupt_mask, random_codes, motion_codes)

    def _project_cond(self, conditioning_features: torch.Tensor) -> torch.Tensor:
        if conditioning_features.dim() == 2:
            conditioning_features = conditioning_features.unsqueeze(1)
        if self.text_proj is not None:
            return self.text_proj(conditioning_features)
        # Speech mode: already projected upstream by audio_proj
        return conditioning_features

    def _fuse_q1(self, q0_emb: torch.Tensor, cond_ctx: torch.Tensor,
                 cond_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """CFB stack for Q1: stream=Q0_emb, cond=speech/text."""
        if not self.q1_use_cond:
            return self.cfb_q1.ln_out(q0_emb)
        return self.cfb_q1(q0_emb, cond_ctx, cond_mask)

    def _fuse_q2(self, q0_emb: torch.Tensor, q1_emb: torch.Tensor,
                 cond_ctx: torch.Tensor,
                 cond_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """CFB stack for Q2: stream=cat(Q0_emb, Q1_emb), cond=speech/text."""
        stream = torch.cat([q0_emb, q1_emb], dim=1)  # (B, 2T', D)
        if not self.q2_use_cond:
            return self.cfb_q2.ln_out(stream)
        return self.cfb_q2(stream, cond_ctx, cond_mask)

    # ----------------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        motion_codes: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pkeep: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pkeep is None:
            pkeep = self.pkeep

        if motion_codes.dim() == 3 and motion_codes.shape[-1] == 6:
            motion_codes = motion_codes[:, :, :3]

        if self.training and self.corrupt_ratio > 0:
            input_codes = self._corrupt_per_time(motion_codes)
        else:
            input_codes = motion_codes

        target_q0 = motion_codes[:, :, 0]
        target_q1 = motion_codes[:, :, 1]
        target_q2 = motion_codes[:, :, 2]

        motion_q0 = input_codes[:, :, 0]
        motion_q1 = input_codes[:, :, 1]
        motion_q2 = input_codes[:, :, 2]

        B, T = motion_q0.shape
        device = motion_q0.device

        if self.training and pkeep < 1.0:
            mask_q0 = torch.bernoulli(pkeep * torch.ones_like(motion_q0, dtype=torch.float)).long()
            motion_q0 = mask_q0 * motion_q0 + (1 - mask_q0) * torch.randint_like(motion_q0, self.num_vq)
            mask_q1 = torch.bernoulli(pkeep * torch.ones_like(motion_q1, dtype=torch.float)).long()
            motion_q1 = mask_q1 * motion_q1 + (1 - mask_q1) * torch.randint_like(motion_q1, self.num_vq)
            mask_q2 = torch.bernoulli(pkeep * torch.ones_like(motion_q2, dtype=torch.float)).long()
            motion_q2 = mask_q2 * motion_q2 + (1 - mask_q2) * torch.randint_like(motion_q2, self.num_vq)

        cond_ctx = self._project_cond(text_features)

        # --- Q0 ---
        if self.q0_decoder_type == 'llamagen':
            logits_q0 = self.q0_decoder(
                idx=motion_q0, cond_context=cond_ctx, cond_mask=text_mask,
            )
        else:
            logits_q0 = self.q0_decoder(
                idx=motion_q0, text_context=cond_ctx, text_mask=text_mask,
            )

        # Q0 tokens for conditioning Q1
        if self.training and pkeep < 1.0:
            pred_q0 = logits_q0.argmax(dim=-1)
            sel = (torch.rand(B, device=device) < pkeep).unsqueeze(1).expand(-1, T)
            q0_for_cond = torch.where(sel, target_q0, pred_q0)
        else:
            q0_for_cond = target_q0
        q0_emb = self.q0_decoder.get_embeddings(q0_for_cond)

        # --- CFB Q1 + Q1 decoder ---
        fused_ctx_q1 = self._fuse_q1(q0_emb, cond_ctx, text_mask if self.q1_use_cond else None)
        logits_q1 = self.q1_decoder(
            idx=motion_q1,
            text_context=fused_ctx_q1,
            text_mask=None,  # fused_ctx_q1 length == T', no padding
        )

        # Q1 tokens for conditioning Q2
        if self.training and pkeep < 1.0:
            pred_q1 = logits_q1.argmax(dim=-1)
            sel = (torch.rand(B, device=device) < pkeep).unsqueeze(1).expand(-1, T)
            q1_for_cond = torch.where(sel, target_q1, pred_q1)
        else:
            q1_for_cond = target_q1
        q1_emb = self.q1_decoder.get_embeddings(q1_for_cond)

        # --- CFB Q2 + Q2 decoder ---
        fused_ctx_q2 = self._fuse_q2(q0_emb, q1_emb, cond_ctx, text_mask if self.q2_use_cond else None)
        logits_q2 = self.q2_decoder(
            idx=motion_q2,
            text_context=fused_ctx_q2,
            text_mask=None,
        )

        return logits_q0, logits_q1, logits_q2

    # ----------------------------------------------------------------------
    # Generation (full-forward, no KV cache for simplicity)
    # ----------------------------------------------------------------------
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
        B = text_features.shape[0]
        device = text_features.device

        cond_ctx = self._project_cond(text_features)
        max_len = min(max_len, self.block_size)

        q0_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q1_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)
        q2_tokens = torch.zeros((B, 0), dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(max_len):
            # Q0
            q0_input = q0_tokens if q0_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device)
            if self.q0_decoder_type == 'llamagen':
                logits_q0 = self.q0_decoder(idx=q0_input, cond_context=cond_ctx, cond_mask=text_mask)
            else:
                logits_q0 = self.q0_decoder(idx=q0_input, text_context=cond_ctx, text_mask=text_mask)
            logits_q0_t = logits_q0[:, -1, :] / temperature
            q0_t = self._sample(logits_q0_t, top_k=top_k, top_p=top_p) if do_sample else logits_q0_t.argmax(dim=-1)

            eos_mask = (q0_t == self.eos_token_id) & ~finished
            lengths[eos_mask] = t
            finished = finished | eos_mask
            if finished.all():
                break

            q0_t = torch.where(finished, torch.zeros_like(q0_t), q0_t)
            q0_tokens = torch.cat([q0_tokens, q0_t.unsqueeze(1)], dim=1)
            q0_emb = self.q0_decoder.get_embeddings(q0_tokens)

            # Q1 via CFB pre-fusion
            fused_ctx_q1 = self._fuse_q1(q0_emb, cond_ctx, text_mask if self.q1_use_cond else None)
            q1_input = q1_tokens if q1_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device)
            logits_q1 = self.q1_decoder(
                idx=q1_input, text_context=fused_ctx_q1, text_mask=None,
            )
            logits_q1_t = logits_q1[:, -1, :] / temperature
            q1_t = self._sample(logits_q1_t, top_k=top_k, top_p=top_p) if do_sample else logits_q1_t.argmax(dim=-1)
            q1_t = torch.where(finished, torch.zeros_like(q1_t), q1_t)
            q1_tokens = torch.cat([q1_tokens, q1_t.unsqueeze(1)], dim=1)
            q1_emb = self.q1_decoder.get_embeddings(q1_tokens)

            # Q2 via CFB pre-fusion
            fused_ctx_q2 = self._fuse_q2(q0_emb, q1_emb, cond_ctx, text_mask if self.q2_use_cond else None)
            q2_input = q2_tokens if q2_tokens.shape[1] > 0 else torch.zeros((B, 1), dtype=torch.long, device=device)
            logits_q2 = self.q2_decoder(
                idx=q2_input, text_context=fused_ctx_q2, text_mask=None,
            )
            logits_q2_t = logits_q2[:, -1, :] / temperature
            q2_t = self._sample(logits_q2_t, top_k=top_k, top_p=top_p) if do_sample else logits_q2_t.argmax(dim=-1)
            q2_t = torch.where(finished, torch.zeros_like(q2_t), q2_t)
            q2_tokens = torch.cat([q2_tokens, q2_t.unsqueeze(1)], dim=1)

        lengths[~finished] = q0_tokens.shape[1]

        if q0_tokens.shape[1] > 0:
            generated_codes = torch.stack([q0_tokens, q1_tokens, q2_tokens], dim=-1)
        else:
            generated_codes = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
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

    # ----------------------------------------------------------------------
    # MotionGPT training interface (mirrors HierarchicalRVQGPT)
    # ----------------------------------------------------------------------
    def encode_text(self, texts: List[str]):
        if self.text_encoder is None:
            raise ValueError("Text encoder not initialized. Set use_speech=False to use text mode.")
        return self.text_encoder(texts)

    def encode_audio(self, audio_waveforms: torch.Tensor):
        if self.speech_encoder is None:
            raise ValueError("Speech encoder not initialized. Set use_speech=True to use speech mode.")
        return self.speech_encoder(audio_waveforms)

    def project_audio(self, audio_features: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        if self.use_contrastive_proj:
            x = self.contrastive_input_proj(audio_features)
            x = self.contrastive_pos_enc(x)
            pad_mask = ~audio_mask.bool() if audio_mask is not None else None
            x = self.contrastive_transformer(x, src_key_padding_mask=pad_mask)
            return self.audio_proj(x)
        return self.audio_proj(audio_features)

    def __call__(self, texts=None, audio_waveforms=None, motion_tokens=None, lengths=None, tasks=None,
                 speech_feats=None, speech_mask=None):
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

        if motion_tokens.dim() == 3 and motion_tokens.shape[-1] == 6:
            motion_tokens = motion_tokens[:, :, :3]

        B, T, _ = motion_tokens.shape
        device = motion_tokens.device

        targets_q0 = torch.full((B, T), -1, dtype=torch.long, device=device)
        targets_q1 = torch.full((B, T), -1, dtype=torch.long, device=device)
        targets_q2 = torch.full((B, T), -1, dtype=torch.long, device=device)

        for i in range(B):
            seq_len = min(lengths[i], T - 1) if lengths is not None else T - 1
            targets_q0[i, :seq_len] = motion_tokens[i, 1:seq_len + 1, 0]
            targets_q1[i, :seq_len] = motion_tokens[i, 1:seq_len + 1, 1]
            targets_q2[i, :seq_len] = motion_tokens[i, 1:seq_len + 1, 2]
            if seq_len < T:
                targets_q0[i, seq_len] = self.eos_token_id
                targets_q1[i, seq_len] = self.eos_token_id
                targets_q2[i, seq_len] = self.eos_token_id

        logits_q0, logits_q1, logits_q2 = self.forward(
            motion_tokens, conditioning_features, text_mask=conditioning_mask,
        )

        logits_q0 = logits_q0[:, :-1, :]
        logits_q1 = logits_q1[:, :-1, :]
        logits_q2 = logits_q2[:, :-1, :]

        targets_q0 = targets_q0[:, :T - 1]
        targets_q1 = targets_q1[:, :T - 1]
        targets_q2 = targets_q2[:, :T - 1]

        loss_q0 = F.cross_entropy(
            logits_q0.reshape(-1, logits_q0.size(-1)), targets_q0.reshape(-1), ignore_index=-1,
        )
        loss_q1 = F.cross_entropy(
            logits_q1.reshape(-1, logits_q1.size(-1)), targets_q1.reshape(-1), ignore_index=-1,
        )
        loss_q2 = F.cross_entropy(
            logits_q2.reshape(-1, logits_q2.size(-1)), targets_q2.reshape(-1), ignore_index=-1,
        )
        total_loss = loss_q0 + loss_q1 + loss_q2

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

    def generate_conditional(self, texts=None, lengths=None, stage='test', tasks=None,
                             audio_waveforms=None, **kwargs):
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

        if lengths is not None:
            if isinstance(lengths, list):
                max_len = max(lengths) // 4
            else:
                max_len = int(lengths.max().item()) // 4
        else:
            max_len = self.block_size
        max_len = min(max_len, self.block_size)

        generated_codes, gen_lengths = self.generate(
            text_features=conditioning_features,
            text_mask=conditioning_mask,
            max_len=max_len,
            do_sample=True,
            temperature=1.0,
        )

        outputs_tokens = []
        for i in range(generated_codes.shape[0]):
            actual_len = gen_lengths[i]
            if actual_len > 0:
                outputs_tokens.append(generated_codes[i, :actual_len, :])
            else:
                outputs_tokens.append(generated_codes[i, :1, :])
        return outputs_tokens
