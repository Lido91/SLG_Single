"""
Triplet Contrastive Model with Transformer-based Projection Heads.

Architecture (per modality):
    Frozen Encoder -> Linear(D_in, 512) -> + PosEmb ->
    TransformerEncoder(4L, 8H, FFN=2048) -> MaskedMeanPool -> L2Norm -> [B, 512]

Loss: Symmetric InfoNCE on all 3 pairs: Speech<->Text, Speech<->Motion, Text<->Motion
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mGPT.archs.speech_encoder import SpeechEncoder, SPEECH_ENCODER_CONFIGS
from mGPT.archs.mgpt_rvq_hierarchical import TextEncoder, TEXT_ENCODER_CONFIGS
from mGPT.archs.mgpt_rvq import RVQVae

from contrastive.loss import symmetric_infonce


class ModalityProjectionHead(nn.Module):
    """
    Shared architecture for all three modality projections.

    Pipeline:
        Linear(input_dim -> proj_dim)
        + Learnable Positional Embedding
        TransformerEncoder (num_layers, nhead, dim_feedforward)
        Masked Mean Pooling
        L2 Normalize  ->  [B, proj_dim]
    """

    def __init__(self, input_dim, proj_dim=512, num_layers=4, nhead=8,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, proj_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, T, input_dim]
            mask: [B, T] float (1=valid, 0=pad) or None

        Returns:
            [B, proj_dim] L2-normalized embedding
        """
        B, T, _ = x.shape
        x = self.input_proj(x)                       # [B, T, proj_dim]
        x = x + self.pos_embedding[:, :T, :]         # add positional info

        # TransformerEncoder: key_padding_mask True = ignore
        kpm = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=kpm)  # [B, T, proj_dim]

        # Masked mean pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return F.normalize(x.float(), dim=-1)


class NewTripletContrastiveModel(pl.LightningModule):
    """
    Triplet contrastive model: Speech + Text + Motion -> shared 512-D space.

    Frozen backbones:
        - SpeechEncoder  (HuBERT-Large, 1024-D)
        - TextEncoder    (CLIP, 512-D)
        - RVQVae.encoder (stride=2, down=2, 512-D)

    Trainable:
        - 3 x ModalityProjectionHead
        - 1 x learnable logit_scale (CLIP-style)
    """

    def __init__(
        self,
        speech_encoder_type: str = 'hubert-large',
        text_encoder_type: str = 'clip',
        vae_config: dict = None,
        vae_checkpoint: str = None,
        proj_dim: int = 512,
        num_proj_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        temperature_init: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        preload_features: bool = False,
        enable_speech: bool = True,
        enable_text: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.preload_features = preload_features
        self.enable_speech = enable_speech
        self.enable_text = enable_text

        assert enable_speech or enable_text, "At least one of speech or text must be enabled"
        if vae_config is None or 'params' not in vae_config:
            raise ValueError("vae_config with 'params' field is required")

        # --- Frozen backbone encoders (skipped when precomputed) ---
        if not preload_features:
            if vae_checkpoint is None:
                raise ValueError("vae_checkpoint required when preload_features=False")
            if enable_speech:
                self.speech_encoder = SpeechEncoder(speech_encoder_type, freeze=True)
            if enable_text:
                self.text_encoder = TextEncoder(text_encoder_type, freeze=True)
            self.motion_encoder = self._load_frozen_vae(vae_config, vae_checkpoint)

        # --- Trainable projection heads ---
        proj_kwargs = dict(
            proj_dim=proj_dim,
            num_layers=num_proj_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        motion_dim = vae_config['params']['code_dim']
        self.motion_proj = ModalityProjectionHead(motion_dim, **proj_kwargs)

        if enable_speech:
            speech_dim = SPEECH_ENCODER_CONFIGS[speech_encoder_type]['dim']
            self.speech_proj = ModalityProjectionHead(speech_dim, **proj_kwargs)

        if enable_text:
            text_dim = TEXT_ENCODER_CONFIGS[text_encoder_type]['dim']
            self.text_proj = ModalityProjectionHead(text_dim, **proj_kwargs)

        # Learnable log-temperature (CLIP-style)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

        # VAE temporal compression factor
        stride_t = vae_config['params'].get('stride_t', 2)
        down_t = vae_config['params'].get('down_t', 2)
        self._temporal_compression = stride_t ** down_t  # typically 4

        # Speech encoder stride for mask computation
        self._speech_encoder_type = speech_encoder_type
        if 'whisper' in speech_encoder_type:
            self._speech_stride = 160
        else:
            self._speech_stride = 320  # HuBERT / WavLM / Wav2Vec2

    def train(self, mode: bool = True):
        """Keep frozen encoders in eval mode."""
        super().train(mode)
        if not self.preload_features:
            if self.enable_speech:
                self.speech_encoder.eval()
            if self.enable_text:
                self.text_encoder.eval()
            self.motion_encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Frozen VAE loading
    # ------------------------------------------------------------------
    def _load_frozen_vae(self, vae_config, vae_checkpoint):
        params = vae_config['params']
        vae = RVQVae(**params)

        if not vae_checkpoint or not os.path.exists(vae_checkpoint):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

        ckpt = torch.load(vae_checkpoint, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = {
                k[len('vae.'):]: v
                for k, v in ckpt['state_dict'].items()
                if k.startswith('vae.')
            }
            if not state_dict:
                raise RuntimeError(f"No 'vae.' keys in checkpoint {vae_checkpoint}")
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

        load_result = vae.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                f"VAE state_dict mismatch. Missing: {load_result.missing_keys}; "
                f"Unexpected: {load_result.unexpected_keys}"
            )

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    # ------------------------------------------------------------------
    # Per-modality encoding
    # ------------------------------------------------------------------
    def encode_speech(self, audio_waveforms, audio_lengths=None):
        with torch.no_grad():
            feats, mask = self.speech_encoder(audio_waveforms)
        if audio_lengths is not None:
            B, T_s = feats.shape[:2]
            mask = torch.zeros(B, T_s, device=feats.device)
            for i, al in enumerate(audio_lengths):
                al = al.item() if isinstance(al, torch.Tensor) else al
                mask[i, :min(al // self._speech_stride, T_s)] = 1.0
        return self.speech_proj(feats, mask)

    def encode_text(self, texts):
        with torch.no_grad():
            feats, mask = self.text_encoder(texts)
        return self.text_proj(feats, mask)

    def encode_motion(self, raw_motion, lengths):
        with torch.no_grad():
            x = self.motion_encoder.preprocess(raw_motion)  # [B, 133, T]
            x = self.motion_encoder.encoder(x)               # [B, 512, T']
            x = x.permute(0, 2, 1)                           # [B, T', 512]
        mask = self._compute_motion_mask(lengths, x.shape[1], x.device)
        return self.motion_proj(x, mask)

    def _compute_motion_mask(self, lengths, T_prime, device):
        B = len(lengths)
        mask = torch.zeros(B, T_prime, device=device)
        for i, l in enumerate(lengths):
            l = l.item() if isinstance(l, torch.Tensor) else l
            mask[i, :min(math.ceil(l / self._temporal_compression), T_prime)] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Encode all modalities (precomputed or live)
    # ------------------------------------------------------------------
    def _encode_all(self, batch):
        speech_emb, text_emb = None, None
        if self.preload_features:
            if self.enable_speech:
                speech_emb = self.speech_proj(batch['speech_feats'], batch['speech_mask'])
            if self.enable_text:
                text_emb = self.text_proj(batch['text_feats'], batch['text_mask'])
            motion_emb = self.motion_proj(batch['motion_feats'], batch['motion_mask'])
        else:
            if self.enable_speech:
                speech_emb = self.encode_speech(
                    batch['audio'], batch.get('audio_length', None))
            if self.enable_text:
                text_emb = self.encode_text(batch['text'])
            motion_emb = self.encode_motion(batch['motion'], batch['length'])
        return speech_emb, text_emb, motion_emb

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch, prefix, precomputed_embs=None):
        if precomputed_embs is not None:
            speech_emb, text_emb, motion_emb = precomputed_embs
        else:
            speech_emb, text_emb, motion_emb = self._encode_all(batch)

        # Gather across GPUs for full negative pool
        if self.trainer.world_size > 1:
            if speech_emb is not None:
                speech_emb = self.all_gather(speech_emb, sync_grads=True).flatten(0, 1)
            if text_emb is not None:
                text_emb = self.all_gather(text_emb, sync_grads=True).flatten(0, 1)
            motion_emb = self.all_gather(motion_emb, sync_grads=True).flatten(0, 1)

        # Compute logit scale in fp32, clamp before exp
        with torch.amp.autocast('cuda', enabled=False):
            scale = self.logit_scale.float().clamp(max=4.6052).exp()

        losses = {}
        loss_terms = []
        if speech_emb is not None and text_emb is not None:
            losses[f'{prefix}/loss_st'] = symmetric_infonce(speech_emb, text_emb, scale)
            loss_terms.append(losses[f'{prefix}/loss_st'])
        if speech_emb is not None:
            losses[f'{prefix}/loss_sm'] = symmetric_infonce(speech_emb, motion_emb, scale)
            loss_terms.append(losses[f'{prefix}/loss_sm'])
        if text_emb is not None:
            losses[f'{prefix}/loss_tm'] = symmetric_infonce(text_emb, motion_emb, scale)
            loss_terms.append(losses[f'{prefix}/loss_tm'])

        total = sum(loss_terms) / len(loss_terms)

        batch_size = motion_emb.shape[0]
        log_kwargs = dict(prog_bar=True, sync_dist=True, batch_size=batch_size)
        if prefix == 'train':
            log_kwargs.update(on_step=True, on_epoch=True)
        else:
            log_kwargs.update(on_step=False, on_epoch=True)

        losses[f'{prefix}/loss'] = total
        losses[f'{prefix}/logit_scale'] = scale
        self.log_dict(losses, **log_kwargs)
        return total

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        return self._shared_step(batch, 'train')

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6052)

    # ------------------------------------------------------------------
    # Validation with retrieval metrics
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        if self.enable_speech:
            self._val_speech_embs = []
        if self.enable_text:
            self._val_text_embs = []
        self._val_motion_embs = []

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        speech_emb, text_emb, motion_emb = self._encode_all(batch)
        if speech_emb is not None:
            self._val_speech_embs.append(speech_emb.detach())
        if text_emb is not None:
            self._val_text_embs.append(text_emb.detach())
        self._val_motion_embs.append(motion_emb.detach())
        return self._shared_step(batch, 'val',
                                 precomputed_embs=(speech_emb, text_emb, motion_emb))

    def on_validation_epoch_end(self):
        if not self._val_motion_embs:
            return

        motion_emb = torch.cat(self._val_motion_embs, dim=0)
        speech_emb = (torch.cat(self._val_speech_embs, dim=0)
                      if self.enable_speech else None)
        text_emb = (torch.cat(self._val_text_embs, dim=0)
                    if self.enable_text else None)

        if self.trainer.world_size > 1:
            motion_emb = self.all_gather(motion_emb).flatten(0, 1)
            if speech_emb is not None:
                speech_emb = self.all_gather(speech_emb).flatten(0, 1)
            if text_emb is not None:
                text_emb = self.all_gather(text_emb).flatten(0, 1)

        motion_emb = motion_emb.cpu()
        if speech_emb is not None:
            speech_emb = speech_emb.cpu()
        if text_emb is not None:
            text_emb = text_emb.cpu()

        pairs = []
        if speech_emb is not None and text_emb is not None:
            pairs += [('S2T', speech_emb, text_emb), ('T2S', text_emb, speech_emb)]
        if speech_emb is not None:
            pairs += [('S2M', speech_emb, motion_emb), ('M2S', motion_emb, speech_emb)]
        if text_emb is not None:
            pairs += [('T2M', text_emb, motion_emb), ('M2T', motion_emb, text_emb)]

        all_r1, all_r5, all_r10 = [], [], []
        metrics = {}
        for name, q, g in pairs:
            r1, r5, r10 = self._recall_at_k(q, g, ks=(1, 5, 10))
            metrics[f'val/{name}_R@1'] = r1
            metrics[f'val/{name}_R@5'] = r5
            metrics[f'val/{name}_R@10'] = r10
            all_r1.append(r1); all_r5.append(r5); all_r10.append(r10)

        metrics['val/avg_R@1'] = np.mean(all_r1)
        metrics['val/avg_R@5'] = np.mean(all_r5)
        metrics['val/avg_R@10'] = np.mean(all_r10)

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        if self.enable_speech:
            self._val_speech_embs.clear()
        if self.enable_text:
            self._val_text_embs.clear()
        self._val_motion_embs.clear()

    @staticmethod
    def _recall_at_k(query, gallery, ks=(1, 5, 10)):
        sim = query @ gallery.T
        sorted_indices = sim.argsort(dim=1, descending=True)
        gt = torch.arange(sim.shape[0]).unsqueeze(1)
        ranks = (sorted_indices == gt).nonzero(as_tuple=True)[1]
        return tuple((ranks < k).float().mean().item() * 100.0 for k in ks)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        proj_params = list(self.motion_proj.parameters())
        if self.enable_speech:
            proj_params += list(self.speech_proj.parameters())
        if self.enable_text:
            proj_params += list(self.text_proj.parameters())
        optimizer = torch.optim.AdamW([
            {'params': proj_params, 'weight_decay': float(self.hparams.weight_decay)},
            {'params': [self.logit_scale], 'weight_decay': 0.0},
        ], lr=float(self.hparams.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.hparams.max_epochs), eta_min=1e-6,
        )
        return [optimizer], [scheduler]
