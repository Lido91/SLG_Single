import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mGPT.archs.mgpt_rvq import RVQVae
from mGPT.archs.mgpt_rvq_hierarchical import TEXT_ENCODER_CONFIGS, TextEncoder
from mGPT.archs.pos_encoding import PositionEmbedding
from mGPT.archs.speech_encoder import SPEECH_ENCODER_CONFIGS, SpeechEncoder

from contrastive.loss import symmetric_infonce


class TransformerProjectionHead(nn.Module):
    """Linear -> PosEnc -> TransformerEncoder -> MaskedMeanPool -> L2Norm."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 512,
        num_layers: int = 4,
        nhead: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_enc = PositionEmbedding(max_seq_len, proj_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        if x.shape[1] > self.pos_enc.embed.shape[0]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max_seq_len "
                f"{self.pos_enc.embed.shape[0]} in TransformerProjectionHead."
            )
        x = self.pos_enc(x)

        pad_mask = None if mask is None else ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)

        return F.normalize(x.float(), dim=-1, eps=1e-6)


class NewContrastiveCodexModel(pl.LightningModule):
    """
    Triplet contrastive pipeline:
    speech waveform -> frozen HuBERT
    text string -> frozen CLIP text encoder
    raw motion -> frozen RVQVae encoder
    each modality -> shared transformer projection head
    pairwise symmetric InfoNCE over (speech, text, motion)
    """

    def __init__(
        self,
        speech_encoder_type: str = "hubert-large",
        text_encoder_type: str = "clip",
        vae_config: Optional[Dict] = None,
        vae_checkpoint: Optional[str] = None,
        proj_dim: int = 512,
        head_num_layers: int = 4,
        head_nhead: int = 8,
        head_ff_dim: int = 2048,
        head_dropout: float = 0.1,
        speech_max_seq_len: int = 1500,
        text_max_seq_len: int = 77,
        motion_max_seq_len: int = 128,
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

        if not enable_speech and not enable_text:
            raise ValueError("At least one of enable_speech or enable_text must be True")

        if vae_config is None or "params" not in vae_config:
            raise ValueError("vae_config with a 'params' field is required")

        if not preload_features:
            if not vae_checkpoint:
                raise ValueError("vae_checkpoint is required when preload_features=False")
            if enable_speech:
                self.speech_encoder = SpeechEncoder(speech_encoder_type, freeze=True)
            if enable_text:
                self.text_encoder = TextEncoder(text_encoder_type, freeze=True)
            self.motion_encoder = self._load_frozen_vae(vae_config, vae_checkpoint)

        motion_dim = vae_config["params"]["code_dim"]

        if enable_speech:
            speech_dim = SPEECH_ENCODER_CONFIGS[speech_encoder_type]["dim"]
            self.speech_head = TransformerProjectionHead(
                speech_dim,
                proj_dim=proj_dim,
                num_layers=head_num_layers,
                nhead=head_nhead,
                ff_dim=head_ff_dim,
                dropout=head_dropout,
                max_seq_len=speech_max_seq_len,
            )
        if enable_text:
            text_dim = TEXT_ENCODER_CONFIGS[text_encoder_type]["dim"]
            self.text_head = TransformerProjectionHead(
                text_dim,
                proj_dim=proj_dim,
                num_layers=head_num_layers,
                nhead=head_nhead,
                ff_dim=head_ff_dim,
                dropout=head_dropout,
                max_seq_len=text_max_seq_len,
            )
        self.motion_head = TransformerProjectionHead(
            motion_dim,
            proj_dim=proj_dim,
            num_layers=head_num_layers,
            nhead=head_nhead,
            ff_dim=head_ff_dim,
            dropout=head_dropout,
            max_seq_len=motion_max_seq_len,
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))

        stride_t = vae_config["params"].get("stride_t", 2)
        down_t = vae_config["params"].get("down_t", 2)
        self._motion_compression = stride_t ** down_t
        self._speech_stride = 160 if "whisper" in speech_encoder_type else 320

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.preload_features:
            if self.enable_speech:
                self.speech_encoder.eval()
            if self.enable_text:
                self.text_encoder.eval()
            self.motion_encoder.eval()
        return self

    @staticmethod
    def _load_frozen_vae(vae_config: dict, vae_checkpoint: str) -> RVQVae:
        params = vae_config["params"]
        vae = RVQVae(**params)

        if not os.path.exists(vae_checkpoint):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

        ckpt = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = {}
            for key, value in ckpt["state_dict"].items():
                if key.startswith("vae."):
                    state_dict[key[len("vae."):]] = value
            if not state_dict:
                raise RuntimeError(
                    f"Checkpoint {vae_checkpoint} does not contain any 'vae.' weights."
                )
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

        load_result = vae.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "VAE checkpoint mismatch. "
                f"Missing: {load_result.missing_keys}; Unexpected: {load_result.unexpected_keys}"
            )

        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    def encode_speech(
        self,
        audio_waveforms: torch.Tensor,
        audio_lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            feats, mask = self.speech_encoder(audio_waveforms)

        if audio_lengths is not None:
            batch_size, num_frames = feats.shape[:2]
            mask = torch.zeros(batch_size, num_frames, device=feats.device)
            for idx, length in enumerate(audio_lengths):
                if isinstance(length, torch.Tensor):
                    length = length.item()
                valid = min(length // self._speech_stride, num_frames)
                mask[idx, :valid] = 1.0

        return self.speech_head(feats, mask)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            feats, mask = self.text_encoder(texts)
        return self.text_head(feats, mask)

    def encode_motion(self, raw_motion: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        with torch.no_grad():
            motion = self.motion_encoder.preprocess(raw_motion)
            motion = self.motion_encoder.encoder(motion)
            motion = motion.permute(0, 2, 1)

        mask = self._build_motion_mask(lengths, motion.shape[1], motion.device)
        return self.motion_head(motion, mask)

    def _build_motion_mask(
        self,
        lengths: List[int],
        encoded_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(len(lengths), encoded_len, device=device)
        for idx, length in enumerate(lengths):
            if isinstance(length, torch.Tensor):
                length = length.item()
            valid = min(math.ceil(length / self._motion_compression), encoded_len)
            mask[idx, :valid] = 1.0
        return mask

    def _encode_batch(self, batch: Dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        speech_emb = None
        text_emb = None
        if self.preload_features:
            if self.enable_speech:
                speech_emb = self.speech_head(batch["speech_feats"], batch["speech_mask"])
            if self.enable_text:
                text_emb = self.text_head(batch["text_feats"], batch["text_mask"])
            motion_emb = self.motion_head(batch["motion_feats"], batch["motion_mask"])
        else:
            if self.enable_speech:
                speech_emb = self.encode_speech(batch["audio"], batch.get("audio_length"))
            if self.enable_text:
                text_emb = self.encode_text(batch["text"])
            motion_emb = self.encode_motion(batch["motion"], batch["length"])
        return speech_emb, text_emb, motion_emb

    @staticmethod
    def _ensure_finite(name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is not None and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite values detected in {name}")

    def _compute_losses(
        self,
        speech_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        motion_emb: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        self._ensure_finite("motion_emb_pre_gather", motion_emb)
        if speech_emb is not None:
            self._ensure_finite("speech_emb_pre_gather", speech_emb)
        if text_emb is not None:
            self._ensure_finite("text_emb_pre_gather", text_emb)

        if self.trainer.world_size > 1:
            if speech_emb is not None:
                speech_emb = self.all_gather(speech_emb, sync_grads=True).flatten(0, 1)
            if text_emb is not None:
                text_emb = self.all_gather(text_emb, sync_grads=True).flatten(0, 1)
            motion_emb = self.all_gather(motion_emb, sync_grads=True).flatten(0, 1)

        with torch.amp.autocast("cuda", enabled=False):
            scale = self.logit_scale.float().clamp(max=4.6052).exp()
        self._ensure_finite("logit_scale", scale)

        losses = {}
        loss_terms = []
        if speech_emb is not None and text_emb is not None:
            loss_st = symmetric_infonce(speech_emb, text_emb, scale)
            self._ensure_finite(f"{prefix}/loss_st", loss_st)
            losses[f"{prefix}/loss_st"] = loss_st
            loss_terms.append(loss_st)
        if speech_emb is not None:
            loss_sm = symmetric_infonce(speech_emb, motion_emb, scale)
            self._ensure_finite(f"{prefix}/loss_sm", loss_sm)
            losses[f"{prefix}/loss_sm"] = loss_sm
            loss_terms.append(loss_sm)
        if text_emb is not None:
            loss_tm = symmetric_infonce(text_emb, motion_emb, scale)
            self._ensure_finite(f"{prefix}/loss_tm", loss_tm)
            losses[f"{prefix}/loss_tm"] = loss_tm
            loss_terms.append(loss_tm)

        total = sum(loss_terms) / len(loss_terms)
        self._ensure_finite(f"{prefix}/loss", total)
        losses[f"{prefix}/loss"] = total
        losses[f"{prefix}/logit_scale"] = scale

        batch_size = motion_emb.shape[0]
        log_kwargs = dict(prog_bar=True, sync_dist=True, batch_size=batch_size)
        if prefix == "train":
            log_kwargs.update(on_step=True, on_epoch=True)
        else:
            log_kwargs.update(on_step=False, on_epoch=True)

        self.log_dict(losses, **log_kwargs)
        return total

    def training_step(self, batch: Dict, batch_idx: int):
        if batch is None:
            return None
        speech_emb, text_emb, motion_emb = self._encode_batch(batch)
        return self._compute_losses(speech_emb, text_emb, motion_emb, "train")

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6052)

    def on_validation_epoch_start(self):
        if self.enable_speech:
            self._val_speech_embs = []
        if self.enable_text:
            self._val_text_embs = []
        self._val_motion_embs = []

    def validation_step(self, batch: Dict, batch_idx: int):
        if batch is None:
            return None

        speech_emb, text_emb, motion_emb = self._encode_batch(batch)
        if speech_emb is not None:
            self._val_speech_embs.append(speech_emb.detach())
        if text_emb is not None:
            self._val_text_embs.append(text_emb.detach())
        self._val_motion_embs.append(motion_emb.detach())
        return self._compute_losses(speech_emb, text_emb, motion_emb, "val")

    def on_validation_epoch_end(self):
        if not self._val_motion_embs:
            return

        speech_emb = (torch.cat(self._val_speech_embs, dim=0)
                      if self.enable_speech else None)
        text_emb = (torch.cat(self._val_text_embs, dim=0)
                    if self.enable_text else None)
        motion_emb = torch.cat(self._val_motion_embs, dim=0)

        if self.trainer.world_size > 1:
            if speech_emb is not None:
                speech_emb = self.all_gather(speech_emb).flatten(0, 1)
            if text_emb is not None:
                text_emb = self.all_gather(text_emb).flatten(0, 1)
            motion_emb = self.all_gather(motion_emb).flatten(0, 1)

        if speech_emb is not None:
            speech_emb = speech_emb.cpu()
        if text_emb is not None:
            text_emb = text_emb.cpu()
        motion_emb = motion_emb.cpu()

        pairs = []
        if speech_emb is not None and text_emb is not None:
            pairs += [("S2T", speech_emb, text_emb), ("T2S", text_emb, speech_emb)]
        if speech_emb is not None:
            pairs += [("S2M", speech_emb, motion_emb), ("M2S", motion_emb, speech_emb)]
        if text_emb is not None:
            pairs += [("T2M", text_emb, motion_emb), ("M2T", motion_emb, text_emb)]

        metrics = {}
        r1_all, r5_all, r10_all = [], [], []
        for name, query, gallery in pairs:
            r1, r5, r10 = self._recall_at_k(query, gallery)
            metrics[f"val/{name}_R@1"] = r1
            metrics[f"val/{name}_R@5"] = r5
            metrics[f"val/{name}_R@10"] = r10
            r1_all.append(r1)
            r5_all.append(r5)
            r10_all.append(r10)

        metrics["val/avg_R@1"] = float(np.mean(r1_all))
        metrics["val/avg_R@5"] = float(np.mean(r5_all))
        metrics["val/avg_R@10"] = float(np.mean(r10_all))
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        if self.enable_speech:
            self._val_speech_embs.clear()
        if self.enable_text:
            self._val_text_embs.clear()
        self._val_motion_embs.clear()

    @staticmethod
    def _recall_at_k(
        query: torch.Tensor,
        gallery: torch.Tensor,
        ks: Tuple[int, int, int] = (1, 5, 10),
    ) -> Tuple[float, float, float]:
        sim = query @ gallery.T
        sorted_indices = sim.argsort(dim=1, descending=True)
        gt = torch.arange(sim.shape[0]).unsqueeze(1)
        ranks = (sorted_indices == gt).nonzero(as_tuple=True)[1]
        return tuple((ranks < k).float().mean().item() * 100.0 for k in ks)

    def configure_optimizers(self):
        proj_params = list(self.motion_head.parameters())
        if self.enable_speech:
            proj_params += list(self.speech_head.parameters())
        if self.enable_text:
            proj_params += list(self.text_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": proj_params, "weight_decay": float(self.hparams.weight_decay)},
                {"params": [self.logit_scale], "weight_decay": 0.0},
            ],
            lr=float(self.hparams.lr),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams.max_epochs),
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]
