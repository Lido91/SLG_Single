"""
MoMask: Generative Masked Modeling for Text-to-Motion Generation

Integrates three stages:
- Stage 1: RVQ-VAE (motion tokenization)
- Stage 2: Masked Transformer (coarse motion generation from text)
- Stage 3: Residual Transformer (motion refinement)

Based on MoMask: https://arxiv.org/abs/2312.00063
Adapted for MotionGPT framework and How2Sign dataset.
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from os.path import join as pjoin

from mGPT.config import instantiate_from_config
from mGPT.models.base import BaseModel
from mGPT.losses.mgpt import GPTLosses


class MoMask(BaseModel):
    """
    MoMask: Three-stage model for text-to-motion generation.

    Stage 1 (vae): Train RVQ-VAE for motion tokenization
    Stage 2 (mask_transformer): Train Masked Transformer for Q0 generation
    Stage 3 (res_transformer): Train Residual Transformer for Q1-Q5 refinement

    Args:
        cfg: configuration object
        datamodule: data module instance
        motion_vae: RVQ-VAE config
        mask_transformer: Masked Transformer config
        res_transformer: Residual Transformer config
        stage: training stage ('vae', 'mask_transformer', 'res_transformer', 'inference')
        codebook_size: number of tokens in codebook
        num_quantizers: number of RVQ layers
    """

    def __init__(
        self,
        cfg,
        datamodule,
        motion_vae: Optional[Dict] = None,
        mask_transformer: Optional[Dict] = None,
        res_transformer: Optional[Dict] = None,
        stage: str = 'vae',
        codebook_size: int = 512,
        num_quantizers: int = 3,
        debug: bool = True,
        condition: str = 'text',
        task: str = 't2m',
        metrics_dict: List[str] = ['MRMetrics'],
        **kwargs
    ):
        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        self.stage = stage
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # Stage 1: RVQ-VAE
        if motion_vae is not None:
            self.vae = instantiate_from_config(motion_vae)
        else:
            self.vae = None

        # Stage 2: Masked Transformer
        if mask_transformer is not None:
            self.mask_transformer = instantiate_from_config(mask_transformer)
        else:
            self.mask_transformer = None

        # Stage 3: Residual Transformer
        if res_transformer is not None:
            self.res_transformer = instantiate_from_config(res_transformer)
        else:
            self.res_transformer = None

        # Freeze models based on stage
        self._setup_training_stage()

        # Instantiate losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Codebook frequency tracking
        self.codePred = []
        self.codeFrequency = torch.zeros((codebook_size,))

    def _setup_training_stage(self):
        """Configure model freezing based on training stage."""
        if self.stage == 'vae':
            # Train VAE only
            pass

        elif self.stage == 'mask_transformer':
            # Freeze VAE, train Masked Transformer
            if self.vae is not None:
                self.vae.eval()
                for p in self.vae.parameters():
                    p.requires_grad = False

        elif self.stage == 'res_transformer':
            # Freeze VAE and Masked Transformer, train Residual Transformer
            if self.vae is not None:
                self.vae.eval()
                for p in self.vae.parameters():
                    p.requires_grad = False
            if self.mask_transformer is not None:
                self.mask_transformer.eval()
                for p in self.mask_transformer.parameters():
                    p.requires_grad = False

        elif self.stage == 'inference':
            # Freeze everything
            if self.vae is not None:
                self.vae.eval()
                for p in self.vae.parameters():
                    p.requires_grad = False
            if self.mask_transformer is not None:
                self.mask_transformer.eval()
                for p in self.mask_transformer.parameters():
                    p.requires_grad = False
            if self.res_transformer is not None:
                self.res_transformer.eval()
                for p in self.res_transformer.parameters():
                    p.requires_grad = False

    def forward(self, batch, task="t2m"):
        """Full generation pipeline: text -> motion."""
        texts = batch["text"]
        lengths = batch["length"]

        # Generate motion from text
        motion = self.generate(texts, lengths)

        # Recover joints for visualization
        joints = self.feats2joints(motion)
        if isinstance(joints, tuple):
            _, joints = joints

        return {
            "texts": texts,
            "feats": motion,
            "joints": joints,
            "length": [m.shape[0] for m in motion]
        }

    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        lengths: Optional[List[int]] = None,
        timesteps: int = 10,
        cond_scale_mask: float = 4.0,
        cond_scale_res: float = 2.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9
    ) -> torch.Tensor:
        """
        Generate motion from text descriptions.

        Args:
            texts: list of text descriptions
            lengths: optional list of target motion lengths (in frames)
            timesteps: number of unmasking iterations for Stage 2
            cond_scale_mask: CFG scale for Masked Transformer
            cond_scale_res: CFG scale for Residual Transformer
            temperature: sampling temperature
            topk_filter_thres: top-k filtering threshold

        Returns:
            motion: (B, T, D) generated motion features
        """
        device = next(self.parameters()).device
        batch_size = len(texts)

        # Default lengths if not provided
        if lengths is None:
            lengths = [100] * batch_size  # Default 100 frames

        # Convert frame lengths to token lengths
        unit_length = self.datamodule.hparams.unit_length  # Usually 4
        m_lens = torch.tensor([l // unit_length for l in lengths], device=device)

        # Stage 2: Generate Q0 tokens from text
        q0_ids = self.mask_transformer.generate(
            conds=texts,
            m_lens=m_lens,
            timesteps=timesteps,
            cond_scale=cond_scale_mask,
            temperature=temperature,
            topk_filter_thres=topk_filter_thres
        )  # (B, T')

        # Stage 3: Generate Q1-Q_{num_quantizers-1} tokens
        if self.res_transformer is not None:
            all_indices = self.res_transformer.generate(
                motion_ids=q0_ids,
                conds=texts,
                m_lens=m_lens,
                temperature=temperature,
                topk_filter_thres=topk_filter_thres,
                cond_scale=cond_scale_res
            )  # (B, T', num_quantizers)
        else:
            # If no residual transformer, use Q0 only
            all_indices = q0_ids.unsqueeze(-1)  # (B, T', 1)

        # Stage 1 Decode: tokens -> motion
        motion = self.vae.decode(all_indices)  # (B, T, D)

        return motion

    # ==================== Training Forward Methods ====================

    def train_vae_forward(self, batch):
        """Stage 1: Train RVQ-VAE."""
        feats_ref = batch["motion"]

        # Get reference joints
        feats2joints_result_ref = self.feats2joints(feats_ref)
        if isinstance(feats2joints_result_ref, tuple):
            _, joints_ref = feats2joints_result_ref
        else:
            joints_ref = feats2joints_result_ref

        # VAE forward
        feats_rst, loss_commit, perplexity = self.vae(feats_ref)

        # Get reconstructed joints
        feats2joints_result_rst = self.feats2joints(feats_rst)
        if isinstance(feats2joints_result_rst, tuple):
            _, joints_rst = feats2joints_result_rst
        else:
            joints_rst = feats2joints_result_rst

        return {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
        }

    def train_mask_transformer_forward(self, batch):
        """Stage 2: Train Masked Transformer."""
        feats = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]

        # Encode motion to tokens using frozen VAE
        with torch.no_grad():
            code_idx, _ = self.vae.encode(feats)  # (B, T', num_quantizers)

        # Convert frame lengths to token lengths
        unit_length = self.datamodule.hparams.unit_length
        m_lens = torch.tensor([l // unit_length for l in lengths], device=feats.device)

        # Train Masked Transformer on Q0 tokens
        q0_ids = code_idx[..., 0]  # (B, T')
        ce_loss, pred_ids, acc = self.mask_transformer(q0_ids, texts, m_lens)

        return {
            "loss": ce_loss,
            "acc": acc,
            "pred_ids": pred_ids,
        }

    def train_res_transformer_forward(self, batch):
        """Stage 3: Train Residual Transformer."""
        feats = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]

        # Encode motion to tokens using frozen VAE
        with torch.no_grad():
            code_idx, _ = self.vae.encode(feats)  # (B, T', num_quantizers)

        # Convert frame lengths to token lengths
        unit_length = self.datamodule.hparams.unit_length
        m_lens = torch.tensor([l // unit_length for l in lengths], device=feats.device)

        # Train Residual Transformer on all quantizer layers
        ce_loss, pred_ids, acc = self.res_transformer(code_idx, texts, m_lens)

        return {
            "loss": ce_loss,
            "acc": acc,
            "pred_ids": pred_ids,
        }

    # ==================== Validation Forward Methods ====================

    @torch.no_grad()
    def val_vae_forward(self, batch, split="val"):
        """Validate RVQ-VAE reconstruction."""
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation if needed
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0
            )
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # VAE encode-decode
        feats_rst = torch.zeros_like(feats_ref)
        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

        # Get joints
        feats2joints_result_ref = self.feats2joints(feats_ref)
        feats2joints_result_rst = self.feats2joints(feats_rst)

        if isinstance(feats2joints_result_ref, tuple):
            vertices_ref, joints_ref = feats2joints_result_ref
            vertices_rst, joints_rst = feats2joints_result_rst
        else:
            joints_ref = feats2joints_result_ref
            joints_rst = feats2joints_result_rst
            vertices_ref = None
            vertices_rst = None

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        return {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
        }

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        """Validate text-to-motion generation."""
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        B = feats_ref.shape[0]

        # Repeat for multimodal evaluation if needed
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0
            )
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Generate motion from text
        motion = self.generate(texts, lengths)

        # Allocate output tensor
        max_len = motion.shape[1]
        C = self.datamodule.nfeats
        feats_rst = torch.zeros(len(texts), max_len, C, device=feats_ref.device)
        feats_rst[:, :motion.shape[1], :] = motion

        # Track generated lengths
        lengths_rst = [motion.shape[1]] * len(texts)

        # Get joints
        feats2joints_result_ref = self.feats2joints(feats_ref)
        feats2joints_result_rst = self.feats2joints(feats_rst)

        if isinstance(feats2joints_result_ref, tuple):
            vertices_ref, joints_ref = feats2joints_result_ref
            vertices_rst, joints_rst = feats2joints_result_rst
        else:
            joints_ref = feats2joints_result_ref
            joints_rst = feats2joints_result_rst
            vertices_ref = None
            vertices_rst = None

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        return {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "lengths_rst": lengths_rst,
        }

    @torch.no_grad()
    def val_res_transformer_forward(self, batch):
        """
        Validate Residual Transformer (Stage 3) with motion reconstruction metrics.

        Uses ground truth Q0 tokens to predict Q1-Q5, then decodes to motion.
        This doesn't require MaskTransformer since we use GT Q0.
        """
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]

        # Encode motion to tokens using frozen VAE
        code_idx, _ = self.vae.encode(feats_ref)  # (B, T', num_quantizers)

        # Convert frame lengths to token lengths
        unit_length = self.datamodule.hparams.unit_length
        m_lens = torch.tensor([l // unit_length for l in lengths], device=feats_ref.device)

        # Generate Q1-Q5 from GT Q0 for motion reconstruction
        q0_ids = code_idx[..., 0]  # (B, T')
        all_indices = self.res_transformer.generate(
            motion_ids=q0_ids,
            conds=texts,
            m_lens=m_lens,
            temperature=1.0,
            topk_filter_thres=0.9,
            cond_scale=2.0
        )  # (B, T', num_quantizers)

        # Clamp indices to valid range (generate() marks padding as -1)
        all_indices = all_indices.clamp(min=0)

        # Decode tokens to motion
        motion = self.vae.decode(all_indices)

        # Allocate output tensor (match reference shape)
        feats_rst = torch.zeros_like(feats_ref)
        feats_rst[:, :motion.shape[1], :] = motion

        # Get joints
        feats2joints_result_ref = self.feats2joints(feats_ref)
        feats2joints_result_rst = self.feats2joints(feats_rst)

        if isinstance(feats2joints_result_ref, tuple):
            vertices_ref, joints_ref = feats2joints_result_ref
            vertices_rst, joints_rst = feats2joints_result_rst
        else:
            joints_ref = feats2joints_result_ref
            joints_rst = feats2joints_result_rst
            vertices_ref = None
            vertices_rst = None

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        return {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
        }

    # ==================== Main Training Step ====================

    def allsplit_step(self, split: str, batch, batch_idx):
        """Main training/validation step."""
        loss = None
        lengths = batch['length']
        src = batch.get('src', None)
        name = batch.get('name', None)

        # Training
        if split == "train":
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                loss = self._losses['losses_train'].update(rs_set)

            elif self.stage == "mask_transformer":
                rs_set = self.train_mask_transformer_forward(batch)
                loss = rs_set["loss"]
                self.log("train/loss", loss, prog_bar=True, sync_dist=True)
                self.log("train/acc", rs_set["acc"], prog_bar=True, sync_dist=True)

            elif self.stage == "res_transformer":
                rs_set = self.train_res_transformer_forward(batch)
                loss = rs_set["loss"]
                self.log("train/loss", loss, prog_bar=True, sync_dist=True)
                self.log("train/acc", rs_set["acc"], prog_bar=True, sync_dist=True)

            # Log learning rate for all training stages
            if self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log("train/lr", current_lr, prog_bar=False, sync_dist=True)

        # Validation
        elif split in ["val", "test"]:
            if self.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                if hasattr(self.metrics, 'MRMetrics'):
                    self.metrics.MRMetrics.update(
                        feats_rst=rs_set["m_rst"],
                        feats_ref=rs_set["m_ref"],
                        joints_rst=rs_set["joints_rst"],
                        joints_ref=rs_set["joints_ref"],
                        vertices_rst=rs_set.get("vertices_rst"),
                        vertices_ref=rs_set.get("vertices_ref"),
                        lengths=lengths,
                        src=src,
                        name=name
                    )

            elif self.stage == "res_transformer":
                # Stage 3: Use GT Q0, validate residual prediction with motion metrics
                rs_set = self.val_res_transformer_forward(batch)
                metric_name = self.hparams.metrics_dict[0] if self.hparams.metrics_dict else 'MRMetrics'
                if hasattr(self.metrics, metric_name):
                    getattr(self.metrics, metric_name).update(
                        feats_rst=rs_set["m_rst"],
                        feats_ref=rs_set["m_ref"],
                        joints_rst=rs_set["joints_rst"],
                        joints_ref=rs_set["joints_ref"],
                        vertices_rst=rs_set.get("vertices_rst"),
                        vertices_ref=rs_set.get("vertices_ref"),
                        lengths=lengths,
                        src=src,
                        name=name
                    )

            elif self.stage in ["mask_transformer", "inference"]:
                # Stage 2 or full inference: requires mask_transformer
                rs_set = self.val_t2m_forward(batch)
                metric_name = self.hparams.metrics_dict[0] if self.hparams.metrics_dict else 'MRMetrics'
                if hasattr(self.metrics, metric_name):
                    getattr(self.metrics, metric_name).update(
                        feats_rst=rs_set["m_rst"],
                        feats_ref=rs_set["m_ref"],
                        joints_rst=rs_set["joints_rst"],
                        joints_ref=rs_set["joints_ref"],
                        vertices_rst=rs_set.get("vertices_rst"),
                        vertices_ref=rs_set.get("vertices_ref"),
                        lengths=lengths,
                        src=src,
                        name=name
                    )

        # Test outputs
        if split == "test":
            return {
                'name': name,
                'feats_ref': rs_set["m_ref"],
                'feats_rst': rs_set.get("m_rst"),
                'lengths': lengths,
                'text': batch.get('text'),
            }

        return loss

    def configure_optimizers(self):
        """Configure optimizers based on training stage."""
        if self.stage == "vae":
            params = self.vae.parameters()
        elif self.stage == "mask_transformer":
            params = self.mask_transformer.parameters_wo_clip()
        elif self.stage == "res_transformer":
            params = self.res_transformer.parameters_wo_clip()
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.cfg.TRAIN.OPTIM.params.lr,
            betas=tuple(self.hparams.cfg.TRAIN.OPTIM.params.betas),
            weight_decay=self.hparams.cfg.TRAIN.OPTIM.params.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.cfg.TRAIN.END_EPOCH,
            eta_min=1e-6
        )

        return [optimizer], [scheduler]