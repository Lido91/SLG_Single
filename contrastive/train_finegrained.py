"""
Training script for Fine-Grained Contrastive Model (paper 2507.23188).

Usage:
    python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml
    python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml --use_gpus 0,1
"""

import argparse
import os
import sys

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contrastive.model_finegrained import FineGrainedContrastiveModel
from contrastive.dataset import (
    ContrastiveH2SDataset, contrastive_collate,
    PrecomputedContrastiveDataset, precomputed_collate,
)


class ProjectionHeadSaver(pl.Callback):
    """Save model components separately when val/avg_R@1 improves."""

    def __init__(self, dirpath='experiments/contrastive_fg/checkpoints'):
        super().__init__()
        self.dirpath = dirpath
        self.best_r1 = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_r1 = trainer.callback_metrics.get('val/avg_R@1')
        if avg_r1 is None:
            return

        avg_r1 = avg_r1.item()
        if avg_r1 > self.best_r1:
            self.best_r1 = avg_r1

            # Only save on rank 0 to avoid DDP file corruption
            if trainer.is_global_zero:
                os.makedirs(self.dirpath, exist_ok=True)

                torch.save(pl_module.motion_encoder.state_dict(),
                           os.path.join(self.dirpath, 'motion_encoder_best.pt'))
                torch.save(pl_module.audio_encoder.state_dict(),
                           os.path.join(self.dirpath, 'audio_encoder_best.pt'))
                torch.save(pl_module.text_encoder.state_dict(),
                           os.path.join(self.dirpath, 'text_encoder_best.pt'))
                torch.save(pl_module.weight_heads.state_dict(),
                           os.path.join(self.dirpath, 'weight_heads_best.pt'))
                torch.save(pl_module.recon_decoder.state_dict(),
                           os.path.join(self.dirpath, 'recon_decoder_best.pt'))
                torch.save({
                    'logit_scale': pl_module.logit_scale.data,
                    'mask_token': pl_module.mask_token.data,
                }, os.path.join(self.dirpath, 'params_best.pt'))

                print(f"\n[FG-Saver] Saved best components (val/avg_R@1={avg_r1:.2f}%)")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Contrastive Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_gpus', type=str, default=None, help='Comma-separated GPU ids')
    parser.add_argument('--nodebug', action='store_true', help='Skip sanity check')
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    logger_cfg = cfg.get('logger', {})

    # Override devices from CLI
    if args.use_gpus is not None:
        train_cfg['devices'] = [int(g) for g in args.use_gpus.split(',')]

    # Load mean/std
    mean = torch.load(data_cfg['mean_path'])
    std = torch.load(data_cfg['std_path'])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    # Datasets
    preload_features = data_cfg.get('preload_features', False)

    common_data_kwargs = dict(
        data_root=data_cfg['root'],
        mean=mean,
        std=std,
        max_motion_length=data_cfg['max_motion_length'],
        min_motion_length=data_cfg['min_motion_length'],
        audio_dir=data_cfg['audio_dir'],
        dataset_name=data_cfg['dataset_name'],
        unit_length=data_cfg.get('unit_length', 4),
        youtube3d_root=data_cfg.get('youtube3d_root', None),
    )

    if preload_features:
        feat_dir = data_cfg.get('precomputed_feat_dir',
                                os.path.join(data_cfg['root'], 'precomputed_feats'))
        precomputed_kwargs = dict(
            feat_root=feat_dir,
            speech_type=model_cfg.get('speech_encoder_type', 'wavlm-large'),
            text_type=model_cfg.get('text_encoder_type', 'distilbert'),
        )
        train_dataset = PrecomputedContrastiveDataset(
            split='train', **precomputed_kwargs, **common_data_kwargs)
        val_dataset = PrecomputedContrastiveDataset(
            split='val', **precomputed_kwargs, **common_data_kwargs)
        collate_fn = precomputed_collate
    else:
        train_dataset = ContrastiveH2SDataset(split='train', **common_data_kwargs)
        val_dataset = ContrastiveH2SDataset(split='val', **common_data_kwargs)
        collate_fn = contrastive_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 16),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 16),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    vae_cfg = cfg['vae']
    vae_config = {'params': vae_cfg['params']}
    motion_cfg = model_cfg.get('motion', {})
    audio_cfg = model_cfg.get('audio', {})
    text_cfg = model_cfg.get('text', {})
    loss_cfg = model_cfg.get('loss', {})
    recon_cfg = model_cfg.get('reconstruction', {})

    model = FineGrainedContrastiveModel(
        speech_encoder_type=model_cfg.get('speech_encoder_type', 'wavlm-large'),
        text_encoder_type=model_cfg.get('text_encoder_type', 'distilbert'),
        latent_dim=model_cfg.get('latent_dim', 512),
        # Motion (VAE-based)
        vae_config=vae_config,
        vae_checkpoint=vae_cfg.get('checkpoint', None),
        motion_num_layers=motion_cfg.get('num_layers', 4),
        motion_nhead=motion_cfg.get('nhead', 8),
        # Audio
        audio_num_memory=audio_cfg.get('num_memory_tokens', 128),
        audio_num_attn_layers=audio_cfg.get('num_attn_layers', 2),
        audio_target_length=audio_cfg.get('target_length', 32),
        # Text
        text_transformer_layers=text_cfg.get('transformer_layers', 2),
        text_max_len=text_cfg.get('max_text_len', 77),
        # Loss
        temperature_init=loss_cfg.get('temperature_init', 0.07),
        lambda_recon=loss_cfg.get('lambda_recon', 0.1),
        mask_ratio=loss_cfg.get('mask_ratio', 0.5),
        # Reconstruction
        recon_num_layers=recon_cfg.get('num_layers', 2),
        recon_nhead=recon_cfg.get('nhead', 8),
        # Optimizer
        lr=train_cfg.get('lr', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        max_epochs=train_cfg.get('max_epochs', 200),
        # Precomputed features
        preload_features=preload_features,
    )

    # Logger
    save_dir = 'experiments/contrastive_fg'
    logger = WandbLogger(
        project=logger_cfg.get('project', 'SLG'),
        name=logger_cfg.get('name', 'FineGrained_Contrastive'),
        save_dir=save_dir,
    )

    # Callbacks
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='fg-contrastive-{epoch:02d}',
        auto_insert_metric_name=False,
        monitor='val/avg_R@1',
        mode='max',
        save_top_k=3,
        save_last=True,
        every_n_epochs=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    proj_saver = ProjectionHeadSaver(dirpath=ckpt_dir)

    # Trainer
    devices = train_cfg.get('devices', [0])
    trainer_kwargs = dict(
        max_epochs=train_cfg.get('max_epochs', 200),
        accelerator='gpu',
        devices=devices,
        strategy='ddp' if len(devices) > 1 else 'auto',
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, proj_saver],
        precision='bf16-mixed',
        gradient_clip_val=1.0,
    )
    if args.nodebug:
        trainer_kwargs['num_sanity_val_steps'] = 0
        trainer_kwargs['enable_model_summary'] = False
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
