"""
Training script for Text-to-Motion Transformer (TM Transformer)
UniMuMo-style architecture adapted for MotionGPT

Usage:
    python train_tm_transformer.py --config configs/tm_h2s.yaml

This script:
1. Loads pre-computed motion tokens from get_motion_code.py for TRAINING
2. Uses VQ-VAE for on-the-fly validation (like mgpt.py)
3. Trains the TM Transformer with per-codebook embeddings
4. Uses T5 cross-attention conditioning
5. Supports CFG (Classifier-Free Guidance) training
"""

import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained_vae
from mGPT.models.tm_lightning import TextToMotionLightning, tm_collate_fn, tm_val_collate_fn

torch.set_float32_matmul_precision('high')


def main():
    # Parse config
    cfg = parse_args(phase="train")

    # Override stage to use LM dataset (Text2MotionDatasetCB for train, Text2MotionDatasetEval for val)
    cfg.TRAIN.STAGE = "lm_pretrain"

    # Logger
    logger = create_logger(cfg, phase="train")
    logger.info(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tensorboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset (H2SDataModule)
    # - Training: Text2MotionDatasetCB (loads pre-computed tokens)
    # - Validation: Text2MotionDatasetEval (loads raw motion features)
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Override collate_fn for TRAINING (pre-computed tokens)
    datamodule.dataloader_options["collate_fn"] = tm_collate_fn
    logger.info("Training collate function set to tm_collate_fn")

    # Get model config from cfg or use defaults
    model_cfg = cfg.get('TM_MODEL', {})

    # Get VAE config for validation (on-the-fly decoding)
    # Use vq.h2s_rvq (RVQVae with 6 codebooks) to match the pre-computed tokens
    vae_cfg = None
    if cfg.TRAIN.get('PRETRAINED_VAE', None):
        # Try h2s_rvq first (6 codebooks), fall back to default
        vae_cfg = cfg.get('vq', {}).get('h2s_rvq', None)
        if vae_cfg is None:
            vae_cfg = cfg.get('vq', {}).get('default', None)
        if vae_cfg:
            logger.info(f"VAE config found ({vae_cfg.get('target', 'unknown')}), will be used for validation")
        else:
            logger.info("No VAE config found, validation will be skipped")

    # Build TM Transformer model
    logger.info("Building TM Transformer model...")
    model = TextToMotionLightning(
        # Codebook settings (from VQ-VAE)
        n_q=model_cfg.get('N_Q', 6),
        card=model_cfg.get('CARD', 512),

        # Transformer settings
        dim=model_cfg.get('DIM', 768),
        num_heads=model_cfg.get('NUM_HEADS', 12),
        num_layers=model_cfg.get('NUM_LAYERS', 12),
        hidden_scale=model_cfg.get('HIDDEN_SCALE', 4),
        dropout=model_cfg.get('DROPOUT', 0.1),
        layer_scale=model_cfg.get('LAYER_SCALE', None),

        # T5 Conditioner
        t5_name=model_cfg.get('T5_NAME', "google/flan-t5-base"),
        t5_finetune=model_cfg.get('T5_FINETUNE', False),

        # CFG
        cfg_dropout=model_cfg.get('CFG_DROPOUT', 0.1),
        cfg_coef=model_cfg.get('CFG_COEF', 3.0),

        # Pattern
        pattern_type=model_cfg.get('PATTERN_TYPE', 'delayed'),
        delays=model_cfg.get('DELAYS', None),

        # Training
        lr=model_cfg.get('LR', 1e-4),
        weight_decay=model_cfg.get('WEIGHT_DECAY', 0.01),
        warmup_steps=model_cfg.get('WARMUP_STEPS', 2000),
        max_steps=model_cfg.get('MAX_STEPS', 200000),
        emb_lr=model_cfg.get('EMB_LR', None),

        # Generation
        gen_temp=model_cfg.get('GEN_TEMP', 1.0),
        gen_top_k=model_cfg.get('GEN_TOP_K', 250),
        gen_top_p=model_cfg.get('GEN_TOP_P', 0.0),
        gen_max_len=model_cfg.get('GEN_MAX_LEN', 256),

        # VAE config for validation
        vae_cfg=vae_cfg,
    )
    logger.info("TM Transformer model built")

    # Load pretrained VAE weights for validation
    if cfg.TRAIN.get('PRETRAINED_VAE', None) and model.vae is not None:
        load_pretrained_vae(cfg, model, logger)
        logger.info("Pretrained VAE loaded for validation")

    # Determine if validation is enabled
    has_val = model.vae is not None
    if has_val:
        logger.info("Validation ENABLED with on-the-fly VAE decoding")
    else:
        logger.info("Validation DISABLED (no VAE configured)")

    # Custom validation dataloader with different collate function
    # We need to override the val_dataloader to use tm_val_collate_fn
    if has_val:
        # Store original val_dataloader method
        original_val_dataloader = datamodule.val_dataloader

        def custom_val_dataloader():
            # Get the original dataloader
            dl = original_val_dataloader()
            # Create new dataloader with tm_val_collate_fn
            return DataLoader(
                dl.dataset,
                batch_size=cfg.EVAL.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.EVAL.NUM_WORKERS,
                collate_fn=tm_val_collate_fn,
                persistent_workers=True,
            )

        datamodule.val_dataloader = custom_val_dataloader
        logger.info("Validation collate function set to tm_val_collate_fn")

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS if has_val else None,
        limit_val_batches=1.0 if has_val else 0,  # Enable/disable validation
        num_sanity_val_steps=2,  # Run 2 validation batches as sanity check before training
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )
    logger.info("Trainer initialized")

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.RESUME)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
