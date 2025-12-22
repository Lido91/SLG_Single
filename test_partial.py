"""
Test RVQ-VAE reconstruction with partial quantizers (3 of 6).

This script loads pre-computed 6-quantizer tokens and tests reconstruction
using only the first 3 quantizers (Q0, Q1, Q2), simulating the setup used
by HierarchicalRVQGPT.

Similar to mgpt_rvq_hierarchical.py:181-182 which uses [:,:,:3]

Usage:
    python test_partial.py --cfg configs/deto_h2s_rvq.yaml
"""

import json
import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm
from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

torch.set_float32_matmul_precision('high')


def print_table(title, metrics, logger=None):
    table = Table(title=title)
    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")
    logger.info(metrics) if logger else None


def main():
    # Parse options
    cfg = parse_args(phase="test")

    # Number of quantizers to use (default 3 for hierarchical compatibility)
    num_quantizers = cfg.get("TEST_NUM_QUANTIZERS", 3)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['USE_TORCH'] = '1'

    cfg.FOLDER = cfg.TEST.FOLDER

    # Logger
    logger = create_logger(cfg, phase="test")
    logger.info(f"Testing RVQ-VAE with {num_quantizers} quantizers (partial decoding)")
    logger.info(f"Loading tokens from: {cfg.DATASET.CODE_PATH}")

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # Dataset - use token dataset to load pre-computed codes
    cfg.TRAIN.STAGE = "token"  # Use token dataset
    datamodule = build_data(cfg)
    datamodule.setup()
    logger.info("Token dataset loaded")

    # Build VAE only
    vae = instantiate_from_config(cfg.model.params.motion_vae)
    vae.eval()

    # Load VAE checkpoint
    if cfg.TRAIN.PRETRAINED_VAE:
        ckpt_path = cfg.TRAIN.PRETRAINED_VAE
    else:
        ckpt_path = cfg.TEST.CHECKPOINTS

    logger.info(f"Loading VAE from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract VAE state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        vae_state = {k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.')}
    else:
        vae_state = ckpt

    vae.load_state_dict(vae_state)
    vae = vae.cuda()
    logger.info("VAE loaded successfully")

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    # Metrics storage
    all_mse = {i: [] for i in range(1, 7)}  # MSE for 1-6 quantizers
    all_l1 = {i: [] for i in range(1, 7)}   # L1 for 1-6 quantizers

    # Test loop
    logger.info(f"Testing on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Get tokens: (B, T, 6)
            tokens = batch["motion"].cuda().long()  # Pre-computed 6-quantizer tokens (must be long for embedding)

            # Get original motion for reference (if available)
            # For token dataset, we need to decode full tokens as reference
            tokens_full = tokens  # (B, T, 6)

            # Decode with full 6 quantizers as reference
            motion_full = vae.decode_partial(tokens_full)  # (B, T*4, 133)

            # Test with different numbers of quantizers
            for n_q in range(1, 7):
                # Take first n_q quantizers: [:, :, :n_q]
                tokens_partial = tokens[:, :, :n_q]  # (B, T, n_q)

                # Decode with partial quantizers
                motion_partial = vae.decode_partial(tokens_partial)  # (B, T*4, 133)

                # Compute reconstruction error vs full reconstruction
                min_len = min(motion_full.shape[1], motion_partial.shape[1])
                mse = ((motion_full[:, :min_len] - motion_partial[:, :min_len]) ** 2).mean().item()
                l1 = (motion_full[:, :min_len] - motion_partial[:, :min_len]).abs().mean().item()

                all_mse[n_q].append(mse)
                all_l1[n_q].append(l1)

    # Compute and print results
    results = {}
    logger.info("\n" + "="*60)
    logger.info("Results: Reconstruction Error vs Full 6-Quantizer Decoding")
    logger.info("="*60)

    for n_q in range(1, 7):
        mean_mse = np.mean(all_mse[n_q])
        mean_l1 = np.mean(all_l1[n_q])
        results[f"Q0-Q{n_q-1}_MSE"] = f"{mean_mse:.6f}"
        results[f"Q0-Q{n_q-1}_L1"] = f"{mean_l1:.6f}"
        logger.info(f"Q0-Q{n_q-1} ({n_q} quantizers): MSE={mean_mse:.6f}, L1={mean_l1:.6f}")

    logger.info("="*60)

    # Highlight the 3-quantizer result (used by HierarchicalRVQGPT)
    logger.info(f"\n>>> HierarchicalRVQGPT uses 3 quantizers (Q0-Q2):")
    logger.info(f"    MSE = {np.mean(all_mse[3]):.6f}")
    logger.info(f"    L1  = {np.mean(all_l1[3]):.6f}")

    print_table("Partial RVQ Decoding Results", results, logger=logger)

    logger.info(f"\nFinished testing")
    logger.info(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
