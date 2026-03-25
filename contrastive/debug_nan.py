"""
Debug script to locate NaN source in FineGrained Contrastive Model.
Runs actual training steps (forward + backward + optimizer) with NaN checks.

Usage:
    python -m contrastive.debug_nan --config contrastive/configs/contrastive_finegrained.yaml --gpu 3
    python -m contrastive.debug_nan --config contrastive/configs/contrastive_finegrained.yaml --gpu 3 --no_amp
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contrastive.model_finegrained import FineGrainedContrastiveModel
from contrastive.dataset import ContrastiveH2SDataset, contrastive_collate
from torch.utils.data import DataLoader


def check(name, t):
    """Check tensor for NaN/Inf. Returns True if bad."""
    if t is None or not isinstance(t, torch.Tensor):
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    tag = ""
    if has_nan:
        tag += " *** NaN ***"
    if has_inf:
        tag += " *** Inf ***"
    if tag:
        print(f"    {name}: shape={list(t.shape)} dtype={t.dtype} "
              f"min={t.min().item():.4f} max={t.max().item():.4f}{tag}")
    return has_nan or has_inf


def check_grads(model, step):
    """Check all trainable parameter gradients for NaN/Inf."""
    bad_params = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                bad_params.append(name)
                gmin = p.grad.min().item()
                gmax = p.grad.max().item()
                print(f"    GRAD NaN/Inf: {name} grad_min={gmin:.4f} grad_max={gmax:.4f}")
    return bad_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Match training batch size')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP to test FP32')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    use_amp = not args.no_amp
    print(f"AMP: {'ON (FP16)' if use_amp else 'OFF (FP32)'}")
    print(f"Batch size: {args.batch_size}, Steps: {args.steps}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    vae_cfg = cfg['vae']

    mean = torch.load(data_cfg['mean_path'])
    std = torch.load(data_cfg['std_path'])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    dataset = ContrastiveH2SDataset(
        split='train',
        data_root=data_cfg['root'],
        mean=mean, std=std,
        max_motion_length=data_cfg['max_motion_length'],
        min_motion_length=data_cfg['min_motion_length'],
        audio_dir=data_cfg['audio_dir'],
        dataset_name=data_cfg['dataset_name'],
        unit_length=data_cfg.get('unit_length', 4),
        youtube3d_root=data_cfg.get('youtube3d_root', None),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=contrastive_collate, num_workers=4, drop_last=True)

    # Build model
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
        vae_config=vae_config,
        vae_checkpoint=vae_cfg.get('checkpoint', None),
        motion_num_layers=motion_cfg.get('num_layers', 4),
        motion_nhead=motion_cfg.get('nhead', 8),
        audio_num_memory=audio_cfg.get('num_memory_tokens', 128),
        audio_num_attn_layers=audio_cfg.get('num_attn_layers', 2),
        audio_target_length=audio_cfg.get('target_length', 32),
        text_transformer_layers=text_cfg.get('transformer_layers', 2),
        text_max_len=text_cfg.get('max_text_len', 77),
        temperature_init=loss_cfg.get('temperature_init', 0.07),
        lambda_recon=loss_cfg.get('lambda_recon', 0.1),
        mask_ratio=loss_cfg.get('mask_ratio', 0.5),
        recon_num_layers=recon_cfg.get('num_layers', 2),
    ).to(device)
    model.train()

    # Same optimizer as training
    optimizer = model.configure_optimizers()[0][0]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nRunning {args.steps} training steps...\n")

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break

        texts = batch['text']
        motion = batch['motion'].to(device)
        audio = batch['audio'].to(device)
        audio_lengths = batch.get('audio_length', None)
        lengths = batch['length']

        optimizer.zero_grad()

        # Forward
        with torch.cuda.amp.autocast(enabled=use_amp):
            motion_tokens, motion_mask = model.encode_motion(motion, lengths)
            speech_tokens, speech_mask = model.encode_speech(audio, audio_lengths)
            text_tokens, text_mask = model.encode_text(texts)

            # Check encoder outputs
            enc_bad = False
            enc_bad |= check("motion_tokens", motion_tokens)
            enc_bad |= check("speech_tokens", speech_tokens)
            enc_bad |= check("text_tokens", text_tokens)

            temperature = model.logit_scale.exp().clamp(min=1.0, max=100.0)

            loss_st, loss_sm, loss_tm = model._compute_all_alignment_losses(
                speech_tokens, speech_mask, text_tokens, text_mask,
                motion_tokens, motion_mask,
            )
            loss_align = (loss_st + loss_sm + loss_tm) / 3.0

            loss_recon = model._compute_reconstruction_loss(
                motion_tokens, motion_mask, text_tokens, text_mask,
                speech_tokens, speech_mask,
            )

            total = loss_align + model.lambda_recon * loss_recon

        # Check losses
        loss_bad = False
        loss_bad |= check("loss_st", loss_st.detach().unsqueeze(0))
        loss_bad |= check("loss_sm", loss_sm.detach().unsqueeze(0))
        loss_bad |= check("loss_tm", loss_tm.detach().unsqueeze(0))
        loss_bad |= check("loss_recon", loss_recon.detach().unsqueeze(0))
        loss_bad |= check("total_loss", total.detach().unsqueeze(0))

        # Backward
        scaler.scale(total).backward()

        # Check gradients BEFORE unscale (still scaled)
        scaler.unscale_(optimizer)
        grad_bad = check_grads(model, step)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Check params after update
        param_bad = False
        for name, p in model.named_parameters():
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                param_bad = True
                print(f"    PARAM NaN/Inf after step: {name}")

        # Summary
        scale = scaler.get_scale()
        status = "OK"
        if enc_bad:
            status = "ENCODER NaN"
        elif loss_bad:
            status = "LOSS NaN"
        elif grad_bad:
            status = f"GRAD NaN ({len(grad_bad)} params)"
        elif param_bad:
            status = "PARAM NaN"

        print(f"  Step {step:3d}: loss={total.item():.4f} "
              f"(st={loss_st.item():.4f} sm={loss_sm.item():.4f} "
              f"tm={loss_tm.item():.4f} recon={loss_recon.item():.4f}) "
              f"T={temperature.item():.2f} scale={scale:.0f} [{status}]")

        if enc_bad or loss_bad or param_bad:
            print(f"\n  >>> First NaN at step {step}. Stopping.")
            break

    print("\nDone.")


if __name__ == '__main__':
    main()
