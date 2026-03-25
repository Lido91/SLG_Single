"""
Cross-modal retrieval evaluation for the contrastive model.

Computes Recall@K (K=1,5,10) and Median Rank for enabled retrieval directions.

For triplet (speech+text+motion):
    Speech -> Text,   Text -> Speech
    Speech -> Motion,  Motion -> Speech
    Text -> Motion,    Motion -> Text

For speech+motion only:
    Speech -> Motion,  Motion -> Speech

For text+motion only:
    Text -> Motion,    Motion -> Text

Usage:
    python -m contrastive.evaluate \
        --config contrastive/configs/contrastive_h2s.yaml \
        --checkpoint experiments/contrastive/.../contrastive-epoch=XX.ckpt \
        --split val \
        --batch_size 64 \
        --gpu 0
"""

import argparse
import os
import sys

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contrastive.model_codex import NewContrastiveCodexModel as TripletContrastiveModel
from contrastive.dataset import ContrastiveH2SDataset, contrastive_collate


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    """Encode the entire dataset and return stacked embeddings for enabled modalities."""
    all_speech, all_text, all_motion = [], [], []
    all_names = []

    for batch in tqdm(dataloader, desc='Encoding'):
        if batch is None:
            continue

        motion = batch['motion'].to(device)
        lengths = batch['length']

        motion_emb = model.encode_motion(motion, lengths)
        all_motion.append(motion_emb.cpu())

        if getattr(model, 'enable_speech', True):
            audio = batch['audio'].to(device)
            audio_lengths = batch.get('audio_length', None)
            speech_emb = model.encode_speech(audio, audio_lengths)
            all_speech.append(speech_emb.cpu())

        if getattr(model, 'enable_text', True):
            texts = batch['text']
            text_emb = model.encode_text(texts)
            all_text.append(text_emb.cpu())

        all_names.extend(batch['name'])

    result = {
        'motion': torch.cat(all_motion, dim=0),
        'names': all_names,
    }
    if all_speech:
        result['speech'] = torch.cat(all_speech, dim=0)
    if all_text:
        result['text'] = torch.cat(all_text, dim=0)

    return result


def retrieval_metrics(query_emb, gallery_emb, ks=(1, 5, 10)):
    """
    Compute Recall@K and Median Rank.

    Assumes the i-th query matches the i-th gallery item (paired data).

    Args:
        query_emb: [N, D] L2-normalized embeddings
        gallery_emb: [N, D] L2-normalized embeddings
        ks: tuple of K values for recall

    Returns:
        dict with recall@k and median_rank
    """
    sim = query_emb @ gallery_emb.T  # [N, N]
    N = sim.shape[0]

    # Ranks of the correct match (0-indexed)
    # For each query i, find rank of gallery item i
    sorted_indices = sim.argsort(dim=1, descending=True)  # [N, N]
    gt_indices = torch.arange(N).unsqueeze(1).to(sorted_indices.device)
    ranks = (sorted_indices == gt_indices).nonzero(as_tuple=True)[1]  # [N]

    results = {}
    for k in ks:
        results[f'R@{k}'] = (ranks < k).float().mean().item() * 100.0
    results['MedianR'] = (ranks.float().median().item() + 1)  # 1-indexed

    return results


def _build_direction_names(enable_speech, enable_text):
    """Return list of (query_name, gallery_name) for enabled modality pairs."""
    directions = []
    if enable_speech and enable_text:
        directions += [('Speech', 'Text'), ('Text', 'Speech')]
    if enable_speech:
        directions += [('Speech', 'Motion'), ('Motion', 'Speech')]
    if enable_text:
        directions += [('Text', 'Motion'), ('Motion', 'Text')]
    return directions


def main():
    parser = argparse.ArgumentParser(description='Contrastive Retrieval Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    enable_speech = model_cfg.get('enable_speech', True)
    enable_text = model_cfg.get('enable_text', True)

    # Load mean/std
    mean = torch.load(data_cfg['mean_path'])
    std = torch.load(data_cfg['std_path'])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    # Dataset
    dataset = ContrastiveH2SDataset(
        split=args.split,
        data_root=data_cfg['root'],
        mean=mean,
        std=std,
        max_motion_length=data_cfg['max_motion_length'],
        min_motion_length=data_cfg['min_motion_length'],
        audio_dir=data_cfg.get('audio_dir', data_cfg['root']),
        dataset_name=data_cfg['dataset_name'],
        unit_length=data_cfg.get('unit_length', 4),
        youtube3d_root=data_cfg.get('youtube3d_root', None),
        require_audio=enable_speech,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=contrastive_collate,
        pin_memory=True,
    )

    # Load model — checkpoint was trained with preload_features=True so frozen
    # encoders are NOT in the state_dict. We load with strict=False, then
    # manually attach the frozen encoders needed for live inference.
    vae_cfg = cfg.get('vae', {})
    model = TripletContrastiveModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
        strict=False,
    )
    # Attach frozen encoders that were skipped during precomputed-feature training
    if not hasattr(model, 'motion_encoder'):
        model.motion_encoder = model._load_frozen_vae(vae_cfg, vae_cfg.get('checkpoint'))
    if enable_speech and not hasattr(model, 'speech_encoder'):
        from mGPT.archs.speech_encoder import SpeechEncoder
        model.speech_encoder = SpeechEncoder(
            model.hparams.speech_encoder_type, freeze=True)
    if enable_text and not hasattr(model, 'text_encoder'):
        from mGPT.archs.mgpt_rvq_hierarchical import TextEncoder
        model.text_encoder = TextEncoder(
            model.hparams.text_encoder_type, freeze=True)
    model.preload_features = False
    model = model.to(device)
    model.eval()

    direction_names = _build_direction_names(enable_speech, enable_text)
    modality_label = []
    if enable_speech:
        modality_label.append('Speech')
    if enable_text:
        modality_label.append('Text')
    modality_label.append('Motion')
    print(f'\nModalities: {" + ".join(modality_label)}')

    # ---- Per-batch retrieval ----
    print(f'\n{"="*62}')
    print(f' Per-Batch Retrieval (batch_size={args.batch_size})')
    print(f'{"="*62}')

    batch_metrics = {f'{q} -> {g}': [] for q, g in direction_names}

    for batch_idx, batch in enumerate(tqdm(loader, desc='Per-batch eval')):
        if batch is None:
            continue

        motion = batch['motion'].to(device)
        lengths = batch['length']

        with torch.no_grad():
            emb_map = {}
            emb_map['Motion'] = model.encode_motion(motion, lengths).cpu()

            if enable_speech:
                audio = batch['audio'].to(device)
                audio_lengths = batch.get('audio_length', None)
                emb_map['Speech'] = model.encode_speech(audio, audio_lengths).cpu()

            if enable_text:
                emb_map['Text'] = model.encode_text(batch['text']).cpu()

        bs = emb_map['Motion'].shape[0]
        if bs < 2:
            continue

        print(f'\n--- Batch {batch_idx} ({bs} samples) ---')
        print(f'{"Direction":<22} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MedR":>8}')
        print('-' * 58)

        for q_name, g_name in direction_names:
            direction = f'{q_name} -> {g_name}'
            metrics = retrieval_metrics(emb_map[q_name], emb_map[g_name])
            batch_metrics[direction].append(metrics)
            print(f'{direction:<22} {metrics["R@1"]:>7.2f}% {metrics["R@5"]:>7.2f}% {metrics["R@10"]:>7.2f}% {metrics["MedianR"]:>7.1f}')

    # Per-batch average across all batches
    print(f'\n{"="*62}')
    print(f' Per-Batch Average (across all batches)')
    print(f'{"="*62}')
    print(f'{"Direction":<22} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MedR":>8}')
    print('-' * 58)
    for q_name, g_name in direction_names:
        direction = f'{q_name} -> {g_name}'
        ms = batch_metrics[direction]
        if ms:
            print(f'{direction:<22} '
                  f'{np.mean([m["R@1"] for m in ms]):>7.2f}% '
                  f'{np.mean([m["R@5"] for m in ms]):>7.2f}% '
                  f'{np.mean([m["R@10"] for m in ms]):>7.2f}% '
                  f'{np.mean([m["MedianR"] for m in ms]):>7.1f}')

    # ---- Global retrieval ----
    print(f'\n{"="*62}')
    print(f' Global Retrieval')
    print(f'{"="*62}')

    embs = extract_all_embeddings(model, loader, device)
    motion_emb = embs['motion']
    N = motion_emb.shape[0]
    print(f'Evaluating on {N} samples from {args.split} split\n')

    pairs = []
    if enable_speech and enable_text:
        pairs += [
            ('Speech', 'Text', embs['speech'], embs['text']),
            ('Text', 'Speech', embs['text'], embs['speech']),
        ]
    if enable_speech:
        pairs += [
            ('Speech', 'Motion', embs['speech'], motion_emb),
            ('Motion', 'Speech', motion_emb, embs['speech']),
        ]
    if enable_text:
        pairs += [
            ('Text', 'Motion', embs['text'], motion_emb),
            ('Motion', 'Text', motion_emb, embs['text']),
        ]

    print(f'{"Direction":<22} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MedR":>8}')
    print('-' * 58)

    all_results = {}
    for q_name, g_name, q_emb, g_emb in pairs:
        direction = f'{q_name} -> {g_name}'
        metrics = retrieval_metrics(q_emb, g_emb)
        all_results[direction] = metrics
        print(f'{direction:<22} {metrics["R@1"]:>7.2f}% {metrics["R@5"]:>7.2f}% {metrics["R@10"]:>7.2f}% {metrics["MedianR"]:>7.1f}')

    # Summary averages
    avg_r1 = np.mean([m['R@1'] for m in all_results.values()])
    avg_r5 = np.mean([m['R@5'] for m in all_results.values()])
    avg_r10 = np.mean([m['R@10'] for m in all_results.values()])
    avg_medr = np.mean([m['MedianR'] for m in all_results.values()])
    print('-' * 58)
    print(f'{"Average":<22} {avg_r1:>7.2f}% {avg_r5:>7.2f}% {avg_r10:>7.2f}% {avg_medr:>7.1f}')


if __name__ == '__main__':
    main()
