"""
Zero-shot Speech <-> Text retrieval evaluation using CLAP.

Uses the pretrained CLAP model (laion/larger_clap_music_and_speech) as a
zero-shot baseline for Speech <-> Text retrieval on YouTube3D / How2Sign.

Computes Recall@K (K=1,5,10) and Median Rank for:
    Speech -> Text,   Text -> Speech

Usage:
    python -m contrastive.evaluate_clap \
        --config contrastive/configs/contrastive_h2s.yaml \
        --split test \
        --batch_size 64 \
        --gpu 2
"""

import argparse
import os
import sys

import torch
import yaml
import numpy as np
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contrastive.dataset import ContrastiveH2SDataset


# ---------- Lightweight dataset: only audio + text ----------

class AudioTextDataset(Dataset):
    """Wraps ContrastiveH2SDataset to return only audio waveforms and text."""

    def __init__(self, base_dataset, clap_sr=48000):
        self.base = base_dataset
        self.clap_sr = clap_sr

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base.all_data[idx]
        name = sample['name']
        text = sample['text']

        audio_path = self.base._get_audio_path(sample)
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.clap_sr:
                waveform = torchaudio.transforms.Resample(sr, self.clap_sr)(waveform)
            # Truncate to 30s
            max_samples = self.clap_sr * 30
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            waveform = waveform.squeeze(0)  # (num_samples,)
        except Exception as e:
            print(f"[CLAP eval] failed to load audio {audio_path}: {e}")
            waveform = torch.zeros(self.clap_sr * 3)

        return {'audio': waveform, 'text': text, 'name': name}


def audio_text_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'audio': [b['audio'].numpy() for b in batch],  # list of np arrays
        'text': [b['text'] for b in batch],
        'name': [b['name'] for b in batch],
    }


# ---------- Retrieval metrics (same as evaluate.py) ----------

def retrieval_metrics(query_emb, gallery_emb, ks=(1, 5, 10)):
    sim = query_emb @ gallery_emb.T
    N = sim.shape[0]
    sorted_indices = sim.argsort(dim=1, descending=True)
    gt_indices = torch.arange(N).unsqueeze(1).to(sorted_indices.device)
    ranks = (sorted_indices == gt_indices).nonzero(as_tuple=True)[1]
    results = {}
    for k in ks:
        results[f'R@{k}'] = (ranks < k).float().mean().item() * 100.0
    results['MedianR'] = ranks.float().median().item() + 1
    return results


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='CLAP Zero-Shot Speech<->Text Retrieval')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--clap_model', type=str, default='laion/larger_clap_music_and_speech')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg = cfg['data']
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load mean/std (needed by ContrastiveH2SDataset init but unused for CLAP)
    mean = torch.load(data_cfg['mean_path'])
    std = torch.load(data_cfg['std_path'])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    # Build base dataset for sample list + audio paths
    base_dataset = ContrastiveH2SDataset(
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
        require_audio=True,
    )

    dataset = AudioTextDataset(base_dataset, clap_sr=48000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=audio_text_collate,
        pin_memory=True,
    )

    # Load CLAP
    print(f'\nLoading CLAP model: {args.clap_model}')
    processor = ClapProcessor.from_pretrained(args.clap_model)
    clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
    clap_model.eval()

    # Extract all embeddings
    all_speech_emb = []
    all_text_emb = []
    all_names = []

    print(f'\nEncoding {len(dataset)} samples...')
    for batch in tqdm(loader, desc='CLAP encoding'):
        if batch is None:
            continue

        # Audio embeddings
        audio_inputs = processor(
            audios=batch['audio'],
            return_tensors='pt',
            sampling_rate=48000,
            padding=True,
        )
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()
                        if isinstance(v, torch.Tensor)}

        # Text embeddings
        text_inputs = processor(
            text=batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()
                       if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            speech_emb = clap_model.get_audio_features(**audio_inputs)
            text_emb = clap_model.get_text_features(**text_inputs)

        # L2 normalize
        speech_emb = speech_emb / speech_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        all_speech_emb.append(speech_emb.cpu())
        all_text_emb.append(text_emb.cpu())
        all_names.extend(batch['name'])

    speech_emb = torch.cat(all_speech_emb, dim=0)
    text_emb = torch.cat(all_text_emb, dim=0)
    N = speech_emb.shape[0]

    # Global retrieval
    print(f'\n{"="*62}')
    print(f' CLAP Zero-Shot Retrieval ({N} samples, {args.split} split)')
    print(f'{"="*62}')
    print(f'{"Direction":<22} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MedR":>8}')
    print('-' * 58)

    pairs = [
        ('Speech', 'Text', speech_emb, text_emb),
        ('Text', 'Speech', text_emb, speech_emb),
    ]

    all_results = {}
    for q_name, g_name, q_emb, g_emb in pairs:
        direction = f'{q_name} -> {g_name}'
        metrics = retrieval_metrics(q_emb, g_emb)
        all_results[direction] = metrics
        print(f'{direction:<22} {metrics["R@1"]:>7.2f}% {metrics["R@5"]:>7.2f}% {metrics["R@10"]:>7.2f}% {metrics["MedianR"]:>7.1f}')

    avg_r1 = np.mean([m['R@1'] for m in all_results.values()])
    avg_r5 = np.mean([m['R@5'] for m in all_results.values()])
    avg_r10 = np.mean([m['R@10'] for m in all_results.values()])
    avg_medr = np.mean([m['MedianR'] for m in all_results.values()])
    print('-' * 58)
    print(f'{"Average":<22} {avg_r1:>7.2f}% {avg_r5:>7.2f}% {avg_r10:>7.2f}% {avg_medr:>7.1f}')


if __name__ == '__main__':
    main()
