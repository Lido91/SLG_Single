"""
Pre-extract frozen encoder features for contrastive training.

Saves per-sample .pt files for speech, text, and motion encoders,
so training only needs to load features + run projection heads.

Automatically adapts to different configs:
  - contrastive_h2s.yaml:         hubert-large + clip + RVQVae
  - contrastive_how2sign.yaml:    hubert-large + clip + RVQVae
  - contrastive_finegrained.yaml: wavlm-large + distilbert + RVQVae

Usage:
    python -m contrastive.extract_features \
        --config contrastive/configs/contrastive_h2s.yaml --gpu 0

    python -m contrastive.extract_features \
        --config contrastive/configs/contrastive_finegrained.yaml --gpu 0

    python -m contrastive.extract_features \
        --config contrastive/configs/contrastive_h2s.yaml --gpu 0 \
        --save_dir /data/hwu/slg_data/Youtube3D/precomputed_feats

    python -m contrastive.extract_features \
        --config contrastive/configs/contrastive_h2s.yaml --gpu 0 \
        --modalities speech  # only extract speech

Output structure under save_dir:
    precomputed_feats/
    ├── {speech_type}/          e.g. hubert-large/ or wavlm-large/
    │   ├── train/{name}.pt    → {'feats': [T_s, dim], 'length': int}
    │   └── val/{name}.pt
    ├── {text_type}/            e.g. clip/ or distilbert/
    │   ├── train/{name}.pt    → {'feats': [seq, dim], 'mask': [seq] (optional)}
    │   └── val/{name}.pt
    └── motion_rvq/
        ├── train/{name}.pt    → {'feats': [T', 512], 'length': int}
        └── val/{name}.pt
"""

import argparse
import math
import os
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contrastive.dataset import ContrastiveH2SDataset
from mGPT.archs.speech_encoder import SpeechEncoder
from mGPT.archs.mgpt_rvq import RVQVae


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_frozen_vae(vae_cfg):
    params = vae_cfg['params']
    vae = RVQVae(**params)
    ckpt_path = vae_cfg.get('checkpoint', None)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            vae_state = {}
            for k, v in state_dict.items():
                if k.startswith('vae.'):
                    vae_state[k[len('vae.'):]] = v
            if vae_state:
                state_dict = vae_state
        else:
            state_dict = ckpt
        vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_text_encoder(text_type):
    """Load frozen text encoder based on type. Returns (encoder, forward_fn)."""
    if text_type == 'clip':
        from mGPT.archs.mgpt_rvq_hierarchical import TextEncoder
        encoder = TextEncoder('clip', freeze=True)
        return encoder
    elif text_type == 'distilbert':
        from transformers import DistilBertModel, DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        # Wrap in a simple namespace so we can treat it uniformly
        class DistilBertEncoder:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            def to(self, device):
                self.model = self.model.to(device)
                return self
            def __call__(self, texts):
                device = next(self.model.parameters()).device
                encoded = self.tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=77, return_tensors='pt',
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    feats = outputs.last_hidden_state  # [B, seq, 768]
                return feats, attention_mask.float()
        return DistilBertEncoder(model, tokenizer)
    elif text_type in ('bert', 'bert-large'):
        from mGPT.archs.mgpt_rvq_hierarchical import TextEncoder
        encoder = TextEncoder(text_type, freeze=True)
        return encoder
    else:
        raise ValueError(f"Unknown text encoder type: {text_type}")


@torch.no_grad()
def extract_speech_features(speech_encoder, dataset, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    speech_encoder.to(device)

    skipped, existing = 0, 0
    for idx in tqdm(range(len(dataset)), desc='Speech'):
        sample = dataset[idx]
        if sample is None:
            skipped += 1
            continue

        name = sample['name']
        out_path = os.path.join(save_dir, f'{name}.pt')
        if os.path.exists(out_path):
            existing += 1
            continue

        audio = sample['audio'].unsqueeze(0).to(device)  # [1, num_samples]
        audio_length = audio.shape[1]

        feats, mask = speech_encoder(audio)  # [1, T_s, dim], [1, T_s]

        # Recompute valid length (HuBERT/WavLM/Wav2Vec2 stride=320)
        T_s = feats.shape[1]
        valid_frames = min(audio_length // 320, T_s)

        torch.save({
            'feats': feats[0].cpu(),
            'length': valid_frames,
        }, out_path)

    print(f'Speech: done ({existing} existed, {skipped} skipped)')


@torch.no_grad()
def extract_text_features(text_encoder, dataset, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    text_encoder.to(device)

    skipped, existing = 0, 0
    for idx in tqdm(range(len(dataset)), desc='Text'):
        sample = dataset[idx]
        if sample is None:
            skipped += 1
            continue

        name = sample['name']
        out_path = os.path.join(save_dir, f'{name}.pt')
        if os.path.exists(out_path):
            existing += 1
            continue

        feats, mask = text_encoder([sample['text']])  # [1, seq, dim]

        save_dict = {'feats': feats[0].cpu()}  # [seq, dim]
        if mask is not None:
            save_dict['mask'] = mask[0].cpu()
        torch.save(save_dict, out_path)

    print(f'Text: done ({existing} existed, {skipped} skipped)')


@torch.no_grad()
def extract_motion_features(vae, dataset, save_dir, device, vae_cfg):
    os.makedirs(save_dir, exist_ok=True)
    vae.to(device)

    stride_t = vae_cfg['params'].get('stride_t', 2)
    down_t = vae_cfg['params'].get('down_t', 2)
    temporal_compression = stride_t ** down_t

    skipped, existing = 0, 0
    for idx in tqdm(range(len(dataset)), desc='Motion'):
        sample = dataset[idx]
        if sample is None:
            skipped += 1
            continue

        name = sample['name']
        out_path = os.path.join(save_dir, f'{name}.pt')
        if os.path.exists(out_path):
            existing += 1
            continue

        motion = sample['motion'].unsqueeze(0).to(device)  # [1, T, 133]
        m_length = sample['length']

        x = vae.preprocess(motion)   # [1, 133, T]
        x = vae.encoder(x)           # [1, 512, T']
        x = x.permute(0, 2, 1)       # [1, T', 512]

        T_prime = x.shape[1]
        valid = min(math.ceil(m_length / temporal_compression), T_prime)

        torch.save({
            'feats': x[0].cpu(),
            'length': valid,
        }, out_path)

    print(f'Motion: done ({existing} existed, {skipped} skipped)')


def main():
    parser = argparse.ArgumentParser(description='Pre-extract frozen encoder features')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated splits (default: train,val)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override save directory (default: {data.root}/precomputed_feats)')
    parser.add_argument('--modalities', type=str, default='speech,text,motion',
                        help='Comma-separated modalities to extract (default: speech,text,motion)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    vae_cfg = cfg['vae']

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    save_root = args.save_dir or os.path.join(data_cfg['root'], 'precomputed_feats')
    modalities = args.modalities.split(',')

    speech_type = model_cfg['speech_encoder_type']
    text_type = model_cfg['text_encoder_type']

    # Print summary
    print(f'{"="*60}')
    print(f'Config:     {args.config}')
    print(f'Dataset:    {data_cfg["dataset_name"]}')
    print(f'Data root:  {data_cfg["root"]}')
    print(f'Speech:     {speech_type}')
    print(f'Text:       {text_type}')
    print(f'Save to:    {save_root}')
    print(f'Modalities: {modalities}')
    print(f'Splits:     {args.splits}')
    print(f'Device:     {device}')
    print(f'{"="*60}')

    # Load mean/std
    mean = torch.load(data_cfg['mean_path'])
    std = torch.load(data_cfg['std_path'])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    # Load encoders (only the ones we need)
    speech_encoder = None
    text_encoder = None
    vae = None

    if 'speech' in modalities:
        print(f'\nLoading speech encoder: {speech_type}')
        speech_encoder = SpeechEncoder(speech_type, freeze=True)

    if 'text' in modalities:
        print(f'\nLoading text encoder: {text_type}')
        text_encoder = load_text_encoder(text_type)

    if 'motion' in modalities:
        print('\nLoading motion VAE')
        vae = load_frozen_vae(vae_cfg)

    splits = args.splits.split(',')
    for split in splits:
        print(f'\n{"="*60}')
        print(f'Processing split: {split}')
        print(f'{"="*60}')

        dataset = ContrastiveH2SDataset(
            data_root=data_cfg['root'],
            split=split,
            mean=mean,
            std=std,
            max_motion_length=data_cfg['max_motion_length'],
            min_motion_length=data_cfg['min_motion_length'],
            audio_dir=data_cfg['audio_dir'],
            dataset_name=data_cfg['dataset_name'],
            unit_length=data_cfg.get('unit_length', 4),
            youtube3d_root=data_cfg.get('youtube3d_root', None),
        )

        if 'speech' in modalities:
            speech_dir = os.path.join(save_root, speech_type, split)
            print(f'\n  → Speech features: {speech_dir}')
            extract_speech_features(speech_encoder, dataset, speech_dir, device)

        if 'text' in modalities:
            text_dir = os.path.join(save_root, text_type, split)
            print(f'\n  → Text features: {text_dir}')
            extract_text_features(text_encoder, dataset, text_dir, device)

        if 'motion' in modalities:
            motion_dir = os.path.join(save_root, 'motion_rvq', split)
            print(f'\n  → Motion features: {motion_dir}')
            extract_motion_features(vae, dataset, motion_dir, device, vae_cfg)

    print(f'\nAll done! Features saved to: {save_root}')


if __name__ == '__main__':
    main()
