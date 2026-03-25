import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin
from copy import deepcopy

# Add project root for mGPT imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mGPT.data.humanml.load_data import (
    load_h2s_sample, load_youtube3d_sample,
    load_csl_sample, load_phoenix_sample,
)
from mGPT.data.audio_utils import load_audio

# Known broken How2Sign IDs
bad_how2sign_ids = [
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front',
    '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front',
    '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front',
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front',
    '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front',
    '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front',
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front',
    'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front',
    'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front',
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front',
    'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front',
    'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front',
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front',
    'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front',
    'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front',
    'g3Cc_1-V31U_12-3-rgb_front',
]


class ContrastiveH2SDataset(data.Dataset):
    """
    Dataset returning (text, raw_motion, audio, length) triplets
    for speech-text-motion contrastive learning.
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        audio_dir,
        dataset_name='youtube3d',
        unit_length=4,
        fps=20,
        youtube3d_root=None,
        require_audio=True,
        **kwargs,
    ):
        self.split = split
        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.youtube3d_root = youtube3d_root or data_root
        self.mean = mean[:133] if mean.shape[0] > 133 else mean
        self.std = std[:133] if std.shape[0] > 133 else std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.audio_dir = audio_dir
        self.require_audio = require_audio

        assert max_motion_length % unit_length == 0 and min_motion_length % unit_length == 0

        self.all_data = []

        if 'how2sign' in dataset_name:
            self._load_how2sign(data_root, split)

        if 'youtube3d' in dataset_name:
            self._load_youtube3d(self.youtube3d_root, split)

        # Filter to samples that have audio files (only when audio is needed)
        if require_audio:
            self._filter_with_audio()

        print(f'[Contrastive-{split}] {len(self.all_data)} samples')

    def _load_how2sign(self, data_root, split):
        data_dir = os.path.join(data_root, split, 'poses')
        csv_path = os.path.join(
            data_root, split, 're_aligned',
            f"how2sign_realigned_{split}_preprocessed_fps.csv",
        )
        if not os.path.exists(csv_path):
            print(f'[Contrastive] How2Sign CSV not found: {csv_path}')
            return

        csv = pd.read_csv(csv_path)
        csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
        csv = csv[csv['DURATION'] < 30].reset_index(drop=True)

        print(f'{split}--loading how2sign annotations...', len(csv))
        for idx in tqdm(range(len(csv))):
            name = csv.iloc[idx]['SENTENCE_NAME']
            if name in bad_how2sign_ids:
                continue
            self.all_data.append({
                'name': name,
                'fps': csv.iloc[idx]['fps'],
                'text': csv.iloc[idx]['SENTENCE'],
                'src': 'how2sign',
                'split': split,
                'data_dir': data_root,
                'poses_dir': data_dir,
            })

    def _load_youtube3d(self, youtube3d_root, split):
        poses_dir = os.path.join(youtube3d_root, split, 'poses')
        csv_path = os.path.join(
            youtube3d_root, split, 're_aligned',
            f"youtube_asl_{split}.csv",
        )
        if not os.path.exists(csv_path):
            print(f'[Contrastive] YouTube3D CSV not found: {csv_path}')
            return

        csv = pd.read_csv(csv_path)
        csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
        csv = csv[csv['DURATION'] < 30].reset_index(drop=True)

        print(f'{split}--loading youtube3d annotations...', len(csv))
        for idx in tqdm(range(len(csv))):
            name = csv.iloc[idx]['SENTENCE_NAME']
            text_val = csv.iloc[idx]['SENTENCE']
            if not isinstance(text_val, str) or pd.isna(text_val) or text_val.strip() == '':
                text_val = 'unknown'
            fps_val = csv.iloc[idx]['fps'] if 'fps' in csv.columns else 24
            self.all_data.append({
                'name': name,
                'fps': fps_val,
                'text': text_val,
                'src': 'youtube3d',
                'split': split,
                'data_dir': youtube3d_root,
                'poses_dir': poses_dir,
            })

    def _get_audio_path(self, sample):
        name = sample['name']
        return pjoin(self.audio_dir, 'speech', f"{self.split}_wavs", f"{name}.wav")

    def _filter_with_audio(self):
        """Keep only samples that have a corresponding audio file."""
        filtered = []
        for sample in self.all_data:
            audio_path = self._get_audio_path(sample)
            if os.path.exists(audio_path):
                filtered.append(sample)
        dropped = len(self.all_data) - len(filtered)
        if dropped > 0:
            print(f'[Contrastive-{self.split}] Dropped {dropped} samples without audio')
        self.all_data = filtered

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        try:
            sample = self.all_data[idx]
            src = sample['src']
            name = sample['name']

            # --- Load motion ---
            if src == 'how2sign':
                clip_poses, text, name, _ = load_h2s_sample(sample, sample['data_dir'])
            elif src == 'youtube3d':
                clip_poses, text, name, _ = load_youtube3d_sample(sample, sample['poses_dir'])
            else:
                return None

            if clip_poses is None or text is None:
                return None

            # Normalize
            clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)
            m_length = clip_poses.shape[0]

            # Resample to fit length constraints
            if m_length < self.min_motion_length:
                idx_resample = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
                clip_poses = clip_poses[idx_resample]
            elif m_length > self.max_motion_length:
                idx_resample = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
                clip_poses = clip_poses[idx_resample]
            else:
                m_length = (m_length // self.unit_length) * self.unit_length
                idx_start = (clip_poses.shape[0] - m_length) // 2
                clip_poses = clip_poses[idx_start:idx_start + m_length]
            m_length = clip_poses.shape[0]

            result = {
                'text': text,
                'motion': torch.from_numpy(clip_poses).float(),
                'length': m_length,
                'name': name,
                'src': src,
            }

            # --- Load audio (only when required) ---
            if self.require_audio:
                audio_path = self._get_audio_path(sample)
                audio = load_audio(audio_path, target_sr=16000, max_duration=30.0)
                result['audio'] = audio

            return result

        except Exception as e:
            import traceback
            print(f"[Contrastive Dataset ERROR] idx={idx}: {e}")
            traceback.print_exc()
            return None


def contrastive_collate(batch):
    """Custom collate that pads motion and audio, skipping None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    texts = [b['text'] for b in batch]
    names = [b['name'] for b in batch]
    srcs = [b['src'] for b in batch]
    lengths = [b['length'] for b in batch]

    # Pad motion to max T in batch
    max_motion_len = max(b['motion'].shape[0] for b in batch)
    nfeats = batch[0]['motion'].shape[1]
    padded_motion = torch.zeros(len(batch), max_motion_len, nfeats)
    for i, b in enumerate(batch):
        T = b['motion'].shape[0]
        padded_motion[i, :T] = b['motion']

    result = {
        'text': texts,
        'motion': padded_motion,
        'length': lengths,
        'name': names,
        'src': srcs,
    }

    # Pad audio to max num_samples in batch (only if present)
    if 'audio' in batch[0]:
        audio_lengths = [b['audio'].shape[0] for b in batch]
        max_audio_len = max(audio_lengths)
        padded_audio = torch.zeros(len(batch), max_audio_len)
        for i, b in enumerate(batch):
            L = b['audio'].shape[0]
            padded_audio[i, :L] = b['audio']
        result['audio'] = padded_audio
        result['audio_length'] = audio_lengths

    return result


class PrecomputedContrastiveDataset(data.Dataset):
    """
    Dataset that loads pre-extracted frozen encoder features (.pt files)
    instead of raw audio/motion data. Speeds up training by skipping
    frozen encoder forward passes.

    Uses ContrastiveH2SDataset internally to build the sample list.
    """

    def __init__(
        self,
        feat_root,
        speech_type,
        text_type,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        audio_dir,
        dataset_name='youtube3d',
        unit_length=4,
        youtube3d_root=None,
        require_speech=True,
        require_text=True,
        **kwargs,
    ):
        self.feat_root = feat_root
        self.speech_type = speech_type
        self.text_type = text_type
        self.split = split
        self.require_speech = require_speech
        self.require_text = require_text

        if self.require_speech and not self.speech_type:
            raise ValueError("speech_type is required when require_speech=True")
        if self.require_text and not self.text_type:
            raise ValueError("text_type is required when require_text=True")

        # Build sample list from original dataset (cheap — only loads metadata)
        base = ContrastiveH2SDataset(
            data_root=data_root,
            split=split,
            mean=mean,
            std=std,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            audio_dir=audio_dir,
            dataset_name=dataset_name,
            unit_length=unit_length,
            youtube3d_root=youtube3d_root,
            require_audio=False,
        )

        # Filter to samples that have required precomputed feature files
        self.samples = []
        for s in base.all_data:
            name = s['name']
            motion_path = pjoin(feat_root, 'motion_rvq', split, f'{name}.pt')
            speech_path = (pjoin(feat_root, speech_type, split, f'{name}.pt')
                           if self.require_speech else None)
            text_path = (pjoin(feat_root, text_type, split, f'{name}.pt')
                         if self.require_text else None)

            has_required = os.path.exists(motion_path)
            if self.require_speech:
                has_required = has_required and os.path.exists(speech_path)
            if self.require_text:
                has_required = has_required and os.path.exists(text_path)

            if has_required:
                sample = {
                    'name': name,
                    'text': s['text'],
                    'motion_path': motion_path,
                }
                if self.require_speech:
                    sample['speech_path'] = speech_path
                if self.require_text:
                    sample['text_path'] = text_path
                self.samples.append(sample)

        dropped = len(base.all_data) - len(self.samples)
        if dropped > 0:
            print(f'[Precomputed-{split}] Dropped {dropped} samples missing features')
        print(f'[Precomputed-{split}] {len(self.samples)} samples with precomputed features')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            s = self.samples[idx]

            motion_data = torch.load(s['motion_path'], map_location='cpu', weights_only=True)
            speech_data = None
            text_data = None
            if self.require_speech:
                speech_data = torch.load(s['speech_path'], map_location='cpu', weights_only=True)
            if self.require_text:
                text_data = torch.load(s['text_path'], map_location='cpu', weights_only=True)

            item = {
                'text': s['text'],
                'motion_feats': motion_data['feats'],           # [T', 512]
                'motion_length': motion_data['length'],
                'name': s['name'],
            }
            if speech_data is not None:
                item['speech_feats'] = speech_data['feats']      # [T_s, D]
                item['speech_length'] = speech_data['length']
            if text_data is not None:
                item['text_feats'] = text_data['feats']          # [seq, D]
                item['text_mask'] = text_data.get('mask', None)  # [seq] or None
            return item
        except Exception as e:
            import traceback
            print(f"[Precomputed Dataset ERROR] idx={idx}: {e}")
            traceback.print_exc()
            return None


def precomputed_collate(batch):
    """Collate for PrecomputedContrastiveDataset — pads feature tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    texts = [b['text'] for b in batch]
    names = [b['name'] for b in batch]
    motion_lengths = [b['motion_length'] for b in batch]

    # Pad motion features [T', 512] → [B, max_T', 512]
    max_motion = max(b['motion_feats'].shape[0] for b in batch)
    motion_dim = batch[0]['motion_feats'].shape[1]
    padded_motion = torch.zeros(len(batch), max_motion, motion_dim)
    motion_mask = torch.zeros(len(batch), max_motion)
    for i, b in enumerate(batch):
        T = b['motion_feats'].shape[0]
        padded_motion[i, :T] = b['motion_feats']
        ml = b['motion_length']
        if isinstance(ml, torch.Tensor):
            ml = ml.item()
        motion_mask[i, :min(ml, T)] = 1.0

    result = {
        'text': texts,
        'name': names,
        'motion_feats': padded_motion,
        'motion_mask': motion_mask,
        'motion_length': motion_lengths,
    }

    if 'speech_feats' in batch[0]:
        speech_lengths = [b['speech_length'] for b in batch]
        max_speech = max(b['speech_feats'].shape[0] for b in batch)
        speech_dim = batch[0]['speech_feats'].shape[1]
        padded_speech = torch.zeros(len(batch), max_speech, speech_dim)
        speech_mask = torch.zeros(len(batch), max_speech)
        for i, b in enumerate(batch):
            T = b['speech_feats'].shape[0]
            padded_speech[i, :T] = b['speech_feats']
            sl = b['speech_length']
            if isinstance(sl, torch.Tensor):
                sl = sl.item()
            speech_mask[i, :min(sl, T)] = 1.0
        result['speech_feats'] = padded_speech
        result['speech_mask'] = speech_mask
        result['speech_length'] = speech_lengths

    if 'text_feats' in batch[0]:
        max_text = max(b['text_feats'].shape[0] for b in batch)
        text_dim = batch[0]['text_feats'].shape[1]
        padded_text = torch.zeros(len(batch), max_text, text_dim)
        text_mask = torch.zeros(len(batch), max_text)
        for i, b in enumerate(batch):
            L = b['text_feats'].shape[0]
            padded_text[i, :L] = b['text_feats']
            if b['text_mask'] is not None:
                text_mask[i, :L] = b['text_mask']
            else:
                text_mask[i, :L] = 1.0
        result['text_feats'] = padded_text
        result['text_mask'] = text_mask

    return result
