import random
import torch
import pickle, gzip
import os, math
import pandas as pd
import codecs as cs
import numpy as np
from tqdm import tqdm
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset
from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_iso_sample, load_youtube3d_sample
import random; random.seed(0)
import json
from copy import deepcopy

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']


class H2SMotionDatasetVQ(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        dataset_name='how2sign',
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        # split='train'
        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        self.youtube3d_root = kwargs.get('youtube3d_root', None)
        self.balanced = kwargs.get('balanced', False)
        self.gloss = kwargs.get('gloss', '')
        self.clip_feat_dir = kwargs.get('clip_feat_dir', None)
        self.audio_dir = kwargs.get('audio_dir', None)
        self.use_speech = kwargs.get('use_speech', False)
        self.precomputed_speech_dir = kwargs.get('precomputed_speech_dir', None)
        self.split = split
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        assert max_motion_length % unit_length == 0 and min_motion_length % unit_length == 0

        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = self.youtube3d_len = 0

        if 'how2sign' in dataset_name:
            self.data_dir = os.path.join(data_root, split, 'poses')

            # -------------------------
            # Helper: pick CSV by suffix
            # -------------------------
            def _make_csv(suffix: str = "") -> str:
                # suffix includes leading "_" if needed
                return os.path.join(
                    data_root, split, 're_aligned',
                    f"how2sign_realigned_{split}_preprocessed_fps{suffix}.csv"
                )

            # -------------------------
            # 1) base or balanced csv
            # -------------------------
            if self.balanced is True:
                base_suffix = "_filtered"
                print(f"[VAE-{split}] Using balanced (filtered) split")
            else:
                base_suffix = ""
                print(f"[VAE-{split}] Using original split")

            # -------------------------
            # 2) raw text or gloss csv (VAE doesn't need this, but use if available)
            # -------------------------
            if self.gloss:
                if self.gloss == "qwen":
                    self.csv_path = _make_csv(f"{base_suffix}_gloss_qwen8b")
                    print(f"[VAE-{split}] Using Qwen gloss CSV (motion data only)")
                elif self.gloss == "t5":
                    self.csv_path = _make_csv(f"{base_suffix}_gloss_t5")
                    print(f"[VAE-{split}] Using T5 gloss CSV (motion data only)")
                else:
                    raise ValueError(f"Unknown gloss type: {self.gloss} (expected 'qwen' or 't5')")
            else:
                self.csv_path = _make_csv(base_suffix)
                print(f"[VAE-{split}] Using raw text CSV (motion data only)")

            self.csv = pd.read_csv(self.csv_path)
            self.fps = self.csv['fps']
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION']<30].reset_index(drop=True) # remove sequences longer than 30 seconds
            self.ids = self.csv['SENTENCE_NAME'] #[:200]

            print(f'{split}--loading how2sign annotations...', len(self.ids))
            bad_set = set(bad_how2sign_ids)
            for row in tqdm(self.csv.itertuples(index=False), total=len(self.csv)):
                name = row.SENTENCE_NAME
                if name in bad_set:
                    continue
                self.all_data.append({'name': name, 'fps': row.fps,
                                        'text': row.SENTENCE, 'src': 'how2sign', 'split': split})
            self.h2s_len += len(self.all_data)
        
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print(f'{split}--loading csl annotations...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'csl'
                self.all_data.append(ann)
            self.csl_len += len(self.ann)

        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print(f'{split}--loading phoenix annotations...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
            self.phoenix_len += len(self.ann)

        if 'youtube3d' in dataset_name:
            self.youtube3d_data_dir = os.path.join(self.youtube3d_root, split, 'poses')

            # Build CSV path for youtube3d
            def _make_youtube3d_csv(suffix: str = "") -> str:
                return os.path.join(
                    self.youtube3d_root, split, 're_aligned',
                    f"youtube_asl_{split}{suffix}.csv"
                )

            if self.balanced is True:
                base_suffix = "_filtered"
            else:
                base_suffix = ""

            if self.gloss:
                if self.gloss == "qwen":
                    youtube3d_csv_path = _make_youtube3d_csv(f"{base_suffix}_qwen8b")
                elif self.gloss == "t5":
                    youtube3d_csv_path = _make_youtube3d_csv(f"{base_suffix}_t5")
                else:
                    raise ValueError(f"Unknown gloss type: {self.gloss}")
            else:
                youtube3d_csv_path = _make_youtube3d_csv(base_suffix)

            youtube3d_csv = pd.read_csv(youtube3d_csv_path)
            youtube3d_csv['DURATION'] = youtube3d_csv['END_REALIGNED'] - youtube3d_csv['START_REALIGNED']
            youtube3d_csv = youtube3d_csv[youtube3d_csv['DURATION'] < 30].reset_index(drop=True)

            print(f'{split}--loading youtube3d annotations...', len(youtube3d_csv))
            has_fps_col = 'fps' in youtube3d_csv.columns
            for row in tqdm(youtube3d_csv.itertuples(index=False), total=len(youtube3d_csv)):
                name = row.SENTENCE_NAME
                text_val = row.SENTENCE
                if not isinstance(text_val, str) or pd.isna(text_val) or (isinstance(text_val, str) and text_val.strip() == ""):
                    text_val = "unknown"
                fps_val = row.fps if has_fps_col else 24
                self.all_data.append({'name': name, 'fps': fps_val,
                                      'text': text_val, 'src': 'youtube3d', 'split': split})
            self.youtube3d_len = len(youtube3d_csv)

        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}, YouTube3D: {self.youtube3d_len}')

        # Preload all speech features into a single contiguous tensor (shared across workers)
        self._speech_offsets = {}  # name -> (start, length)
        self._speech_feats_all = None
        if self.precomputed_speech_dir:
            feat_dir = os.path.join(self.precomputed_speech_dir, split)
            if os.path.isdir(feat_dir):
                names = [d['name'] for d in self.all_data]
                print(f'[{split}] Preloading speech features from {feat_dir}...')
                feats_list = []
                total_frames = 0
                for name in tqdm(names):
                    feat_path = os.path.join(feat_dir, f'{name}.pt')
                    if os.path.exists(feat_path):
                        feat_data = torch.load(feat_path, map_location='cpu', weights_only=True)
                        f = feat_data['features'].float()  # [seq_len, D]
                        self._speech_offsets[name] = (total_frames, f.shape[0])
                        feats_list.append(f)
                        total_frames += f.shape[0]
                if feats_list:
                    self._speech_feats_all = torch.cat(feats_list, dim=0)  # [total_frames, D]
                print(f'[{split}] Loaded {len(self._speech_offsets)}/{len(names)} speech features ({total_frames} frames)')
        

    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        try:
            sample = self.all_data[idx]
            src = sample.get('src', 'unknown')
            name = sample.get('name', 'unknown')

            # Load the sample data
            if src == 'how2sign':
                clip_poses, text, name, _ = load_h2s_sample(sample, self.root_dir)
            elif src == 'csl':
                clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
            elif src == 'phoenix':
                clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
            elif src == 'youtube3d':
                clip_poses, text, name, _ = load_youtube3d_sample(sample, self.youtube3d_data_dir)
            elif src == 'asl_iso':
                clip_poses, text, name, _ = load_iso_sample(sample, self.root_dir, dataset='asl_iso')
                src = 'how2sign'
            elif src == 'csl_iso':
                clip_poses, text, name, _ = load_iso_sample(sample, self.csl_root, dataset='csl_iso')
                src = 'csl'
            elif src == 'phoenix_iso':
                clip_poses, text, name, _ = load_iso_sample(sample, self.phoenix_root, dataset='phoenix_iso')
                src = 'phoenix'
            else:
                print(f"[VAE Dataset] Unknown source '{src}' (repr={repr(src)}) for idx={idx}, sample keys={list(sample.keys())}")
                return None

            # Check if loading failed (returns None for invalid samples like short sequences)
            if clip_poses is None or text is None:
                # Silently skip - this is normal for sequences < 4 frames
                return None

            # Normalize poses
            clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
            m_length = clip_poses.shape[0]

            # Resample to fit length constraints
            if m_length < self.min_motion_length:
                idx_resample = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
                clip_poses = clip_poses[idx_resample]
            elif m_length > self.max_motion_length:
                idx_resample = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
                clip_poses = clip_poses[idx_resample]
            else:
                m_length = (m_length // self.unit_length) * self.unit_length
                idx_start = (clip_poses.shape[0] - m_length) // 2
                clip_poses = clip_poses[idx_start:idx_start + m_length]
            m_length = clip_poses.shape[0]

            # Load precomputed CLIP text feature if available
            clip_text_feat = None
            if self.clip_feat_dir is not None:
                feat_path = os.path.join(self.clip_feat_dir, self.split, f'{name}.pt')
                if os.path.exists(feat_path):
                    feat_data = torch.load(feat_path, map_location='cpu', weights_only=True)
                    clip_text_feat = feat_data['feats'].squeeze(0)  # [512]

            # Load audio / precomputed speech features
            audio_waveform = None
            speech_feats = None
            speech_length = None
            if name in self._speech_offsets:
                start, length = self._speech_offsets[name]
                speech_feats = self._speech_feats_all[start:start + length]  # [seq_len, D]
                speech_length = length
            elif self.use_speech and self.audio_dir:
                audio_path = os.path.join(self.audio_dir, 'speech', f'{self.split}_wavs', f'{name}.wav')
                if os.path.exists(audio_path):
                    from ..audio_utils import load_audio
                    audio_waveform = load_audio(audio_path)  # [num_samples]

            # Return 13-element tuple for collate function
            # Index 10 = audio waveform, 11 = precomputed speech feats, 12 = speech length
            return text, torch.from_numpy(clip_poses).float(), m_length, name, None, None, None, clip_text_feat, None, src, audio_waveform, speech_feats, speech_length

        except Exception as e:
            import traceback
            print(f"[VAE Dataset ERROR] Failed loading idx={idx}: {e}")
            traceback.print_exc()
            return None
