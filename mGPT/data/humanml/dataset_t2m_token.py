import random
import numpy as np
from torch.utils import data
from .dataset_t2m import Text2MotionDataset
import codecs as cs
from os.path import join as pjoin
import os
import pandas as pd
import math
import torch
import pickle, gzip
from copy import deepcopy
from tqdm import tqdm
from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_iso_sample, load_youtube3d_sample
from ..audio_utils import load_audio, check_audio_motion_sync

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']

class Text2MotionDatasetToken(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.split = split  # Store split for audio path construction
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        self.youtube3d_root = kwargs.get('youtube3d_root', None)
        self.balanced = kwargs.get('balanced', False)
        self.gloss = kwargs.get('gloss', '')

        # Audio support for speech-driven generation
        self.audio_dir = kwargs.get('audio_dir', None)
        self.use_speech = kwargs.get('use_speech', False)
        self.preload_audio = kwargs.get('preload_audio', False)

        # Precomputed speech feature support (skip HuBERT at training time)
        self.precomputed_speech_dir = kwargs.get('precomputed_speech_dir', None)
        self.use_precomputed_speech = self.precomputed_speech_dir is not None

        if self.use_speech:
            if self.use_precomputed_speech:
                print(f"Precomputed speech feature mode enabled. Dir: {self.precomputed_speech_dir}")
            else:
                print(f"Audio mode enabled. Audio directory: {self.audio_dir}")
                if self.preload_audio:
                    print("Audio preloading enabled (will cache audio in RAM)")

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
            self.csv_path = os.path.join(data_root, split, 're_aligned', 'how2sign_realigned_'+split+'_preprocessed_fps.csv')
            self.csv = pd.read_csv(self.csv_path)
            self.fps = self.csv['fps']
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION']<30].reset_index(drop=True) # remove sequences longer than 30 seconds
            self.ids = self.csv['SENTENCE_NAME'] #[:100]

            print(f'{split}--loading how2sign annotations...', len(self.ids))
            for idx in tqdm(range(len(self.ids))):
                name = self.ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({'name': name, 'fps': self.csv[self.csv['SENTENCE_NAME']==name]['fps'].item(), 
                                        'text': self.csv[self.csv['SENTENCE_NAME']==name]['SENTENCE'].item(), 'src': 'how2sign'})
            self.h2s_len += len(self.all_data)
            
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:100]

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
                self.ann = pickle.load(f) #[:100]

            print(f'{split}--loading phoenix annotations...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
            self.phoenix_len += len(self.ann)

        if 'youtube3d' in dataset_name:
            self.youtube3d_data_dir = os.path.join(self.youtube3d_root, split, 'poses')

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
                    youtube3d_csv_path = _make_youtube3d_csv(f"{base_suffix}_gloss_qwen8b")
                elif self.gloss == "t5":
                    youtube3d_csv_path = _make_youtube3d_csv(f"{base_suffix}_gloss_t5")
                else:
                    raise ValueError(f"Unknown gloss type: {self.gloss}")
            else:
                youtube3d_csv_path = _make_youtube3d_csv(base_suffix)

            youtube3d_csv = pd.read_csv(youtube3d_csv_path)
            youtube3d_csv['DURATION'] = youtube3d_csv['END_REALIGNED'] - youtube3d_csv['START_REALIGNED']
            youtube3d_csv = youtube3d_csv[youtube3d_csv['DURATION'] < 30].reset_index(drop=True)
            youtube3d_ids = youtube3d_csv['SENTENCE_NAME']

            print(f'{split}--loading youtube3d annotations...', len(youtube3d_ids))
            for idx in tqdm(range(len(youtube3d_ids))):
                name = youtube3d_ids[idx]
                text_val = youtube3d_csv[youtube3d_csv['SENTENCE_NAME'] == name]['SENTENCE'].item()
                if not isinstance(text_val, str) or pd.isna(text_val):
                    text_val = "unknown"
                elif text_val.strip() == "":
                    text_val = "unknown"
                fps_val = youtube3d_csv[youtube3d_csv['SENTENCE_NAME'] == name]['fps'].item() if 'fps' in youtube3d_csv.columns else 24
                self.all_data.append({'name': name, 'fps': fps_val,
                                      'text': text_val, 'src': 'youtube3d'})
            self.youtube3d_len += len(youtube3d_ids)

        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}, YouTube3D: {self.youtube3d_len}')

        # Preload audio if requested
        if self.use_speech and self.preload_audio:
            self._preload_audio_cache()

    def _preload_audio_cache(self):
        """Preload all audio files into RAM for faster training"""
        self.audio_cache = {}
        print("Preloading audio files...")
        for sample in tqdm(self.all_data, desc="Caching audio"):
            name = sample['name']
            audio_path = self._get_audio_path(sample)
            if audio_path and os.path.exists(audio_path):
                try:
                    audio = load_audio(audio_path, target_sr=16000)
                    self.audio_cache[name] = audio
                except Exception as e:
                    print(f"Failed to load audio for {name}: {e}")

    def _get_audio_path(self, sample):
        """
        Get audio file path for a sample.

        Audio directory structure:
        - Youtube3D: {audio_dir}/speech/{split}_wavs/{SENTENCE_NAME}.wav
        - How2Sign: {audio_dir}/speech/{split}_wavs/{SENTENCE_NAME}.wav
        """
        if not self.audio_dir:
            return None

        name = sample['name']
        src = sample['src']

        # Audio path: {audio_dir}/speech/{split}_wavs/{SENTENCE_NAME}.wav
        audio_path = pjoin(self.audio_dir, 'speech', f"{self.split}_wavs", f"{name}.wav")

        return audio_path

    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        name = sample['name']

        # Load motion data
        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.data_dir)
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
            raise ValueError(f"Unknown source type: {src}")

        # Normalize motion
        clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
        m_length = clip_poses.shape[0]

        # Resample motion to fit length constraints
        if m_length < self.min_motion_length:
            idx = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        elif m_length > self.max_motion_length:
            idx = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[idx:idx + m_length]
        m_length = clip_poses.shape[0]

        # NEW: Load audio if in speech mode
        audio = None
        speech_feats = None
        speech_length = None
        if self.use_speech:
            if self.use_precomputed_speech:
                # Load precomputed HuBERT features
                feat_path = pjoin(self.precomputed_speech_dir, self.split, f'{name}.pt')
                if os.path.exists(feat_path):
                    try:
                        feat_data = torch.load(feat_path, map_location='cpu', weights_only=True)
                        speech_feats = feat_data['feats']    # [T_s, D]
                        speech_length = feat_data['length']  # int
                    except Exception as e:
                        print(f"Warning: Failed to load precomputed speech for {name}: {e}")
                else:
                    print(f"Warning: Missing precomputed speech feature: {feat_path}")
            elif self.preload_audio and name in self.audio_cache:
                # Use cached audio
                audio = self.audio_cache[name]
            else:
                # Load audio on-the-fly
                audio_path = self._get_audio_path(sample)
                if audio_path and os.path.exists(audio_path):
                    try:
                        audio = load_audio(audio_path, target_sr=16000)
                    except Exception as e:
                        print(f"Warning: Failed to load audio for {name}: {e}")
                        # Fallback to silent audio (3 seconds)
                        audio = torch.zeros(16000 * 3)
                else:
                    # Missing audio file, use silent audio
                    audio = torch.zeros(16000 * 3)

        # Return format: (name, poses, length, ..., src, audio, speech_feats, speech_length)
        return name, torch.from_numpy(clip_poses).float(), m_length, True, True, True, True, True, True, src, audio, speech_feats, speech_length

