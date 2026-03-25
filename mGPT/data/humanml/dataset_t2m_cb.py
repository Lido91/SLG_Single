import rich
import random
import pickle
import os, gzip
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_youtube3d_sample
from ..audio_utils import load_audio

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']

class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length
        self.data_root = data_root
        self.split = split
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        self.youtube3d_root = kwargs.get('youtube3d_root', None)
        self.balanced = kwargs.get('balanced', False)
        self.gloss = kwargs.get('gloss', '')

        # Audio support for speech-driven generation
        self.audio_dir = kwargs.get('audio_dir', None)
        self.use_speech = kwargs.get('use_speech', False)
        self.precomputed_speech_dir = kwargs.get('precomputed_speech_dir', None)
        self.use_precomputed_speech = self.precomputed_speech_dir is not None

        # Data mean and std
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        assert max_motion_length % unit_length == 0 and min_motion_length % unit_length == 0
        self.max_motion_length = max_motion_length // self.unit_length
        self.min_motion_length = min_motion_length // self.unit_length  #4x downsampling in code

        # Data path
        # Use the split parameter passed to __init__ (train/val/test)
        self.code_path = code_path
        
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin('prepare/instructions', 'pretrain.json')
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin('prepare/instructions', 'instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")
        
        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = self.youtube3d_len = 0
        if 'how2sign' in dataset_name:
            self.data_dir = os.path.join(data_root, split, 'poses')
            self.csv_path = os.path.join(data_root, split, 're_aligned', 'how2sign_realigned_'+split+'_preprocessed_fps.csv')
            self.csv = pd.read_csv(self.csv_path)
            self.fps = self.csv['fps']
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION']<30].reset_index(drop=True) # remove sequences longer than 30 seconds
            self.ids = self.csv['SENTENCE_NAME'] #[:200]

            print('loading how2sign data...', len(self.ids))
            for idx in tqdm(range(len(self.ids))):
                name = self.ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({'name': name, 'fps': self.csv[self.csv['SENTENCE_NAME']==name]['fps'].item(), 
                                        'text': self.csv[self.csv['SENTENCE_NAME']==name]['SENTENCE'].item(), 'src': 'how2sign'})
                # _, text, n, code = load_h2s_sample(idx, self.ids, self.csv, self.data_dir, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue  #some samples are missing due to too short length
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'how2sign'})
            self.h2s_len = len(self.all_data)
        
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print('loading csl data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'csl'
                self.all_data.append(ann)
                # _, text, n, code = load_csl_sample(idx, self.ann, self.csl_root, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'csl'})
            self.csl_len = len(self.ann)

        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print('loading phoenix data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
                # _, text, n, code = load_phoenix_sample(idx, self.ann, self.phoenix_root, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'phoenix'})
            self.phoenix_len = len(self.ann)

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
            youtube3d_ids = youtube3d_csv['SENTENCE_NAME']

            print(f'{split}--loading youtube3d data...', len(youtube3d_ids))
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
            self.youtube3d_len = len(youtube3d_ids)

        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}, YouTube3D: {self.youtube3d_len}')

        # self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])


    def __len__(self):
        return len(self.all_data) * len(self.tasks)


    def __getitem__(self, idx):
        data_idx = idx % len(self.all_data)
        task_idx = idx // len(self.all_data)
        sample = self.all_data[data_idx]
        src = sample['src']
        # caption = sample['text']
        # m_tokens = sample['code']

        # Determine if we should load precomputed codes
        need_code = self.code_path is not None
        code_full_path = os.path.join(self.data_root, self.code_path) if self.code_path else None

        if src == 'how2sign':
            _, caption, name, m_tokens = load_h2s_sample(sample, self.data_dir, need_pose=False, code_path=code_full_path, need_code=need_code)
        elif src == 'csl':
            _, caption, name, m_tokens = load_csl_sample(sample, self.csl_root, need_pose=False, code_path=code_full_path, need_code=need_code)
        elif src == 'phoenix':
            _, caption, name, m_tokens = load_phoenix_sample(sample, self.phoenix_root, need_pose=False, code_path=code_full_path, need_code=need_code)
        elif src == 'youtube3d':
            _, caption, name, m_tokens = load_youtube3d_sample(sample, self.youtube3d_data_dir, need_pose=False, code_path=code_full_path, need_code=need_code)
        else:
            raise ValueError(f"Unknown source type: {src}")

        all_captions = [caption]
        # print(m_tokens.shape)
        m_length = m_tokens.shape[0]
        if m_length < self.min_motion_length:
            idx = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
            m_tokens = m_tokens[idx]
        elif m_length > self.max_motion_length:
            idx = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
            m_tokens = m_tokens[idx]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = (m_tokens.shape[0] - m_length) // 2
            m_tokens = m_tokens[idx:idx + m_length]

        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_length = m_tokens.shape[0]

        tasks = self.tasks[task_idx]

        # Load audio / precomputed speech features
        audio = None
        speech_feats = None
        speech_length = None
        if self.use_speech:
            if self.use_precomputed_speech:
                feat_path = pjoin(self.precomputed_speech_dir, self.split, f'{name}.pt')
                if os.path.exists(feat_path):
                    try:
                        feat_data = torch.load(feat_path, map_location='cpu', weights_only=True)
                        speech_feats = feat_data['feats']
                        speech_length = feat_data['length']
                    except Exception as e:
                        print(f"Warning: Failed to load precomputed speech for {name}: {e}")
                else:
                    print(f"Warning: Missing precomputed speech feature: {feat_path}")
            elif self.audio_dir:
                audio_path = pjoin(self.audio_dir, 'speech', f"{self.split}_wavs", f"{name}.wav")
                if os.path.exists(audio_path):
                    try:
                        audio = load_audio(audio_path, target_sr=16000)
                    except Exception as e:
                        audio = torch.zeros(16000 * 3)
                else:
                    audio = torch.zeros(16000 * 3)

        return caption, torch.from_numpy(m_tokens).long(), m_length, name, None, None, None, all_captions, tasks, src, audio, speech_feats, speech_length
