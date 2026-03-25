# Triplet Contrastive Learning: Speech-Text-Motion Alignment

Standalone module that learns a shared 512D embedding space aligning **Speech**, **Text**, and **Motion** via contrastive learning.

---

## Directory Structure

```
contrastive/
├── __init__.py
├── loss.py               # Symmetric InfoNCE (CLIP-style)
├── model.py              # TripletContrastiveModel + ModalityProjectionHead
├── dataset.py            # ContrastiveH2SDataset + collate function
├── train.py              # PyTorch Lightning training script
├── configs/
│   └── contrastive_h2s.yaml
└── README.md
```

---

## Architecture

```
Speech Waveform ──► SpeechEncoder (frozen, HuBERT-Large) ──► [B, T_s, 1024]
                          │
                    SpeechProjection (trainable)
                    Linear(1024→512) + PosEnc + TransformerEncoder(4L,8H) + MeanPool + L2Norm
                          ↓ [B, 512]

Text String ──────► CLIP (frozen) ──► [B, 1, 512]
                          │
                    TextProjection (trainable)
                    Linear(512→512) + PosEnc + TransformerEncoder(4L,8H) + MeanPool + L2Norm
                          ↓ [B, 512]

Raw Motion ───────► RVQVae.encoder (frozen) ──► [B, 512, T'] → permute → [B, T', 512]
                          │
                    MotionProjection (trainable)
                    Linear(512→512) + PosEnc + TransformerEncoder(4L,8H) + MeanPool + L2Norm
                          ↓ [B, 512]

                    ┌──────────────────────────────┐
                    │  3 Pairwise InfoNCE Losses:   │
                    │  L(speech, text)              │
                    │  L(speech, motion)            │
                    │  L(text, motion)              │
                    │  + learnable temperature τ    │
                    │  total = mean of 3 losses     │
                    └──────────────────────────────┘
```

**Frozen encoders** — no gradients flow through them.
**Trainable** — only the 3 projection heads + 1 learnable log-temperature scalar.

---

## How to Run

### 1. Prerequisites

Make sure you have:
- A pretrained RVQ-VAE checkpoint (from VQ training stage)
- Audio files at `{audio_dir}/speech/{split}_wavs/{SENTENCE_NAME}.wav`
- Mean/std tensors for motion normalization (`youtube3d_mean.pt`, `youtube3d_std.pt`)
- YouTube3D (or How2Sign) data with CSV annotations and pose files

### 2. Update Config

Edit `contrastive/configs/contrastive_h2s.yaml`:

```yaml
# Point to your data
data:
  root: /data/hwu/slg_data/Youtube3D
  youtube3d_root: /data/hwu/slg_data/Youtube3D
  audio_dir: /data/hwu/slg_data/Youtube3D
  mean_path: /data/hwu/slg_data/Youtube3D/youtube3d_mean.pt
  std_path: /data/hwu/slg_data/Youtube3D/youtube3d_std.pt

# Point to your pretrained VAE checkpoint
vae:
  checkpoint: "experiments/mgpt/YOUR_VAE_RUN/checkpoints/best.ckpt"

# Adjust GPUs
training:
  devices: [0, 1]
```

### 3. Train

From the **project root** (`MotionGPT/`):

```bash
# Single or multi-GPU
python -m contrastive.train --config contrastive/configs/contrastive_h2s.yaml

# Resume from checkpoint
python -m contrastive.train --config contrastive/configs/contrastive_h2s.yaml \
    --resume experiments/contrastive/checkpoints/last.ckpt
```

### 4. Monitor

Training logs to **Weights & Biases** (project: `SLG`, run name: `Contrastive_Speech_Text_Motion`).

Key metrics to watch:
| Metric | Description | Expected start (B=32) |
|--------|-------------|----------------------|
| `train/loss` | Average of 3 InfoNCE losses | ~3.47 (= log(32)) |
| `train/loss_st` | Speech ↔ Text | ~3.47 |
| `train/loss_sm` | Speech ↔ Motion | ~3.47 |
| `train/loss_tm` | Text ↔ Motion | ~3.47 |
| `train/logit_scale` | Temperature (1/τ) | ~14.3 (= 1/0.07) |

All 3 losses should decrease over training. `logit_scale` will evolve (clamped at 100).

### 5. Checkpoints

Saved to `experiments/contrastive/checkpoints/`:
- `last.ckpt` — most recent
- Top 3 by `val/loss` — e.g. `contrastive-epoch=05-val/loss=2.1234.ckpt`

---

## Implementation Details

### Loss Function (`loss.py`)

Symmetric InfoNCE (same as CLIP):
```
logits = scale * emb_a @ emb_b.T    # [B, B] similarity matrix
loss = (CE(logits, arange(B)) + CE(logits.T, arange(B))) / 2
```

### Projection Head (`model.py: ModalityProjectionHead`)

Each modality gets the same architecture:
1. `nn.Linear(input_dim → 512)` — dimension alignment
2. `PositionEmbedding` — sinusoidal positional encoding (from `mGPT/archs/pos_encoding.py`)
3. `nn.TransformerEncoder` — 4 layers, 8 heads, FFN dim = 2048
4. Masked mean pooling — respects padding masks
5. L2 normalization — unit sphere embeddings

### Dataset (`dataset.py: ContrastiveH2SDataset`)

- Loads annotations from CSV (reuses H2S/YouTube3D loading from `mGPT/data/humanml/load_data.py`)
- Loads audio via `mGPT/data/audio_utils.load_audio()` at 16kHz
- Filters out samples without audio files
- Motion normalized with precomputed mean/std, resampled to `[min_length, max_length]`
- Custom `contrastive_collate` pads motion and audio, skips failed samples

### Motion Encoding Path

```
raw_motion [B, T, 133]
    → vae.preprocess()    → [B, 133, T]
    → vae.encoder()       → [B, 512, T']     (T' = T/4, temporal compression)
    → permute(0,2,1)      → [B, T', 512]
    → motion_proj()       → [B, 512]          (L2-normalized)
```

### Training Config

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01) — only projection params + logit_scale
- **Scheduler**: CosineAnnealingLR over max_epochs
- **Precision**: 16-mixed (AMP)
- **Gradient clip**: 1.0
- **Strategy**: DDP if multiple GPUs, auto otherwise

---

## Supported Encoder Variants

### Speech Encoders
| Type | Output Dim | Model |
|------|-----------|-------|
| `hubert-base` | 768 | facebook/hubert-base-ls960 |
| `hubert-large` | 1024 | facebook/hubert-large-ll60k |
| `whisper-base` | 512 | openai/whisper-base |
| `whisper-large-v3` | 1280 | openai/whisper-large-v3 |

### Text Encoders
| Type | Output Dim | Notes |
|------|-----------|-------|
| `clip` | 512 | Sentence-level, [B, 1, 512] |
| `bert` | 768 | Token-level, [B, seq_len, 768] + mask |
| `bert-large` | 1024 | Token-level, [B, seq_len, 1024] + mask |

---

## Dependencies

No new dependencies beyond what mGPT already requires:
- `pytorch-lightning`
- `wandb`
- `transformers` (for HuBERT/Whisper)
- `clip` (OpenAI CLIP)
- `torchaudio`
