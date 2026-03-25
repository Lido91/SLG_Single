# Fine-Grained Contrastive Learning (paper 2507.23188)

Sequence-level contrastive model for **Speech-Motion-Text** retrieval, adapted from "Multi-Modal Motion Retrieval by Learning a Fine-Grained Joint Embedding Space" (Yu et al., 2025).

Paper: `document/2507.23188v1.pdf`

---

## Key Differences from Basic Contrastive (`model.py`)

| | Basic (`model.py`) | Fine-Grained (`model_finegrained.py`) |
|---|---|---|
| Similarity | Global: single 512D vector per sample | Sequence-level: token-level max similarity (Eq. 8) |
| Loss | InfoNCE (cross-entropy) | Bidirectional KL divergence (Eq. 5) |
| Motion tokens | Mean-pooled to 1 vector | Sequence of ~50 tokens |
| Audio tokens | Mean-pooled to 1 vector | 32 tokens (memory-retrieval compressed) |
| Text tokens | Mean-pooled to 1 vector | Full word-level token sequence |
| Auxiliary loss | None | Masked motion reconstruction (Eq. 10-11) |

---

## Architecture

```
Speech Waveform ──> WavLM (frozen) ──> [B, T_s, 1024]
                        │
                  MemoryRetrievalAudioEncoder
                  Linear(1024→512)
                  Cross-Attn(Q=audio, K=V=128 learnable memory tokens) x2
                  + residual + LayerNorm
                  PosEnc → AvgPool1d → L2Norm
                        ↓ [B, 32, 512]    ← 32 audio tokens

Text String ────> DistilBERT (frozen) ──> [B, L_t, 768]
                        │
                  SequenceTextEncoder
                  Linear(768→512) + PosEnc
                  TransformerEncoder(2 layers, 8 heads)
                  L2Norm
                        ↓ [B, L_t, 512]   ← ~15 text tokens

Raw Motion ─────> RVQVae.encoder (frozen) ──> [B, 512, T']
                        │
                  VaeMotionEncoder
                  Linear(512→512) + PosEnc
                  TransformerEncoder(2 layers)
                  AvgPool1d(kernel=2, stride=2)    ← paper Section IV
                  TransformerEncoder(2 layers)
                  L2Norm
                        ↓ [B, T'/2, 512]  ← ~50 motion tokens
```

### Frozen components
- WavLM-large (audio feature extraction)
- DistilBERT (text feature extraction)
- RVQVae encoder (motion feature extraction)

### Trainable components
- 3 projection + transformer heads (one per modality)
- 128 learnable memory tokens (audio encoder)
- 3 weight heads for token importance (Eq. 8: `w_x`, `w_y`)
- 1 learnable temperature (logit_scale)
- Reconstruction decoder (2-layer transformer)
- 1 mask token embedding

---

## Loss Design

### Sequence-Level Similarity (Eq. 8)

For a pair of samples `(x_i, y_j)` from two modalities:

```
h(x_i, y_j) = 0.5 * Σ_a  w_x[a] * max_b <x_i[a], y_j[b]>     (each x token finds best y match)
             + 0.5 * Σ_b  w_y[b] * max_a <y_j[b], x_i[a]>     (each y token finds best x match)
```

- `w_x`, `w_y`: learned softmax weights from `nn.Linear(512, 1)` per modality
- `<·,·>`: cosine similarity (tokens are L2-normalized)
- This produces a `[B, B]` similarity matrix per modality pair

### Alignment Loss (Eq. 4-5)

```
L_align = L(speech↔text) + L(speech↔motion) + L(text↔motion)    (sum, weight=1 each)

L(x↔y) = KL(softmax(sim/τ), I) + KL(softmax(sim.T/τ), I)       (bidirectional KL)
```

- Target = identity matrix (diagonal = positive pairs, off-diagonal = negatives)
- `τ` = learnable temperature, initialized to 0.07, clamped to [0.01, 1.0]

### Reconstruction Loss (Eq. 10-11)

```
1. Randomly mask 50% of motion tokens
2. Replace masked tokens with learnable [MASK] embedding
3. Concatenate [masked_motion, text_tokens, audio_tokens] + modality embeddings
4. Pass through 2-layer transformer decoder
5. L_recon = MSE(reconstructed[masked], original[masked])
```

### Total Loss (Eq. 12)

```
L = L_align + 0.1 * L_recon
```

---

## Batch Structure

```python
batch = {
    'text':         List[str],              # B strings
    'motion':       Tensor [B, T_max, 133], # zero-padded to batch max
    'audio':        Tensor [B, S_max],      # zero-padded 16kHz waveforms
    'length':       List[int],              # original motion frame counts
    'audio_length': List[int],              # original audio sample counts
}
```

Samples at the same index `i` are aligned: `text[i]`, `motion[i]`, `audio[i]` describe the **same action**. Negatives come from other samples in the batch.

---

## Files

```
contrastive/
├── model_finegrained.py       # Main model (this doc)
│   ├── VaeMotionEncoder         # Frozen RVQVae + transformer with intermediate pooling
│   ├── MemoryRetrievalAudioEncoder  # WavLM + memory cross-attention + avg pool
│   ├── SequenceTextEncoder      # DistilBERT + transformer
│   ├── ReconstructionDecoder    # Masked motion token reconstruction
│   └── FineGrainedContrastiveModel  # Lightning module
├── loss_finegrained.py        # Sequence-level similarity + KL loss
│   ├── sequence_level_similarity()  # Eq. 8
│   ├── bidirectional_kl_loss()      # Eq. 5
│   └── compute_alignment_loss()     # Combined
├── train_finegrained.py       # Training script
├── dataset.py                 # Shared dataset (same as basic contrastive)
└── configs/
    └── contrastive_finegrained.yaml
```

---

## How to Run

```bash
# From project root (MotionGPT/)
python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml

# Specify GPUs
python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml --use_gpus 0,1

# Resume
python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml \
    --resume experiments/contrastive_fg/checkpoints/last.ckpt
```

---

## Training Config

| Parameter | Value | Source |
|---|---|---|
| Latent dim (C) | 512 | Paper Section IV |
| Batch size | 32/GPU (64 total) | Paper: 64 on 8 GPUs |
| Epochs | 200 | Paper |
| Optimizer | AdamW (lr=1e-4, wd=0.01) | Paper |
| LR schedule | Linear decay to 1e-5 after 100 epochs | Paper |
| Precision | bf16-mixed | Adapted (paper doesn't specify) |
| Gradient clip | 1.0 | Stability |
| Temperature init | 0.07 (clamped to [0.01, 1.0]) | CLIP convention |
| λ_recon | 0.1 | Paper |
| Mask ratio | 0.5 | Default |
| Audio target length | 32 tokens | Config |
| Memory tokens | 128 | Config |
| Motion num_layers | 4 (2+pool+2) | Paper |
| Text num_layers | 2 | Paper |

---

## Key Metrics

| Metric | Description |
|---|---|
| `train/loss` | Total: L_align + 0.1 * L_recon |
| `train/loss_align` | Sum of 3 pairwise alignment losses |
| `train/loss_st` | Speech ↔ Text KL loss |
| `train/loss_sm` | Speech ↔ Motion KL loss |
| `train/loss_tm` | Text ↔ Motion KL loss |
| `train/loss_recon` | Masked motion reconstruction MSE |
| `train/temperature` | Learned 1/τ (starts ~14.3) |
| `val/avg_R@1` | Average Recall@1 across 6 retrieval directions |
| `val/S2M_R@1` | Speech→Motion Recall@1 |
| `val/T2M_R@1` | Text→Motion Recall@1 |

---

## Adaptations from Paper

The paper targets 4 modalities (text, audio, video, motion) on HumanML3D/KIT-ML. Our adaptation:

| Aspect | Paper | Our Implementation |
|---|---|---|
| Modalities | 4 (text, audio, video, motion) | 3 (text, audio, motion) — no video |
| Motion encoder | Body-part decomposition (8 parts) | Frozen RVQVae encoder |
| Audio backbone | WavLM | WavLM (same) |
| Text backbone | DistilBERT | DistilBERT (same) |
| Audio cross-attn | 1 layer, no residual | 2 layers + residual + LayerNorm |
| Data | HumanML3D, KIT-ML | YouTube3D / How2Sign (sign language) |
| Alignment pairs | 6 (all 4C2 combinations) | 3 (all 3C2 combinations) |

---

## Numerical Stability

Several guards against NaN/gradient explosion:

1. **bf16-mixed** instead of fp16-mixed — BF16 has same exponent range as FP32, prevents gradient overflow in the 4D `[B,B,Lx,Ly]` similarity tensor backward
2. **`autocast(enabled=False)`** — loss computation forced to FP32
3. **`logit_scale.clamp()` before `exp()`** — prevents exp overflow (`exp(>88)` = inf in FP32)
4. **`masked_fill` instead of `* mask`** — avoids `-inf * 0 = NaN` in similarity masking
5. **`nan_to_num(0.0)` after softmax** — guards against all-padded sequences
6. **Gradient clipping** at 1.0
