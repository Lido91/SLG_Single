# Fine-Grained Contrastive Learning for Motion Retrieval

Based on paper: **"Multi-Modal Motion Retrieval by Learning a Fine-Grained Joint Embedding Space"** (arXiv 2507.23188)

Adapted for **Speech-Motion-Text** (3 modalities, no Video).

## Run Command

```bash
python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml
python -m contrastive.train_finegrained --config contrastive/configs/contrastive_finegrained.yaml --use_gpus 0,1
```

---

## File Structure

All new files live alongside the original contrastive module. **No existing files are modified.**

```
contrastive/
├── model.py                 # UNCHANGED — original global-embedding model
├── loss.py                  # UNCHANGED — original symmetric InfoNCE
├── train.py                 # UNCHANGED — original training script
├── dataset.py               # UNCHANGED — reused by both old and new
├── evaluate.py              # UNCHANGED
│
├── model_finegrained.py     # NEW — fine-grained model (this doc)
├── loss_finegrained.py      # NEW — sequence-level loss functions
├── train_finegrained.py     # NEW — training script
│
└── configs/
    ├── contrastive_h2s.yaml         # UNCHANGED
    ├── contrastive_how2sign.yaml    # UNCHANGED
    └── contrastive_finegrained.yaml # NEW — config for fine-grained model
```

---

## Architecture Overview

### Old Model vs New Model — Key Difference

The old model compresses each modality into **1 global vector** then computes cosine similarity. The new model keeps **token sequences** and computes fine-grained token-level similarity.

```
Old model (model.py):
  Motion [B,T,133] → VAE(frozen) → TransformerProj → MeanPool → 1 vector [B,512]
  Audio  [B,N]     → HuBERT(frozen) → TransformerProj → MeanPool → 1 vector [B,512]
  Text   list[str] → CLIP(frozen) → TransformerProj → MeanPool → 1 vector [B,512]
  Loss: InfoNCE on 3 pairs of global vectors

New model (model_finegrained.py):
  Motion [B,T,133] → VAE(frozen) → Linear+PosEnc+Transformer → T' tokens [B,T',512]
  Audio  [B,N]     → WavLM(frozen) → MemoryRetrieval+AvgPool  → 32 tokens [B,32,512]
  Text   list[str] → DistilBERT(frozen) → Transformer          → L_t tokens [B,L_t,512]
  Loss: Sequence-level KL on 3 pairs of token sequences + Reconstruction
```

### Complete Training Step Data Flow

```
Input batch:
  texts:         list[str]        — B sentences
  motion:        [B, T, 133]      — normalized SMPL-X upper body
  audio:         [B, num_samples] — 16kHz waveform
  lengths:       list[int]        — valid frame counts per sample
  audio_lengths: list[int]        — valid sample counts per audio

                        ┌──────────────────────┐
  motion [B,T,133]    → │    VaeMotionEncoder   │ → motion_tokens [B, T', 512]
                        │  (frozen VAE + TF)    │   motion_mask   [B, T']
                        └──────────────────────┘
                             T' = T / 4 (temporal compression)

                        ┌──────────────────────┐
  audio [B,N]         → │ MemoryRetrievalAudio │ → speech_tokens [B, 32, 512]
                        │  (frozen WavLM)       │   speech_mask   [B, 32]
                        └──────────────────────┘

                        ┌──────────────────────┐
  texts list[str]     → │  SequenceTextEncoder  │ → text_tokens   [B, L_t, 512]
                        │ (frozen DistilBERT)   │   text_mask     [B, L_t]
                        └──────────────────────┘

                              │ All tokens are L2-normalized per-token
                              ▼

            ┌─────────── Loss Computation ───────────┐
            │                                         │
            │  L_align = (L_st + L_sm + L_tm) / 3    │
            │  L_recon = masked motion reconstruction │
            │  L_total = L_align + 0.1 * L_recon      │
            │                                         │
            └─────────────────────────────────────────┘
```

---

## Three Encoders

### 1. VaeMotionEncoder

**File:** `model_finegrained.py` class `VaeMotionEncoder`

Uses the same frozen RVQVae encoder as the old model, but does NOT pool to a single vector.

```
Pipeline:
  motion [B, T, 133]
    → vae.preprocess()           → [B, 133, T]
    → vae.encoder() (frozen)     → [B, 512, T']     where T' = T / 4
    → permute                    → [B, T', 512]
    → Linear(512, 512)           → [B, T', 512]      (trainable)
    → PositionalEncoding         → [B, T', 512]
    → TransformerEncoder(4 layers, 8 heads)           (trainable)
    → L2 normalize per token     → [B, T', 512]

Output: T' tokens per sample (e.g. T=400 → T'=100)
```

**Frozen:** RVQVae encoder weights (loaded from checkpoint, no gradients)
**Trainable:** Linear projection, positional encoding, 4-layer transformer

### 2. MemoryRetrievalAudioEncoder

**File:** `model_finegrained.py` class `MemoryRetrievalAudioEncoder`

Paper Section III-A.4 / Fig.5. Compresses variable-length audio features into fixed-length tokens using learnable memory.

```
Pipeline:
  audio [B, num_samples] at 16kHz
    → WavLM(frozen)              → [B, T_s, 1024]    (T_s varies per sample)
    → Linear(1024, 512)          → [B, T_s, 512]     (trainable)
    → Cross-Attention × 2 layers:
        Q = audio features  [B, T_s, 512]
        K = memory tokens   [B, 128, 512]   (learnable parameter)
        V = memory tokens   [B, 128, 512]
        → Attention(Q,K,V)  → [B, T_s, 512]
        → LayerNorm(residual)
    → PositionalEncoding         → [B, T_s, 512]
    → AdaptiveAvgPool1d(32)      → [B, 32, 512]      (fixed length!)
    → L2 normalize per token     → [B, 32, 512]

Output: always 32 tokens regardless of audio length
```

**Frozen:** WavLM encoder
**Trainable:** Linear projection, 128 memory tokens (nn.Parameter), 2 cross-attention layers + norms, positional encoding

**Why memory-retrieval instead of simple AvgPool?**
Paper Table VII ablation shows memory-retrieval (R@1=16.19) >> AvgPool-4 (R@1=9.33) >> Conv1D (R@1=9.03).
The memory tokens learn to filter out speaker identity / noise and retain motion-relevant semantics.

### 3. SequenceTextEncoder

**File:** `model_finegrained.py` class `SequenceTextEncoder`

Paper Section III-A.2. Keeps the full token sequence instead of collapsing to CLS/global token.

```
Pipeline (distilbert mode):
  texts list[str]
    → DistilBERT tokenizer       → input_ids [B, L_t]
    → DistilBERT(frozen)         → [B, L_t, 768]     (all token embeddings!)
    → Linear(768, 512)           → [B, L_t, 512]     (trainable)
    → PositionalEncoding
    → TransformerEncoder(2 layers)                     (trainable)
    → L2 normalize per token     → [B, L_t, 512]

Output: L_t tokens per sample (L_t = number of word tokens)
```

Also supports `clip` mode for compatibility (but CLIP only returns 1 token for `[B, 1, 512]`).

**Frozen:** DistilBERT
**Trainable:** Linear projection, positional encoding, 2-layer transformer

---

## Loss Functions

**File:** `loss_finegrained.py`

### Total Loss

```
L_total = L_align + λ_recon * L_recon

L_align = (L_st + L_sm + L_tm) / 3
  where:
    L_st = alignment_loss(speech_tokens, text_tokens)
    L_sm = alignment_loss(speech_tokens, motion_tokens)
    L_tm = alignment_loss(text_tokens, motion_tokens)

λ_recon = 0.1
```

### Loss 1: Sequence-Level Alignment Loss (Paper Eq. 5-8)

Each `alignment_loss(x, y)` consists of two steps:

#### Step A — Sequence-Level Similarity (Eq. 8)

Computes fine-grained similarity between two token sequences:

```
h(e_x, e_y) = 0.5 * Σ_i  w_x^i * max_j <e_x^i, e_y^j>     (x→y direction)
            + 0.5 * Σ_j  w_y^j * max_i <e_y^j, e_x^i>       (y→x direction)
```

Detailed computation:

```
Given: tokens_x [B, L_x, C], tokens_y [B, L_y, C]  (both L2-normalized)

1. Token weights (learned):
   w_x = softmax( Linear(tokens_x) )    → [B, L_x]   per-token importance
   w_y = softmax( Linear(tokens_y) )    → [B, L_y]

2. Pairwise token similarity:
   token_sim[b1, b2, i, j] = <tokens_x[b1, i], tokens_y[b2, j]>
   → [B, B, L_x, L_y]

3. Max-over-tokens:
   x→y: max_sim_xy[b1, b2, i] = max_j  token_sim[b1, b2, i, j]
        "for each x-token, find its best-matching y-token"
   y→x: max_sim_yx[b1, b2, j] = max_i  token_sim[b1, b2, i, j]
        "for each y-token, find its best-matching x-token"

4. Weighted sum → [B, B] similarity matrix:
   sim[b1, b2] = 0.5 * Σ_i w_x[b1,i] * max_sim_xy[b1,b2,i]
               + 0.5 * Σ_j w_y[b2,j] * max_sim_yx[b1,b2,j]
```

**Intuition:** For the text "a person walks forward then turns around" paired with a 100-frame motion:
- The word "turns" will have high max-similarity with frames 60-70 (the turning frames)
- If the weight head learns that "turns" is important (high w), this match dominates the score
- Global pooling would average "turns" with filler words like "a", "person", losing precision

#### Step B — Bidirectional KL Divergence (Eq. 5)

```
S_target = I   (identity matrix: i-th query should match i-th sample)

S_pred_x2y = softmax( sim_matrix / τ )       row-wise softmax
S_pred_y2x = softmax( sim_matrix.T / τ )

L_align^xy = KL(S_target || S_pred_x2y) + KL(S_target || S_pred_y2x)
```

τ (temperature) is a learnable parameter initialized to `1/0.07 ≈ 14.3`.

### Loss 2: Reconstruction Loss (Paper Eq. 10-11)

Auxiliary task: randomly mask motion tokens, use text + audio to reconstruct them.

```
1. Randomly mask 50% of valid motion tokens → replace with learnable [MASK] token
2. Concatenate:  [masked_motion + type_embed_0,  text + type_embed_1,  audio + type_embed_2]
   → [B, L_m + L_t + L_a, 512]
3. TransformerEncoder (2 layers) → extract first L_m positions → Linear
4. L_recon = MSE( reconstructed[masked_positions], original[masked_positions] )
```

**Intuition:** If "turns around" motion tokens are masked, the model must use:
- Text tokens containing "turns around"
- Audio tokens of the corresponding speech
- Remaining unmasked motion tokens (walking part)
to predict what the turning tokens should look like. This forces real cross-modal semantic alignment.

### Loss Summary Table

| Loss | Formula | Weight | Paper Reference |
|------|---------|--------|-----------------|
| `L_st` | KL(softmax(h(speech,text)/τ), I) bidirectional | 1/3 | Eq. 4-5 |
| `L_sm` | KL(softmax(h(speech,motion)/τ), I) bidirectional | 1/3 | Eq. 4-5 |
| `L_tm` | KL(softmax(h(text,motion)/τ), I) bidirectional | 1/3 | Eq. 4-5 |
| `L_recon` | MSE on masked motion tokens | 0.1 | Eq. 11-12 |
| **Total** | **(L_st + L_sm + L_tm)/3 + 0.1 * L_recon** | | Eq. 12 |

---

## Comparison: Old Model vs New Model

| Aspect | Old (`model.py`) | New (`model_finegrained.py`) |
|--------|------------------|------------------------------|
| Motion encoder | VAE → Proj → MeanPool → **1 vector** | VAE → Linear+TF → **T' tokens** |
| Audio encoder | HuBERT → Proj → MeanPool → **1 vector** | WavLM → MemoryRetrieval → **32 tokens** |
| Text encoder | CLIP(CLS) → Proj → MeanPool → **1 vector** | DistilBERT(all tokens) → TF → **L_t tokens** |
| Similarity | `cosine(vec_a, vec_b)` scalar | `h(tokens_a, tokens_b)` via max+weights |
| Contrastive loss | Symmetric InfoNCE (cross-entropy) | Bidirectional KL divergence |
| Auxiliary loss | None | Masked motion reconstruction (0.1 weight) |
| Temperature | Learnable logit_scale | Learnable logit_scale |
| Validation metric | R@K via global cosine | R@K via weighted-pool to global embedding |

---

## Trainable vs Frozen Parameters

| Component | Frozen | Trainable |
|-----------|--------|-----------|
| RVQVae encoder | All params | — |
| WavLM | All params | — |
| DistilBERT | All params | — |
| Motion: Linear + PosEnc + Transformer(4L) | — | All |
| Audio: Linear + 128 memory tokens + CrossAttn(2L) + PosEnc | — | All |
| Text: Linear + PosEnc + Transformer(2L) | — | All |
| Token weight heads: 3 × Linear(512,1) | — | All |
| Reconstruction decoder: Embedding(3,512) + Transformer(2L) + Linear | — | All |
| logit_scale, mask_token | — | All |

---

## Validation / Retrieval

During validation, token sequences need to be collapsed to global embeddings for R@K computation (since retrieval is done via single-vector similarity):

```
For each modality:
  1. Collect all token sequences [N, L, 512] from val set
  2. Compute weights: w = softmax(Linear(tokens))    using the same weight_head
  3. Weighted pool: emb = Σ w_i * token_i            → [N, 512]
  4. L2 normalize
```

Then compute R@1, R@5, R@10 for all 6 directions (S→T, T→S, S→M, M→S, T→M, M→T).

---

## Config File Reference

`contrastive/configs/contrastive_finegrained.yaml`:

```yaml
model:
  speech_encoder_type: wavlm-large    # Frozen audio backbone
  text_encoder_type: distilbert       # Frozen text backbone (or 'clip')
  latent_dim: 512                     # Shared embedding dimension

  motion:
    num_layers: 4                     # Transformer layers on top of VAE
    nhead: 8

  audio:
    num_memory_tokens: 128            # Learnable memory for cross-attention
    num_attn_layers: 2
    target_length: 32                 # Fixed output length after pooling

  text:
    transformer_layers: 2             # Transformer layers on top of DistilBERT
    max_text_len: 77

  loss:
    temperature_init: 0.07
    lambda_recon: 0.1                 # Reconstruction loss weight
    mask_ratio: 0.5                   # Fraction of motion tokens to mask

  reconstruction:
    num_layers: 2
    nhead: 8

vae:                                  # Same VAE as original contrastive
  params: { nfeats: 133, num_quantizers: 3, code_dim: 512, ... }
  checkpoint: "path/to/vae/best.ckpt"

training:
  batch_size: 32
  max_epochs: 200
  lr: 1e-4
  weight_decay: 0.01
```

---

## Checkpoints Saved

Saved to `experiments/contrastive_fg/checkpoints/` when `val/avg_R@1` improves:

```
motion_encoder_best.pt     — VaeMotionEncoder state_dict
audio_encoder_best.pt      — MemoryRetrievalAudioEncoder state_dict
text_encoder_best.pt       — SequenceTextEncoder state_dict
weight_heads_best.pt       — 3 token weight heads
recon_decoder_best.pt      — ReconstructionDecoder state_dict
params_best.pt             — logit_scale + mask_token
fg-contrastive-{epoch}.ckpt — Full Lightning checkpoint (top 3 by val/avg_R@1)
```
