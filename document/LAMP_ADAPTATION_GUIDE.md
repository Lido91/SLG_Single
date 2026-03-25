# LaMP Architecture Adaptation for MotionGPT

**Date:** 2026-02-10
**Author:** Claude (Anthropic)
**Project:** Sign Language Generation with LaMP + Masked Transformer T2M

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Summary](#architecture-summary)
3. [Files Created](#files-created)
4. [Files Modified](#files-modified)
5. [Training Pipeline](#training-pipeline)
6. [Configuration Details](#configuration-details)
7. [Usage Instructions](#usage-instructions)
8. [Implementation Details](#implementation-details)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This document describes the complete adaptation of the **LaMP (Language-Motion Pretraining)** architecture from the original LaMP codebase to the MotionGPT framework for sign language generation on the How2Sign dataset.

### What Was Adapted?

**Full LaMP + T2M Pipeline:**
1. **Stage 1 (Existing):** VQ-VAE Motion Tokenizer (RVQ-VAE with 6 quantizers)
2. **Stage 2 (NEW):** LaMP Contrastive Pretraining (QFormer-based text-motion alignment)
3. **Stage 3 (NEW):** Masked Transformer T2M Generation (Non-autoregressive generation)

### Key Differences from Original LaMP

| Aspect | Original LaMP | Adapted Version |
|--------|--------------|-----------------|
| **Motion Encoder** | Custom VQ-VAE encoder | MotionGPT's RVQ-VAE encoder (6 quantizers) |
| **Dataset** | HumanML3D (263D body-only) | How2Sign (133D full-body + hands + face) |
| **Data Format** | Motion features | Sign language poses (36D joint + shape params) |
| **Framework** | Standalone training scripts | PyTorch Lightning (MotionGPT pipeline) |
| **Config System** | Argparse options | OmegaConf YAML configs |
| **Generation** | Uses all quantizers | Can use Q0 only (coarse) or all 6 |

---

## Architecture Summary

### Stage 1: VQ-VAE (Existing)

**File:** `mGPT/archs/mgpt_rvq.py`

```
Motion Features [B, T, 133]
    ↓ Encoder (Conv1D + ResNet)
Continuous Embeddings [B, code_dim, T']
    ↓ Residual VQ (6 quantizers)
Discrete Tokens [B, T', 6]
    ↓ Decoder (Transpose Conv1D)
Reconstructed Motion [B, T, 133]
```

**Key Methods:**
- `forward(features)` → reconstruction, loss, perplexity
- `encode(features)` → discrete codes [B, T', 6]
- `encode_continuous(features)` → continuous embeddings [B, code_dim] (for LaMP)
- `decode(code_idx)` → reconstructed motion
- `decode_q0_only(q0_idx)` → decode from Q0 only
- `decode_partial(code_idx)` → decode from subset of quantizers

---

### Stage 2: LaMP Pretraining (NEW)

**File:** `mGPT/archs/lamp/lamp_model.py`

```
Motion [B, T, 133]                    Text (List[str])
    ↓ RVQ-VAE Encoder (frozen)           ↓ BERT Tokenizer
Motion Embeddings [B, T', code_dim]   Text Tokens [B, L]
    ↓ Project (512 → 1408)               ↓ BERT Encoding [B, L, 768]
Motion Features [B, T', 1408]         Text Embeddings [B, L, 768]
    ↓                                    ↓
    ┌────────────────────────────────────┐
    │         QFormer (Cross-Attention)  │
    │  Query Tokens [32, 768] (learnable)│
    └────────────────────────────────────┘
           ↓                        ↓
    Motion Queries [B, 32, 768]   Text Features [B, 768]
           ↓                        ↓
    ┌─────────────────────────────────────────┐
    │         4 Training Objectives:          │
    │  1. PTC: Point-Text Contrastive         │
    │  2. PTM: Point-Text Matching            │
    │  3. LM: Language Modeling (M2T)         │
    │  4. GEN: Generation (T2M token predict) │
    └─────────────────────────────────────────┘
```

**4 Training Objectives:**

1. **PTC (Point-Text Contrastive):**
   - Contrastive learning between motion and text
   - InfoNCE loss with temperature scaling
   - Aligns motion queries with text CLS token
   - Loss: `(loss_p2t + loss_t2p) / 2`

2. **PTM (Point-Text Matching):**
   - Binary classification: positive vs negative pairs
   - Hard negative mining based on similarity
   - QFormer processes motion + text jointly
   - Loss: `CrossEntropy(pos=1, neg=0)`

3. **LM (Language Modeling - M2T):**
   - Motion-to-text captioning
   - Uses motion query outputs as prefix
   - Autoregressive text generation
   - Loss: `CrossEntropy(predicted_text, target_text)`

4. **GEN (Generation - T2M):**
   - Text-to-motion token prediction
   - Predicts Q0 tokens (coarse codes) from text
   - Cross-entropy on codebook vocabulary
   - Loss: `CrossEntropy(predicted_tokens, vq_tokens)`

**Total Loss:**
```python
total_loss = loss_ptc + loss_ptm + loss_lm + loss_gen
```

---

### Stage 3: Masked Transformer T2M (NEW)

**File:** `mGPT/archs/mgpt_masked_t2m.py`

```
Text (List[str])
    ↓ QFormer (frozen, from LaMP)
Text Features [B, 768]
    ↓ Linear Projection
Conditioning [B, latent_dim]
    ↓
┌──────────────────────────────────────┐
│   Masked Transformer Generation      │
│                                      │
│  Start: All tokens = [MASK]         │
│    ↓ Iterative Refinement (18 steps)│
│  Cosine Masking Schedule             │
│    ↓ Transformer Encoder             │
│  Predict Masked Tokens               │
│    ↓ Sample from Logits              │
│  Update Tokens (keep high confidence)│
│    ↓ Repeat with fewer masks         │
│  Final: All tokens predicted         │
└──────────────────────────────────────┘
    ↓
Motion Tokens [B, T]
    ↓ VQ-VAE Decode
Motion Features [B, T, 133]
```

**Key Features:**

1. **Masked Prediction:**
   - BERT-style masking during training
   - 88% mask token, 10% random, 2% correct
   - Loss: Cross-entropy on masked positions only

2. **Iterative Refinement:**
   - 18 timesteps from fully masked to fully predicted
   - Cosine schedule: `cos(t * π/2)` determines mask ratio
   - Each step: predict → keep high-confidence → re-mask low-confidence

3. **Classifier-Free Guidance:**
   - Two forward passes: conditional and unconditional
   - Scaled logits: `logits_uncond + scale * (logits_cond - logits_uncond)`
   - Default scale: 3

4. **Non-Autoregressive:**
   - Predicts all tokens in parallel (vs. GPT's sequential)
   - Faster generation (18 steps vs. 100+ for GPT)
   - Quality comparable to autoregressive models

---

## Files Created

### 1. LaMP Architecture Files

#### `mGPT/archs/lamp/__init__.py`
- Package initialization
- Exports `LaMP` class

#### `mGPT/archs/lamp/qformer_base.py`
- Base class for QFormer-based models
- Utilities for initializing QFormer, tokenizer, motion encoder
- Optimizer parameter groups with layer-wise LR decay

**Key Methods:**
```python
QFormer_Base.init_tokenizer() → BertTokenizer
QFormer_Base.init_Qformer(num_query_token, vision_width, cross_attention_freq) → (Qformer, query_tokens)
QFormer_Base.init_motion_encoder_from_vae(vae_encoder) → motion_encoder
```

#### `mGPT/archs/lamp/lamp_model.py`
- Main LaMP model adapted for MotionGPT
- Uses frozen RVQ-VAE encoder for motion features
- 4 training objectives (PTC, PTM, LM, GEN)

**Constructor Signature:**
```python
LaMP(
    vq_model,                    # Pretrained RVQ-VAE (frozen)
    nfeats=133,                  # How2Sign features
    num_query_token=32,          # Learnable queries
    cross_attention_freq=2,      # Cross-attn every 2 blocks
    embed_dim=512,               # Contrastive embedding dim
    max_txt_len=32,              # Max text length
    motion_encoder_dim=512,      # RVQ encoder output dim
    num_tokens=512,              # Codebook size
)
```

**Key Methods:**
```python
forward(motion, text) → (QFormer_Output, text_feat, motion_feat)
encode_text(text) → text_features [B, embed_dim]
encode_motion(motion) → motion_features [B, embed_dim]
```

#### `mGPT/archs/lamp/basemodel.py` (Copied)
- Utilities for distributed training
- `all_gather_with_grad()`, `concat_all_gather()`

#### `mGPT/archs/lamp/QFormer.py` (Copied)
- BERT-based QFormer implementation
- Cross-attention layers for vision-language fusion

#### `mGPT/archs/lamp/QFormer_output.py` (Copied)
- Output dataclass for QFormer
- `QFormer_Output(loss, loss_ptc, loss_ptm, loss_lm, loss_gen)`

---

### 2. Masked Transformer T2M

#### `mGPT/archs/mgpt_masked_t2m.py`
- Non-autoregressive masked transformer
- Uses pretrained QFormer from LaMP (frozen)
- Iterative refinement with classifier-free guidance

**Constructor Signature:**
```python
MaskedTransformerT2M(
    num_tokens=512,              # Codebook size
    code_dim=512,                # Token embedding dim
    latent_dim=256,              # Transformer hidden dim
    ff_size=1024,                # Feedforward dim
    num_layers=8,                # Transformer layers
    num_heads=4,                 # Attention heads
    dropout=0.1,                 # Dropout rate
    cond_drop_prob=0.1,          # CFG dropout
    qformer_hidden_size=768,     # QFormer output dim
    pretrained_qformer_path=None,# LaMP checkpoint path
)
```

**Key Methods:**
```python
forward(ids, texts, m_lens) → (ce_loss, pred_id, acc)
generate(texts, m_lens, timesteps=18, cond_scale=3, ...) → ids [B, T]
generate_conditional(texts, lengths, ...) → List[Tensor]  # MotionGPT interface
encode_text_qformer(texts) → text_features [B, 768]
```

**Helper Functions:**
- `cosine_schedule(t)` → masking schedule
- `top_k(logits, thres)` → nucleus filtering
- `gumbel_sample(logits, temp)` → Gumbel-softmax sampling
- `cal_performance(logits, labels)` → loss + accuracy

---

### 3. Configuration Files

#### `configs/lamp_h2s.yaml`
- Configuration for LaMP pretraining stage
- Batch size: 32
- Training epochs: 100
- 4 loss weights (all 1.0)
- No metrics (contrastive learning only)

**Key Settings:**
```yaml
TRAIN:
  STAGE: lamp
  BATCH_SIZE: 32
  END_EPOCH: 100

LOSS:
  LAMBDA_PTC: 1.0
  LAMBDA_PTM: 1.0
  LAMBDA_LM: 1.0
  LAMBDA_GEN: 1.0
```

#### `configs/masked_t2m_h2s.yaml`
- Configuration for Masked Transformer T2M stage
- Batch size: 32
- Training epochs: 150
- Loads pretrained QFormer from LaMP

**Key Settings:**
```yaml
TRAIN:
  STAGE: lm_masked_t2m
  BATCH_SIZE: 32
  END_EPOCH: 150

lm:
  masked_t2m:
    pretrained_qformer_path: "experiments/mgpt/LaMP_H2S_Pretrain/checkpoints/last.ckpt"
```

---

## Files Modified

### 1. `mGPT/losses/mgpt.py`

**Changes:**
- Added loss handling for `stage == "lamp"`
- Added loss handling for `stage == "lm_masked_t2m"`
- Updated `__init__()` to register new losses
- Updated `update()` to compute new losses

**New Loss Registrations:**
```python
elif stage == "lamp":
    losses = ["lamp_ptc", "lamp_ptm", "lamp_lm", "lamp_gen"]
elif stage == "lm_masked_t2m":
    losses = ["masked_ce", "masked_acc"]
```

**Loss Update Logic:**
```python
if self.stage == "lamp":
    total += self._update_loss("lamp_ptc", rs_set['outputs'].loss_ptc, ...)
    total += self._update_loss("lamp_ptm", rs_set['outputs'].loss_ptm, ...)
    total += self._update_loss("lamp_lm", rs_set['outputs'].loss_lm, ...)
    total += self._update_loss("lamp_gen", rs_set['outputs'].loss_gen, ...)

if self.stage == "lm_masked_t2m":
    total += self._update_loss("masked_ce", rs_set['ce_loss'], ...)
    self._update_loss("masked_acc", rs_set['acc'], ...)
```

---

### 2. `mGPT/models/mgpt.py`

**New Methods Added:**

#### `train_lamp_forward(batch)`
```python
def train_lamp_forward(self, batch):
    feats_ref = batch["motion"]  # Raw motion features
    texts = batch["text"]
    outputs, text_feat, motion_feat = self.lm(feats_ref, texts)
    return {'outputs': outputs, 'text_feat': text_feat, 'motion_feat': motion_feat}
```

#### `train_masked_t2m_forward(batch)`
```python
def train_masked_t2m_forward(self, batch):
    tokens_ref = batch["motion"]  # Motion tokens
    texts = batch["text"]
    lengths = batch["length"]

    # Convert frame lengths to token lengths
    token_lengths = [l // 4 for l in lengths]

    # Use first quantizer only (Q0)
    if tokens_ref.dim() == 3:
        tokens_ref = tokens_ref[:, :, 0]

    ce_loss, pred_id, acc = self.lm(tokens_ref, texts, token_lengths)
    return {'ce_loss': ce_loss, 'pred_id': pred_id, 'acc': acc}
```

#### `val_masked_t2m_forward(batch)`
```python
@torch.no_grad()
def val_masked_t2m_forward(self, batch):
    # Generate tokens using masked transformer
    outputs_tokens = self.lm.generate_conditional(texts, lengths, stage='test')

    # Decode tokens to motion (Q0 only)
    for i in range(len(outputs_tokens)):
        motion = self.vae.decode_q0_only(outputs_tokens[i])
        feats_rst[i, :actual_len, :] = motion[0, :actual_len, :]

    # Return features, joints, vertices for metrics
    return {
        "m_ref": feats_ref,
        "m_rst": feats_rst,
        "joints_ref": joints_ref,
        "joints_rst": joints_rst,
        "vertices_ref": vertices_ref,
        "vertices_rst": vertices_rst,
        "lengths_rst": lengths_rst
    }
```

**Modified `allsplit_step()` method:**

Added routing for new stages:
```python
elif self.hparams.stage == "lamp" and split in ["train"]:
    rs_set = self.train_lamp_forward(batch)
    loss = self._losses['losses_' + split].update(rs_set)

elif self.hparams.stage == "lm_masked_t2m" and split in ["train"]:
    rs_set = self.train_masked_t2m_forward(batch)
    loss = self._losses['losses_' + split].update(rs_set)
```

Added validation handling:
```python
elif self.hparams.stage == "lm_masked_t2m":
    rs_set = self.val_masked_t2m_forward(batch)
    # Update TM2TMetrics
    getattr(self.metrics, metric_name).update(...)
```

---

## Training Pipeline

### Complete 3-Stage Training Sequence

```bash
# Stage 1: Train VQ-VAE (if not already trained)
python train.py --cfg configs/deto_h2s_rvq_3.yaml

# Stage 2: LaMP Pretraining
python train.py --cfg configs/lamp_h2s.yaml \
    --pretrained_vae experiments/mgpt/DETO_RVQ_wholebody_3/checkpoints/min-how2sign_MPJPE_PA_handepoch=489.ckpt

# Stage 3: Masked T2M Generation
python train.py --cfg configs/masked_t2m_h2s.yaml \
    --pretrained_vae experiments/mgpt/DETO_RVQ_wholebody_3/checkpoints/min-how2sign_MPJPE_PA_handepoch=489.ckpt
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Stage 1: VQ-VAE Training                   │
│                                                                 │
│  Motion [B,T,133] → Encoder → RVQ(6) → Decoder → Recons        │
│  Loss: Reconstruction + Velocity + Commitment                  │
└─────────────────────────────────────────────────────────────────┘
                               ↓ (save checkpoint)
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: LaMP Pretraining                    │
│                                                                 │
│  Motion → RVQ Encoder (frozen) → Embeddings                     │
│  Text → BERT Tokenizer → Text Embeddings                       │
│         ↓                                                       │
│      QFormer (cross-attention)                                  │
│         ↓                                                       │
│  4 Losses: PTC + PTM + LM + GEN                                 │
└─────────────────────────────────────────────────────────────────┘
                               ↓ (save checkpoint with QFormer)
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 3: Masked T2M Generation                  │
│                                                                 │
│  Text → QFormer (frozen, from LaMP) → Text Features             │
│         ↓                                                       │
│  Masked Transformer (learnable)                                 │
│         ↓                                                       │
│  Motion Tokens → VQ Decode → Motion                             │
│  Loss: Masked Cross-Entropy                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Details

### LaMP Config (`lamp_h2s.yaml`)

**Model Instantiation:**
```yaml
model:
  target: mGPT.models.mgpt.MotionGPT
  params:
    stage: lamp
    lm:
      target: mGPT.archs.lamp.lamp_model.LaMP
      params:
        nfeats: 133
        num_query_token: 32
        embed_dim: 512
        num_tokens: 512
    motion_vae:
      target: mGPT.archs.mgpt_rvq.RVQVae
      params:
        num_quantizers: 6
        code_num: 512
        nfeats: 133
```

**Loss Weights:**
```yaml
LOSS:
  LAMBDA_PTC: 1.0   # Contrastive
  LAMBDA_PTM: 1.0   # Matching
  LAMBDA_LM: 1.0    # M2T captioning
  LAMBDA_GEN: 1.0   # T2M token prediction
```

**Training Settings:**
```yaml
TRAIN:
  STAGE: lamp
  BATCH_SIZE: 32
  END_EPOCH: 100
  OPTIM:
    lr: 1e-4
    weight_decay: 0.01
```

---

### Masked T2M Config (`masked_t2m_h2s.yaml`)

**Model Instantiation:**
```yaml
model:
  target: mGPT.models.mgpt.MotionGPT
  params:
    stage: lm_masked_t2m
    lm:
      target: mGPT.archs.mgpt_masked_t2m.MaskedTransformerT2M
      params:
        num_tokens: 512
        code_dim: 512
        latent_dim: 256
        ff_size: 1024
        num_layers: 8
        num_heads: 4
        dropout: 0.1
        cond_drop_prob: 0.1  # CFG
        pretrained_qformer_path: "path/to/lamp/checkpoint.ckpt"
```

**Generation Settings:**
```yaml
# In code (not config):
generate(
    texts=texts,
    m_lens=token_lengths,
    timesteps=18,           # Iterative refinement steps
    cond_scale=3,          # Classifier-free guidance scale
    temperature=1.0,       # Sampling temperature
    topk_filter_thres=0.9, # Nucleus sampling threshold
    gsample=False          # Use multinomial (not Gumbel)
)
```

**Training Settings:**
```yaml
TRAIN:
  STAGE: lm_masked_t2m
  BATCH_SIZE: 32
  END_EPOCH: 150
  OPTIM:
    lr: 1e-4
    weight_decay: 0.01

METRIC:
  TYPE: ["TM2TMetrics"]  # Text-to-motion evaluation
```

---

## Usage Instructions

### 1. Prerequisites

**Install Dependencies:**
```bash
# QFormer requires transformers
pip install transformers

# Ensure you have the MotionGPT dependencies
pip install -r requirements.txt
```

**Prepare Data:**
- How2Sign dataset in `/data/hwu/slg_data/How2Sign`
- Normalized mean/std files
- Pre-tokenized motion codes (for Masked T2M stage)

---

### 2. Training LaMP (Stage 2)

```bash
# Train LaMP contrastive pretraining
python train.py \
    --cfg configs/lamp_h2s.yaml \
    --batch_size 32 \
    --device 0 1 2

# Monitor training
wandb login
# Check project "SLG" for "LaMP_H2S_Pretrain"
```

**Expected Training Time:**
- 100 epochs × ~2000 batches/epoch = ~200k iterations
- ~3-4 days on 3x RTX 3090 GPUs

**What to Monitor:**
- `lamp/ptc/train` - Should decrease to ~0.5-1.0
- `lamp/ptm/train` - Should decrease to ~0.3-0.5
- `lamp/lm/train` - Should decrease to ~2.0-3.0
- `lamp/gen/train` - Should decrease to ~3.0-4.0
- `total/train` - Sum of above

**Checkpoints:**
- Saved in `experiments/mgpt/LaMP_H2S_Pretrain/checkpoints/`
- Use `last.ckpt` for Stage 3

---

### 3. Training Masked T2M (Stage 3)

```bash
# Train Masked Transformer T2M
python train.py \
    --cfg configs/masked_t2m_h2s.yaml \
    --batch_size 32 \
    --device 0 1 2

# Ensure LaMP checkpoint path is correct in config:
# pretrained_qformer_path: "experiments/mgpt/LaMP_H2S_Pretrain/checkpoints/last.ckpt"
```

**Expected Training Time:**
- 150 epochs × ~2000 batches/epoch = ~300k iterations
- ~5-6 days on 3x RTX 3090 GPUs

**What to Monitor:**
- `masked/ce/train` - Should decrease to ~1.5-2.5
- `masked/acc/train` - Should increase to ~0.6-0.7
- Validation metrics (every 10 epochs):
  - `Metrics/how2sign_DTW_MPJPE_PA_lhand` (lower is better)
  - `Metrics/how2sign_DTW_MPJPE_PA_rhand` (lower is better)

---

### 4. Testing / Inference

```bash
# Test Masked T2M model
python test.py \
    --cfg configs/masked_t2m_h2s.yaml \
    --batch_size 16

# Generate from custom text
python demo.py \
    --cfg configs/masked_t2m_h2s.yaml \
    --example "path/to/text_prompts.txt" \
    --out_dir "outputs/masked_t2m/"
```

**Generation Parameters (tunable in code):**
```python
# In mGPT/archs/mgpt_masked_t2m.py → generate()
timesteps=18          # More steps = higher quality, slower
cond_scale=3          # Higher = more aligned with text
temperature=1.0       # Higher = more diverse
topk_filter_thres=0.9 # Higher = more diverse
```

---

## Implementation Details

### LaMP Model Architecture

**Text Encoding Pipeline:**
```python
text → BertTokenizer → [B, L] token IDs
    ↓ Qformer.bert()
text_output → [B, L, 768]
    ↓ CLS token [B, 768]
    ↓ text_proj (Linear)
text_feat → [B, 512] normalized
```

**Motion Encoding Pipeline:**
```python
motion [B, T, 133] → permute → [B, 133, T]
    ↓ RVQ Encoder (frozen)
motion_embeds → [B, 512, T']
    ↓ permute → [B, T', 512]
    ↓ motion_projection (512→1408)
motion_embeds → [B, T', 1408]
    ↓ QFormer cross-attention with queries
query_output → [B, 32, 768]
    ↓ motion_proj (Linear)
motion_feats → [B, 32, 512] normalized
    ↓ mean pooling
motion_feats_pooled → [B, 512] normalized
```

**Loss Computation:**

1. **PTC Loss:**
```python
# Motion → Text similarity
sim_p2t = temp * (motion_feats @ text_feat_all.T).max(dim=-1)[0]  # [B, B_all]
loss_p2t = CrossEntropy(sim_p2t, targets)

# Text → Motion similarity
sim_t2p = temp * (text_feat @ motion_feats_all.permute(0,2,1)).max(dim=-1)[0]
loss_t2p = CrossEntropy(sim_t2p, targets)

loss_ptc = (loss_p2t + loss_t2p) / 2
```

2. **PTM Loss:**
```python
# Sample hard negatives based on similarity
neg_motion = sample_hard_negative(weights_t2p)
neg_text = sample_hard_negative(weights_p2t)

# Create triplets: (pos, pos), (neg, pos), (pos, neg)
triplets = concat([motion, neg_motion, motion], dim=0)
text_triplets = concat([text, text, neg_text], dim=0)

# QFormer processes triplets
output_ptm = Qformer(text_triplets, encoder_hidden_states=triplets)
logits = itm_head(output_ptm.mean(dim=1))  # [3B, 2]

# Labels: first B positive, next 2B negative
labels = concat([ones(B), zeros(2B)], dim=0)
loss_ptm = CrossEntropy(logits, labels)
```

3. **LM Loss (M2T):**
```python
# Motion queries as prefix for text generation
lm_output = Qformer(
    decoder_input_ids=text_tokens,
    past_key_values=query_output.past_key_values,  # Motion context
    labels=text_labels
)
loss_lm = lm_output.loss
```

4. **GEN Loss (T2M):**
```python
# Text → motion token prediction
text_embeds = Qformer.bert.embeddings(text_tokens)
text_embeds = text_embeds @ text_projection  # Project to motion space

text_query_output = Qformer(
    query_embeds=query_tokens,
    encoder_hidden_states=text_embeds
)

prediction = motion_cls(text_query_output)  # [B, 32, 512]

# Get VQ targets (Q0 only)
motion_targets = vq_model.encode(motion)[:, :, 0]  # [B, T']

# Match lengths and compute loss
loss_gen = CrossEntropy(prediction, motion_targets)
```

---

### Masked T2M Model Architecture

**Text Encoding:**
```python
text → BertTokenizer → [B, L] token IDs
    ↓ Qformer.bert (frozen)
text_output → [B, L, 768]
    ↓ text2former (Linear 768→1408)
text_embeds → [B, L, 1408]
    ↓ Qformer cross-attention (frozen)
query_output → [B, 32, 768]
    ↓ mean pooling
text_features → [B, 768]
```

**Token Embedding:**
```python
motion_ids [B, T] → token_emb(ids)
    ↓ [B, T, code_dim=512]
    ↓ input_process (Linear 512→256)
    ↓ permute to [T, B, 256]
    ↓ position_enc (sinusoidal)
x → [T, B, latent_dim=256]
```

**Conditioning:**
```python
text_features [B, 768] → cond_emb (Linear 768→256)
    ↓ unsqueeze(0)
cond → [1, B, 256]

# Prepend to sequence
xseq = concat([cond, x], dim=0)  # [T+1, B, 256]
```

**Transformer Forward:**
```python
xseq [T+1, B, 256] → TransformerEncoder(num_layers=8)
    ↓ causal mask (autoregressive)
output → [T+1, B, 256]
    ↓ drop conditioning [1:]
output → [T, B, 256]
    ↓ output_process (BERT-style: Dense→GELU→LN→Linear)
logits → [B, 512, T]
```

**Training - Random Masking:**
```python
# Sample random masking ratio from cosine schedule
rand_time ~ Uniform(0, 1)
rand_mask_prob = cos(rand_time * π/2)

# Randomly select positions to mask
mask = random_select(num_tokens * rand_mask_prob)

# BERT masking: 88% [MASK], 10% random, 2% correct
x_ids_corrupted = apply_bert_masking(x_ids, mask)

# Predict original tokens
logits = transformer_forward(x_ids_corrupted, text_features)
loss = CrossEntropy(logits[mask], x_ids[mask])
```

**Inference - Iterative Refinement:**
```python
# Start from all masked
ids = [MASK] * T
scores = [0] * T

for timestep in linspace(0, 1, 18):
    # Cosine schedule determines how many to mask
    rand_mask_prob = cos(timestep * π/2)
    num_to_mask = int(T * rand_mask_prob)

    # Mask lowest-confidence tokens
    ids[lowest_k_scores(num_to_mask)] = [MASK]

    # Predict masked tokens
    logits = transformer_forward(ids, text_features, cond_scale=3)

    # Sample predictions
    pred_ids = sample(logits / temperature)

    # Update ids and scores
    ids[mask] = pred_ids[mask]
    scores = confidence(logits, pred_ids)
```

**Classifier-Free Guidance:**
```python
# Two forward passes
logits_cond = transformer_forward(ids, text_features, force_mask=False)
logits_uncond = transformer_forward(ids, zeros_like(text_features), force_mask=True)

# Scale towards conditional
logits_scaled = logits_uncond + cond_scale * (logits_cond - logits_uncond)
```

---

## Troubleshooting

### Common Issues

#### 1. **LaMP Training - Loss NaN**

**Symptoms:**
- `lamp/ptc/train` becomes NaN
- Temperature parameter explodes

**Solutions:**
```python
# In lamp_model.py, check temperature initialization
self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

# Clip temperature during forward
temp = torch.clamp(self.temp.exp(), min=0.01, max=100)
```

#### 2. **Masked T2M - Poor Generation Quality**

**Symptoms:**
- Generated motions are static or repetitive
- DTW MPJPE very high (>100mm)

**Solutions:**
- Increase `timesteps` from 18 to 36
- Increase `cond_scale` from 3 to 5
- Check if QFormer loaded correctly (should see "QFormer loaded successfully")
- Verify token embeddings initialized from VQ codebook

#### 3. **QFormer Not Loading from LaMP Checkpoint**

**Symptoms:**
```
KeyError: "Cannot find QFormer weights in checkpoint"
```

**Solutions:**
```python
# Check checkpoint structure
ckpt = torch.load("path/to/lamp.ckpt")
print(ckpt.keys())  # Should have 'Qformer' or 'model'

# If weights are nested:
if 'model' in ckpt:
    qformer_weights = {k.replace("lm.Qformer.", ""): v
                      for k, v in ckpt['model'].items() if 'Qformer' in k}
```

#### 4. **CUDA Out of Memory**

**Symptoms:**
- OOM during LaMP training
- Large batch contrastive learning

**Solutions:**
```yaml
# Reduce batch size in config
TRAIN:
  BATCH_SIZE: 16  # Down from 32

# Enable gradient checkpointing (if implemented)
lm:
  lamp:
    params:
      use_gradient_checkpointing: true
```

#### 5. **Metrics Not Updating**

**Symptoms:**
- Validation runs but no metrics logged
- `Metrics/how2sign_DTW_MPJPE_PA_lhand` missing

**Solutions:**
```python
# Check metric instantiation in mgpt.py
print(f"Available metrics: {dir(self.metrics)}")

# Ensure metric type matches config
METRIC:
  TYPE: ["TM2TMetrics"]  # Not MRMetrics for T2M

# Check feats2joints returns vertices
result = self.feats2joints(feats_ref)
if isinstance(result, tuple):
    vertices_ref, joints_ref = result
else:
    # Need to handle vertices separately
```

#### 6. **Token Length Mismatch**

**Symptoms:**
```
RuntimeError: Expected tensor of size [B, 100] but got [B, 25]
```

**Solutions:**
```python
# Check UNIT_LEN in config
DATASET:
  H2S:
    UNIT_LEN: 4  # This determines token length: T_frames / 4 = T_tokens

# Verify downsampling matches VAE
# RVQ-VAE with down_t=2, stride_t=2 → downsample by 4

# In masked T2M forward:
token_lengths = [l // 4 for l in lengths]  # Frame → token conversion
```

---

### Debugging Tips

**Enable Debug Mode:**
```yaml
# In config
DEBUG: true

# This enables:
# - Offline wandb logging
# - VAL_EVERY_STEPS: 1
# - Name prefix: debug--
```

**Print Model Architecture:**
```python
# In train.py after model instantiation
print(model)
print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
```

**Visualize Attention:**
```python
# In lamp_model.py forward()
with torch.no_grad():
    # Save QFormer attention maps
    attentions = query_output.attentions
    torch.save(attentions, f"debug/attentions_step{step}.pt")
```

**Check Gradient Flow:**
```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT")
```

---

## Performance Benchmarks

### Expected Results (How2Sign Test Set)

| Model | DTW MPJPE-PA (Left Hand) | DTW MPJPE-PA (Right Hand) | Training Time |
|-------|-------------------------|--------------------------|---------------|
| **Hierarchical RVQ-GPT** | ~45mm | ~45mm | 3-4 days |
| **LaMP + Masked T2M** | ~40mm (expected) | ~40mm (expected) | 7-10 days |

**Note:** These are projected benchmarks. Actual results depend on:
- Quality of LaMP pretraining
- QFormer convergence
- Masked T2M refinement steps
- Dataset size and quality

---

## Future Improvements

### Short-term (Next Sprint)

1. **Gradient Checkpointing:**
   - Reduce memory for larger batches
   - Enable batch size 64+ for contrastive learning

2. **Mixed Precision Training:**
   - Add `torch.cuda.amp` support
   - 2x speedup with fp16

3. **Multi-Dataset Training:**
   - Add CSL-Daily, Phoenix-2014T to LaMP
   - Cross-dataset evaluation

### Long-term

1. **Hierarchical Masked Generation:**
   - Generate Q0→Q1→Q2 iteratively
   - Better quality than Q0 only

2. **Cross-Modal Retrieval:**
   - Use LaMP embeddings for text→motion search
   - Motion→text captioning

3. **Fine-tuning:**
   - Instruction tuning on LaMP + Masked T2M
   - Multi-task learning (T2M + M2T + editing)

---

## Citation

If you use this adapted architecture, please cite:

```bibtex
@misc{lamp_motiongpt_adaptation,
  title={LaMP Architecture Adaptation for Sign Language Generation},
  author={Claude (Anthropic)},
  year={2026},
  note={Adapted from LaMP and MotionGPT for How2Sign dataset}
}

@article{lamp2024,
  title={LaMP: Language-Motion Pretraining for Motion Generation},
  author={Original LaMP Authors},
  journal={arXiv preprint},
  year={2024}
}

@article{motiongpt2023,
  title={MotionGPT: Human Motion as Foreign Language},
  author={Original MotionGPT Authors},
  journal={NeurIPS},
  year={2023}
}
```

---

## Contact & Support

For issues, questions, or contributions:

- **GitHub Issues:** [MotionGPT Issues](https://github.com/yourrepo/issues)
- **Documentation:** This file + code comments
- **Email:** [your.email@domain.com]

---

**Document Version:** 1.0
**Last Updated:** 2026-02-10
**Status:** Complete ✅

All files created, tested interfaces validated, ready for training.
