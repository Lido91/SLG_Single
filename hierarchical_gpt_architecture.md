# Hierarchical RVQ-GPT Architecture Visualization

## Overview

This document visualizes the Hierarchical RVQ-GPT architecture for motion generation, showing how it transforms text/speech into motion codes through a coarse-to-fine hierarchical process.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL RVQ-GPT                             │
│                                                                     │
│  Input: Text/Speech → Output: Motion Codes (Q0, Q1, Q2)           │
│                                                                     │
│  Key Innovation: Q0 → Q1 → Q2 (Coarse → Medium → Fine)            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT MODALITY                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      │                         │
              ┌───────▼──────┐          ┌──────▼──────┐
              │  Text Input  │          │ Speech Input │
              │  (List[str]) │          │ (Waveform)   │
              └───────┬──────┘          └──────┬───────┘
                      │                        │
              ┌───────▼──────┐          ┌──────▼──────────┐
              │ Text Encoder │          │ Speech Encoder  │
              │   (CLIP/     │          │   (HuBERT/      │
              │    BERT)     │          │    Whisper)     │
              └───────┬──────┘          └──────┬──────────┘
                      │                        │
              (B, S, 512/768)           (B, T_audio, 1024)
                      │                        │
              ┌───────▼──────┐          ┌──────▼──────────┐
              │  Text Proj   │          │  Audio Proj     │
              │ 512→1024     │          │ 1024→1024       │
              └───────┬──────┘          └──────┬──────────┘
                      │                        │
                      └────────────┬───────────┘
                                   │
                         Conditioning Features
                           (B, S, 1024)
                                   │
┌──────────────────────────────────┼───────────────────────────────────┐
│                                  │                                   │
│                           HIERARCHICAL                               │
│                        DECODER PIPELINE                              │
│                                  │                                   │
│  ┌───────────────────────────────▼────────────────────────────┐    │
│  │                      LEVEL 1: Q0 DECODER                    │    │
│  │                    (COARSE MOTION CODES)                    │    │
│  │                                                              │    │
│  │  Input: Previous Q0 tokens (B, t)                          │    │
│  │         Conditioning features (B, S, 1024)                  │    │
│  │                                                              │    │
│  │  ┌──────────────────────────────────────────────┐          │    │
│  │  │  Token Embedding (512 → 1024)                │          │    │
│  │  └────────────────┬─────────────────────────────┘          │    │
│  │                   │                                         │    │
│  │  ┌────────────────▼─────────────────────────────┐          │    │
│  │  │  Positional Encoding (Learned/Sinusoidal)    │          │    │
│  │  └────────────────┬─────────────────────────────┘          │    │
│  │                   │                                         │    │
│  │  ┌────────────────▼─────────────────────────────┐          │    │
│  │  │  Transformer Layers × 9                      │          │    │
│  │  │                                               │          │    │
│  │  │  Each layer contains:                        │          │    │
│  │  │  ┌──────────────────────────────────┐       │          │    │
│  │  │  │ 1. Causal Self-Attention         │       │          │    │
│  │  │  │    - Attends to previous Q0      │       │          │    │
│  │  │  └──────────────┬───────────────────┘       │          │    │
│  │  │                 │                            │          │    │
│  │  │  ┌──────────────▼───────────────────┐       │          │    │
│  │  │  │ 2. Cross-Attention to Conditioning│      │          │    │
│  │  │  │    - Attends to text/speech       │      │          │    │
│  │  │  └──────────────┬───────────────────┘       │          │    │
│  │  │                 │                            │          │    │
│  │  │  ┌──────────────▼───────────────────┐       │          │    │
│  │  │  │ 3. Feed-Forward Network          │       │          │    │
│  │  │  │    - 1024 → 4096 → 1024          │       │          │    │
│  │  │  └──────────────┬───────────────────┘       │          │    │
│  │  │                 │                            │          │    │
│  │  │  ┌──────────────▼───────────────────┐       │          │    │
│  │  │  │ 4. Layer Norm + Residual         │       │          │    │
│  │  │  └──────────────────────────────────┘       │          │    │
│  │  └─────────────────┬──────────────────────────┘          │    │
│  │                    │                                      │    │
│  │  ┌─────────────────▼──────────────────────────┐          │    │
│  │  │  Output Projection (1024 → 513)            │          │    │
│  │  │  (512 codes + 1 EOS token)                 │          │    │
│  │  └────────────────┬───────────────────────────┘          │    │
│  │                   │                                       │    │
│  │        Q0 Logits (B, t, 513)                             │    │
│  │                   │                                       │    │
│  │  ┌────────────────▼───────────────────────────┐          │    │
│  │  │  Argmax/Sample → Q0 Token                  │          │    │
│  │  └────────────────┬───────────────────────────┘          │    │
│  └──────────────────┼────────────────────────────────────────┘    │
│                     │                                             │
│         Q0 Embeddings (B, t, 1024)                                │
│                     │                                             │
│  ┌──────────────────▼─────────────────────────────────────────┐  │
│  │                   LEVEL 2: Q1 DECODER                       │  │
│  │                 (MEDIUM REFINEMENT CODES)                   │  │
│  │                                                              │  │
│  │  Input: Previous Q1 tokens (B, t)                          │  │
│  │         Conditioning features (B, S, 1024)                  │  │
│  │         Q0 embeddings (B, t, 1024) ★ HIERARCHICAL         │  │
│  │                                                              │  │
│  │  ┌──────────────────────────────────────────────┐          │  │
│  │  │  Token Embedding (512 → 1024)                │          │  │
│  │  └────────────────┬─────────────────────────────┘          │  │
│  │                   │                                         │  │
│  │  ┌────────────────▼─────────────────────────────┐          │  │
│  │  │  Positional Encoding                         │          │  │
│  │  └────────────────┬─────────────────────────────┘          │  │
│  │                   │                                         │  │
│  │  ┌────────────────▼─────────────────────────────┐          │  │
│  │  │  Transformer Layers × 9                      │          │  │
│  │  │                                               │          │  │
│  │  │  Each layer contains:                        │          │  │
│  │  │  ┌──────────────────────────────────┐       │          │  │
│  │  │  │ 1. Causal Self-Attention         │       │          │  │
│  │  │  │    - Attends to previous Q1      │       │          │  │
│  │  │  └──────────────┬───────────────────┘       │          │  │
│  │  │                 │                            │          │  │
│  │  │  ┌──────────────▼───────────────────┐       │          │  │
│  │  │  │ 2. Cross-Attention to Conditioning│      │          │  │
│  │  │  │    - Attends to text/speech       │      │          │  │
│  │  │  └──────────────┬───────────────────┘       │          │  │
│  │  │                 │                            │          │  │
│  │  │  ┌──────────────▼───────────────────┐       │          │  │
│  │  │  │ 3. Cross-Attention to Q0 ★       │       │          │  │
│  │  │  │    - Attends to coarse codes     │       │          │  │
│  │  │  └──────────────┬───────────────────┘       │          │  │
│  │  │                 │                            │          │  │
│  │  │  ┌──────────────▼───────────────────┐       │          │  │
│  │  │  │ 4. Feed-Forward Network          │       │          │  │
│  │  │  │    - 1024 → 4096 → 1024          │       │          │  │
│  │  │  └──────────────┬───────────────────┘       │          │  │
│  │  │                 │                            │          │  │
│  │  │  ┌──────────────▼───────────────────┐       │          │  │
│  │  │  │ 5. Layer Norm + Residual         │       │          │  │
│  │  │  └──────────────────────────────────┘       │          │  │
│  │  └─────────────────┬──────────────────────────┘          │  │
│  │                    │                                      │  │
│  │  ┌─────────────────▼──────────────────────────┐          │  │
│  │  │  Output Projection (1024 → 513)            │          │  │
│  │  └────────────────┬───────────────────────────┘          │  │
│  │                   │                                       │  │
│  │        Q1 Logits (B, t, 513)                             │  │
│  │                   │                                       │  │
│  │  ┌────────────────▼───────────────────────────┐          │  │
│  │  │  Argmax/Sample → Q1 Token                  │          │  │
│  │  └────────────────┬───────────────────────────┘          │  │
│  └──────────────────┼────────────────────────────────────────┘  │
│                     │                                           │
│         Q1 Embeddings (B, t, 1024)                              │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐  │
│  │                   LEVEL 3: Q2 DECODER                     │  │
│  │                  (FINE REFINEMENT CODES)                  │  │
│  │                                                            │  │
│  │  Input: Previous Q2 tokens (B, t)                        │  │
│  │         Conditioning features (B, S, 1024)                │  │
│  │         Q0 + Q1 embeddings (B, 2t, 1024) ★ HIERARCHICAL │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────┐        │  │
│  │  │  Token Embedding (512 → 1024)                │        │  │
│  │  └────────────────┬─────────────────────────────┘        │  │
│  │                   │                                       │  │
│  │  ┌────────────────▼─────────────────────────────┐        │  │
│  │  │  Positional Encoding                         │        │  │
│  │  └────────────────┬─────────────────────────────┘        │  │
│  │                   │                                       │  │
│  │  ┌────────────────▼─────────────────────────────┐        │  │
│  │  │  Transformer Layers × 9                      │        │  │
│  │  │                                               │        │  │
│  │  │  Each layer contains:                        │        │  │
│  │  │  ┌──────────────────────────────────┐       │        │  │
│  │  │  │ 1. Causal Self-Attention         │       │        │  │
│  │  │  │    - Attends to previous Q2      │       │        │  │
│  │  │  └──────────────┬───────────────────┘       │        │  │
│  │  │                 │                            │        │  │
│  │  │  ┌──────────────▼───────────────────┐       │        │  │
│  │  │  │ 2. Cross-Attention to Conditioning│      │        │  │
│  │  │  │    - Attends to text/speech       │      │        │  │
│  │  │  └──────────────┬───────────────────┘       │        │  │
│  │  │                 │                            │        │  │
│  │  │  ┌──────────────▼───────────────────┐       │        │  │
│  │  │  │ 3. Cross-Attention to Q0+Q1 ★    │       │        │  │
│  │  │  │    - Attends to coarse+medium    │       │        │  │
│  │  │  └──────────────┬───────────────────┘       │        │  │
│  │  │                 │                            │        │  │
│  │  │  ┌──────────────▼───────────────────┐       │        │  │
│  │  │  │ 4. Feed-Forward Network          │       │        │  │
│  │  │  │    - 1024 → 4096 → 1024          │       │        │  │
│  │  │  └──────────────┬───────────────────┘       │        │  │
│  │  │                 │                            │        │  │
│  │  │  ┌──────────────▼───────────────────┐       │        │  │
│  │  │  │ 5. Layer Norm + Residual         │       │        │  │
│  │  │  └──────────────────────────────────┘       │        │  │
│  │  └─────────────────┬──────────────────────────┘        │  │
│  │                    │                                    │  │
│  │  ┌─────────────────▼──────────────────────────┐        │  │
│  │  │  Output Projection (1024 → 513)            │        │  │
│  │  └────────────────┬───────────────────────────┘        │  │
│  │                   │                                     │  │
│  │        Q2 Logits (B, t, 513)                           │  │
│  │                   │                                     │  │
│  │  ┌────────────────▼───────────────────────────┐        │  │
│  │  │  Argmax/Sample → Q2 Token                  │        │  │
│  │  └────────────────┬───────────────────────────┘        │  │
│  └──────────────────┼──────────────────────────────────────┘  │
│                     │                                         │
└─────────────────────┼─────────────────────────────────────────┘
                      │
              Q0, Q1, Q2 Tokens
                      │
              ┌───────▼────────┐
              │   RVQ-VAE      │
              │   Decoder      │
              │   (Frozen)     │
              └───────┬────────┘
                      │
              Motion Sequence
           (B, T_frames, 133)
```

---

## Autoregressive Generation Process

```
For each timestep t = 0, 1, 2, ..., max_len:

Step 1: Generate Q0[t]
┌─────────────────────────────────────────────┐
│ Input:  Q0[0:t], Conditioning               │
│ Output: Q0[t] ~ P(Q0[t] | Q0[<t], cond)    │
│                                             │
│ Check: Is Q0[t] == EOS?                     │
│   Yes → STOP                                │
│   No  → Continue                            │
└─────────────────────────────────────────────┘
           ↓
Step 2: Generate Q1[t] (conditioned on Q0[0:t])
┌─────────────────────────────────────────────┐
│ Input:  Q1[0:t], Conditioning, Q0[0:t] ★   │
│ Output: Q1[t] ~ P(Q1[t] | Q1[<t], Q0[≤t])  │
└─────────────────────────────────────────────┘
           ↓
Step 3: Generate Q2[t] (conditioned on Q0[0:t] + Q1[0:t])
┌──────────────────────────────────────────────┐
│ Input:  Q2[0:t], Conditioning,              │
│         Q0[0:t] + Q1[0:t] ★                 │
│ Output: Q2[t] ~ P(Q2[t] | Q2[<t], Q0, Q1)  │
└──────────────────────────────────────────────┘
           ↓
    [Q0[t], Q1[t], Q2[t]] generated
           ↓
    Continue to t+1
```

---

## Training Process

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                         │
└──────────────────────────────────────────────────────────┘

1. Load Batch
   ┌─────────────────────────────────────────┐
   │ Texts/Audio: (B,)                       │
   │ Motion Codes: (B, T, 6)                 │
   │   → Use first 3: (B, T, 3)              │
   │   → Split: Q0, Q1, Q2 each (B, T)       │
   └─────────────────────────────────────────┘
           ↓
2. Forward Pass (Teacher Forcing)
   ┌─────────────────────────────────────────┐
   │ Q0: Input=GT[:-1], Target=GT[1:]        │
   │   → Logits_Q0: (B, T, 513)              │
   │   → Loss_Q0 = CE(Logits_Q0, Target_Q0)  │
   │                                         │
   │ Q1: Input=GT[:-1] + Q0_emb ★            │
   │   → Logits_Q1: (B, T, 513)              │
   │   → Loss_Q1 = CE(Logits_Q1, Target_Q1)  │
   │                                         │
   │ Q2: Input=GT[:-1] + Q0_emb + Q1_emb ★  │
   │   → Logits_Q2: (B, T, 513)              │
   │   → Loss_Q2 = CE(Logits_Q2, Target_Q2)  │
   └─────────────────────────────────────────┘
           ↓
3. Compute Total Loss
   ┌─────────────────────────────────────────┐
   │ Total Loss = Loss_Q0 + Loss_Q1 + Loss_Q2│
   │                                         │
   │ Optional: Weighted                      │
   │   = 1.0*Loss_Q0 + 0.5*Loss_Q1 +         │
   │     0.25*Loss_Q2                        │
   └─────────────────────────────────────────┘
           ↓
4. Backward + Optimizer Step
   ┌─────────────────────────────────────────┐
   │ Loss.backward()                         │
   │ optimizer.step()                        │
   │ optimizer.zero_grad()                   │
   └─────────────────────────────────────────┘
```

---

## Scheduled Sampling Strategies

```
┌──────────────────────────────────────────────────────────────┐
│              SCHEDULED SAMPLING (Optional)                   │
│                                                              │
│  Controls mixing of GT vs predicted tokens during training  │
└──────────────────────────────────────────────────────────────┘

Strategy 1: Full Teacher Forcing (pkeep=1.0)
┌─────────────────────────────────────────┐
│ Q1 Conditioning: Use GT Q0 tokens      │
│ Q2 Conditioning: Use GT Q1 tokens      │
│                                         │
│ ✓ Stable training                      │
│ ✗ Exposure bias                        │
└─────────────────────────────────────────┘

Strategy 2: Per-Sample Sampling (pkeep=0.5)
┌─────────────────────────────────────────┐
│ For each sample in batch:              │
│   if rand() < pkeep:                   │
│     Use GT token                       │
│   else:                                │
│     Use predicted token (argmax)       │
│                                         │
│ ✓ Reduces exposure bias               │
│ ✗ More unstable                        │
└─────────────────────────────────────────┘

Strategy 3: T2M-Style (per-token, random)
┌─────────────────────────────────────────┐
│ For each token position:              │
│   if rand() < pkeep:                   │
│     Use GT token                       │
│   else:                                │
│     Use random token                   │
│                                         │
│ ✓ Most robust to distribution shift   │
│ ✓ Matches T2M-GPT training             │
└─────────────────────────────────────────┘
```

---

## Architecture Variants

### Variant 1: 3-Layer Hierarchical (Default)
```
Uses: Q0, Q1, Q2 (first 3 of 6 quantizers)
Size: ~258M parameters
Speed: ~3× slower than baseline (3 forward passes/timestep)
```

### Variant 2: 6-Layer Hierarchical
```
Uses: All 6 quantizers (Q0, Q1, Q2, Q3, Q4, Q5)

Conditioning Modes:
  a) Chain mode:
     Q0 ← text
     Q1 ← Q0 + text
     Q2 ← Q1 + text
     Q3 ← Q2 + text
     ...

  b) Full mode:
     Q0 ← text
     Q1 ← Q0 + text
     Q2 ← Q0 + Q1 + text
     Q3 ← Q0 + Q1 + Q2 + text
     ...

Size: ~516M parameters
Speed: ~6× slower than baseline
```

### Variant 3: T2M-Style Q0 + Hierarchical Q1/Q2
```
Q0: T2M-GPT style (text prepended to sequence)
  - No cross-attention
  - Sinusoidal positional encoding
  - 9 layers

Q1, Q2: Cross-attention style
  - Cross-attention to text
  - Cross-attention to previous quantizers
  - 3 layers each (lighter)

Size: ~180M parameters
Speed: Moderate
```

---

## Key Features

### 1. Hierarchical Conditioning
```
Q0: Independent (coarse motion structure)
Q1: Depends on Q0 (medium refinement)
Q2: Depends on Q0+Q1 (fine details)

Each level refines the previous!
```

### 2. Multi-Modal Support
```
Text Mode:
  CLIP (512D)  → 1024D
  BERT (768D)  → 1024D

Speech Mode:
  HuBERT (1024D) → 1024D
  Whisper (1280D) → 1024D
  WavLM (1024D)  → 1024D
```

### 3. EOS Token Handling
```
Codebook: [0, 1, 2, ..., 511]
EOS Token: 512

Generation stops when Q0 predicts EOS
Q1, Q2 continue to predict EOS for finished samples
```

### 4. Progressive Refinement
```
Can decode with:
  - Q0 only (coarse)
  - Q0+Q1 (medium)
  - Q0+Q1+Q2 (full fidelity)

Allows quality/speed tradeoff!
```

---

## Comparison: Baseline vs Hierarchical

```
┌────────────────────────────────────────────────────────────┐
│              BASELINE RVQ-GPT                              │
│                                                            │
│  Text → Single Decoder → [Q0, Q1, Q2, Q3, Q4, Q5]        │
│                           (all parallel)                   │
│                                                            │
│  ✓ Fast (1 forward pass/timestep)                        │
│  ✗ No explicit coordination between quantizers           │
│  ✗ Ignores RVQ's hierarchical structure                  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│           HIERARCHICAL RVQ-GPT (NEW)                       │
│                                                            │
│  Text → Q0 Decoder → Q0                                   │
│            ↓                                               │
│         Q1 Decoder ← Q0 → Q1                              │
│            ↓                                               │
│         Q2 Decoder ← Q0+Q1 → Q2                           │
│                                                            │
│  ✓ Explicit hierarchical dependency                       │
│  ✓ Matches RVQ's coarse-to-fine structure                │
│  ✓ Better coherence between quantizer levels             │
│  ✗ Slower (3 forward passes/timestep for 3-layer)        │
└────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Joint Probability
```
P(Q0, Q1, Q2 | text) = ∏_{t=1}^{T} [
    P(Q0^t | Q0^{<t}, text) ×
    P(Q1^t | Q1^{<t}, Q0^{≤t}, text) ×
    P(Q2^t | Q2^{<t}, Q0^{≤t}, Q1^{≤t}, text)
]

Where:
  ^{<t}  = tokens before timestep t (causal)
  ^{≤t}  = tokens up to and including t
  Q0^t   = coarse code at time t
  Q1^t   = medium refinement at time t
  Q2^t   = fine refinement at time t
```

### Factorization Advantage
```
Traditional: P(Q0, Q1, Q2 | text) = ∏_t ∏_q P(Q_q^t | Q_{all}^{<t}, text)
             (all quantizers independent)

Hierarchical: P(Q0, Q1, Q2 | text) = ∏_t P(Q0^t | ...) ×
                                          P(Q1^t | Q0, ...) ×
                                          P(Q2^t | Q0, Q1, ...)
             (explicit coarse-to-fine dependency)
```

---

## Model Statistics

```
┌─────────────────────────────────────────────────────────┐
│                  3-LAYER HIERARCHICAL                   │
├─────────────────────────────────────────────────────────┤
│ Q0 Decoder:           ~82M parameters                   │
│ Q1 Decoder:           ~88M parameters                   │
│ Q2 Decoder:           ~88M parameters                   │
│ ─────────────────────────────────────────────────────── │
│ Total:                ~258M parameters                  │
│                                                         │
│ Embedding Dim:        1024                              │
│ Num Layers/Decoder:   9                                 │
│ Num Heads:            16                                │
│ FFN Dim:              4096                              │
│ Codebook Size:        512 (+ 1 EOS = 513)              │
│ Block Size:           200 tokens (800 frames @ 4× ds)  │
└─────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Training
```bash
# Generate motion tokens from VAE
python get_motion_code.py --cfg configs/deto_h2s_rvq.yaml

# Train hierarchical GPT
python train.py --cfg configs/deto_h2s_rvq_hierarchical_3layer.yaml --nodebug
```

### Generation
```python
from mGPT.archs.mgpt_rvq_hierarchical import HierarchicalRVQGPT

# Initialize model
model = HierarchicalRVQGPT(
    num_vq=512,
    embed_dim=1024,
    block_size=200,
    num_layers=9,
    text_encoder_type='clip'
)

# Generate from text
texts = ["A person waves hello"]
codes, lengths = model.generate_conditional(
    texts=texts,
    max_len=50,
    temperature=1.0
)

# codes: (B, T, 3) - Q0, Q1, Q2 codes
# Decode with VAE to get motion
motion = vae.decode(codes)  # (B, T*4, 133)
```

---

## References

- **SOKE**: Hierarchical sign language generation (body parts)
- **SoundStream**: Residual vector quantization for audio
- **T2M-GPT**: Text prepending for motion generation
- **MoMask**: Masked modeling for motion

---

**Implementation**: `mGPT/archs/mgpt_rvq_hierarchical.py`
**Documentation**: `document/HIERARCHICAL_RVQ_IMPLEMENTATION.md`
**Tests**: `test_hierarchical_rvq.py`
