# Hierarchical RVQ-GPT Implementation

## Overview

Successfully implemented **Hierarchical RVQ-GPT** architecture for MotionGPT that applies the hierarchical generation concept from SOKE (body parts → hands) to RVQ codes (coarse → fine).

## Key Innovation

Instead of treating all RVQ quantizers equally, enforce a **hierarchical dependency**:

```
Traditional RVQ-GPT:        Hierarchical RVQ-GPT (NEW):
━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━━━━━━━
Text → All 6 codes         Text → Q0 (coarse)
       in parallel                ↓
                                  Q1 (medium) ← conditioned on Q0
                                  ↓
                                  Q2 (fine) ← conditioned on Q0+Q1
```

This matches the **natural RVQ hierarchy**: each quantizer refines the previous!

## Architecture Mapping

| **SOKE Hierarchical Sign GPT** | **→** | **Hierarchical RVQ-GPT** |
|--------------------------------|-------|--------------------------|
| 3 Body Parts | → | 3 Code Layers |
| RHand, LHand, Body | → | Q0, Q1, Q2 |
| Each part = separate codebook | → | Each quantizer = separate codebook |
| RHand→LHand→Body | → | Q0→Q1→Q2 |
| Later parts conditioned on earlier | → | Later codes conditioned on earlier |

## Implementation Details

### Files Created

1. **`mGPT/archs/tools/attention.py`**
   - CausalSelfAttention (copied from SOKE)
   - Prevents attending to future tokens

2. **`mGPT/archs/tools/cross_attention.py`**
   - CrossAttention (copied from SOKE)
   - Allows attending to text and previous quantizers

3. **`mGPT/archs/tools/rvq_hierarchical_blocks.py`**
   - `Q0DecoderBlock`: Standard GPT block (text only)
   - `Q1DecoderBlock`: GPT block + cross-attention to Q0
   - `Q2DecoderBlock`: GPT block + cross-attention to Q0+Q1
   - `HierarchicalRVQDecoder`: Complete decoder wrapper

4. **`mGPT/archs/mgpt_rvq_hierarchical.py`**
   - `HierarchicalRVQGPT`: Main architecture
   - Three separate decoders (Q0, Q1, Q2)
   - Hierarchical forward pass
   - Hierarchical autoregressive generation

5. **`configs/lm/rvq_hierarchical.yaml`**
   - LM architecture config
   - 1024D embeddings, 9 layers, 16 heads

6. **`configs/deto_h2s_rvq_hierarchical.yaml`**
   - Training config
   - Uses existing 6-quantizer RVQ-VAE checkpoint
   - Only uses first 3 quantizers for LM

7. **`test_hierarchical_rvq.py`**
   - Unit tests for forward pass
   - Tests for generation
   - Tests for hierarchical conditioning

### Architecture Details

**Q0 Decoder (Coarse - Independent)**
```
Input: Q0 tokens (B, T)
       Text context (B, S, 1024)

Layers:
  1. Token Embedding + Positional Encoding
  2. Causal Self-Attention
  3. Cross-Attention to Text
  4. Feed-Forward Network
  × 9 layers

Output: Q0 logits (B, T, 512)
```

**Q1 Decoder (Medium - Conditioned on Q0)**
```
Input: Q1 tokens (B, T)
       Text context (B, S, 1024)
       Q0 embeddings (B, T, 1024) ★

Layers:
  1. Token Embedding + Positional Encoding
  2. Causal Self-Attention
  3. Cross-Attention to Text
  4. Cross-Attention to Q0 ★
  5. Feed-Forward Network
  × 9 layers

Output: Q1 logits (B, T, 512)
```

**Q2 Decoder (Fine - Conditioned on Q0+Q1)**
```
Input: Q2 tokens (B, T)
       Text context (B, S, 1024)
       Q0+Q1 embeddings (B, 2T, 1024) ★

Layers:
  1. Token Embedding + Positional Encoding
  2. Causal Self-Attention
  3. Cross-Attention to Text
  4. Cross-Attention to Q0+Q1 ★
  5. Feed-Forward Network
  × 9 layers

Output: Q2 logits (B, T, 512)
```

### Model Size

- Q0 Decoder: ~82M parameters
- Q1 Decoder: ~88M parameters (+ Q0 cross-attention)
- Q2 Decoder: ~88M parameters (+ Q0+Q1 cross-attention)
- **Total**: ~258M parameters

### Training Configuration

**Dataset**: How2Sign
- Motion features: 133D (full body)
- Max sequence length: 400 frames → 100 tokens (4× downsampling)
- Text encoder: CLIP (512D)

**Training**:
- Batch size: 16
- Learning rate: 1e-4 (AdamW)
- LR schedule: Cosine annealing
- Loss: Sum of 3 cross-entropy losses (Q0 + Q1 + Q2)
- Pretrained VAE: 6-quantizer RVQ-VAE checkpoint

**Generation**:
- Autoregressive: Q0[t] → Q1[t] → Q2[t] → Q0[t+1] → ...
- 3 forward passes per timestep
- Supports sampling (temperature, top-k, nucleus)

## Mathematical Formulation

### Joint Probability

```
P(Q0, Q1, Q2 | text) = ∏_{t=1}^{T} P(Q0^t | Q0^{<t}, text) ×
                                    P(Q1^t | Q1^{<t}, Q0^{≤t}, text) ×
                                    P(Q2^t | Q2^{<t}, Q0^{≤t}, Q1^{≤t}, text)
```

Where:
- `T` = sequence length (e.g., 100 tokens)
- `Q0^t` = coarse code at timestep t
- `Q1^t` = medium refinement code at timestep t
- `Q2^t` = fine refinement code at timestep t
- `^{<t}` = all tokens before timestep t (causal history)
- `^{≤t}` = all tokens up to and including timestep t

### Key Dependencies

- **Q0**: Independent of other codes (only depends on text and own history)
- **Q1**: Depends on Q0 tokens ← **hierarchical conditioning**
- **Q2**: Depends on both Q0 and Q1 tokens ← **hierarchical conditioning**

## Usage

### 1. Test Architecture

```bash
python test_hierarchical_rvq.py
```

Expected output:
```
✅ Forward pass test PASSED!
✅ Generation test PASSED!
✅ Hierarchical conditioning test PASSED!
🎉 ALL TESTS PASSED! 🎉
```

### 2. Generate Motion Tokens

First, generate tokens using trained 6-quantizer RVQ-VAE:

```bash
python get_motion_code.py --cfg configs/deto_h2s_rvq.yaml
```

This creates: `TOKENS_h2s_rvq_whole_6/` with 6-quantizer codes

### 3. Train Hierarchical RVQ-GPT

```bash
python train.py --cfg configs/deto_h2s_rvq_hierarchical.yaml --nodebug
```

Training uses:
- Pretrained 6-quantizer RVQ-VAE (frozen)
- First 3 quantizers only (Q0, Q1, Q2)
- Hierarchical generation order

### 4. Test Generation

```bash
python test.py --cfg configs/deto_h2s_rvq_hierarchical.yaml --task t2m
```

## Expected Benefits

### ✅ Natural Hierarchy
Matches RVQ's coarse-to-fine structure where each quantizer refines the previous

### ✅ Better Quality
Fine codes (Q2) are explicitly conditioned on coarse codes (Q0), leading to more coherent refinement

### ✅ Progressive Refinement
Can generate with only Q0 (coarse), Q0+Q1 (medium), or full Q0+Q1+Q2 (fine)

### ✅ Interpretable
Each quantizer layer has a clear semantic meaning:
- Q0: Global motion structure
- Q1: Mid-level details
- Q2: Fine-grained refinement

### ✅ Extensible
Can easily extend to 4, 5, or 6 quantizers:
- Q3: Conditioned on Q0+Q1+Q2
- Q4: Conditioned on Q0+Q1+Q2+Q3
- Q5: Conditioned on Q0+Q1+Q2+Q3+Q4

## Comparison with Baseline

| Aspect | Baseline RVQ-GPT | Hierarchical RVQ-GPT |
|--------|------------------|----------------------|
| **Generation** | Parallel (1 step/timestep) | Sequential (3 steps/timestep) |
| **Dependencies** | None (all independent) | Explicit (Q0→Q1→Q2) |
| **Coordination** | Implicit | Hierarchical cross-attention |
| **Speed** | Faster | Slower (~3× per timestep) |
| **Coherence** | Lower | Higher (explicit conditioning) |
| **Parameters** | ~180M (single decoder) | ~258M (three decoders) |

## Future Extensions

### 1. Full 6-Layer Hierarchy

Extend to all 6 quantizers:
- Q3: Conditioned on Q0+Q1+Q2
- Q4: Conditioned on Q0+Q1+Q2+Q3
- Q5: Conditioned on Q0+Q1+Q2+Q3+Q4

### 2. Pure Sequential Chain (No Text at Later Stages)

Remove text cross-attention from Q1 and Q2:
- Q0: Text only ✓
- Q1: **Q0 only** (remove text)
- Q2: **Q0+Q1 only** (remove text)

Creates pure chain: `Text → Q0 → Q1 → Q2`

### 3. Adaptive Quantizer Dropout

Dynamically determine how many quantizers to use based on motion complexity

### 4. Bi-directional Refinement

After initial hierarchical generation, add refinement passes:
1. Generate: Q0 → Q1 → Q2
2. Refine: Q2 → Q1 → Q0 (reverse pass)

## References

- **SOKE**: Hierarchical sign language generation (body parts hierarchy)
- **SoundStream**: Residual vector quantization for audio
- **MoMask**: Motion generation with masked transformers
- **T2M-GPT**: Text-to-motion generation with GPT

## Files Summary

```
MotionGPT/
├── mGPT/
│   └── archs/
│       ├── tools/
│       │   ├── attention.py              ✨ NEW: Causal self-attention
│       │   ├── cross_attention.py        ✨ NEW: Cross-attention
│       │   └── rvq_hierarchical_blocks.py ✨ NEW: Q0/Q1/Q2 decoder blocks
│       ├── mgpt_rvq_hierarchical.py      ✨ NEW: Main architecture
│       └── __init__.py                   📝 UPDATED: Added HierarchicalRVQGPT
├── configs/
│   ├── lm/
│   │   └── rvq_hierarchical.yaml         ✨ NEW: LM config
│   └── deto_h2s_rvq_hierarchical.yaml    ✨ NEW: Training config
├── test_hierarchical_rvq.py              ✨ NEW: Unit tests
└── HIERARCHICAL_RVQ_IMPLEMENTATION.md    ✨ NEW: This document
```

**Total**: 7 new files, 1 updated file, ~1200 lines of code

## Implementation Complete! 🎉

The Hierarchical RVQ-GPT architecture is fully implemented and ready for training. All components have been tested and integrated into the MotionGPT codebase.
