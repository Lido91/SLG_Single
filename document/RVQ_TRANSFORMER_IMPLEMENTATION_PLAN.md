# RVQ Transformer Implementation Plan

## Replicating UniMuMo's Residual Token Transformer Architecture for MotionGPT

**Date:** 2025-12-10
**Dataset:** How2Sign
**Stage:** Stage 2 - Motion Generation Transformer (RVQ-VAE already pretrained)

---

## 1. Overview

### Goal
Implement a transformer architecture that learns from Residual VQ tokens, following UniMuMo's design principles exactly.

### Current Setup
- **RVQ-VAE:** Pretrained with 6 quantizers × 512 tokens each
- **Pre-extracted tokens:** Available for How2Sign dataset
- **Token format:** [B, K=6, T] discrete integer tensors

### UniMuMo Key Design Principles
1. **Additive Embeddings:** Sum embeddings across all K codebooks (not concatenate)
2. **Delayed Pattern Interleaving:** Causal sequence organization for autoregressive generation
3. **Separate Prediction Heads:** One linear layer per codebook
4. **Per-Codebook Loss:** Independent cross-entropy for each quantizer level

---

## 2. Architecture Design

### 2.1 Token Embedding Strategy

```
Input: motion_tokens [B, K=6, T]
       where each token is in [0, 511]

For each timestep t:
    embedded[t] = Σ(motion_emb[k](tokens[:, k, t])) for k in [0, 5]

Output: [B, T, dim] - single embedding per timestep
```

**Key Points:**
- Separate embedding table for each codebook: `nn.ModuleList([nn.Embedding(512, dim) for _ in range(6)])`
- SUM across codebooks (not concatenate) - mirrors RVQ's additive reconstruction
- Each codebook learns different granularity:
  - Codebook 0: Coarse motion (pose)
  - Codebook 5: Fine details (subtle movements)

### 2.2 Delayed Pattern Interleaving

Original codes (parallel):
```
Codebook 0: [c0_0, c0_1, c0_2, c0_3, ...]
Codebook 1: [c1_0, c1_1, c1_2, c1_3, ...]
Codebook 2: [c2_0, c2_1, c2_2, c2_3, ...]
...
Codebook 5: [c5_0, c5_1, c5_2, c5_3, ...]
```

Interleaved sequence (delayed pattern):
```
Step 0: [c0_0]
Step 1: [c0_1, c1_0]
Step 2: [c0_2, c1_1, c2_0]
Step 3: [c0_3, c1_2, c2_1, c3_0]
Step 4: [c0_4, c1_3, c2_2, c3_1, c4_0]
Step 5: [c0_5, c1_4, c2_3, c3_2, c4_1, c5_0]
Step 6: [c0_6, c1_5, c2_4, c3_3, c4_2, c5_1]
...
```

**Causality:** Token `c_k_t` can only depend on:
- `c_k'_t'` where `t' < t` (past timesteps)
- `c_k'_t` where `k' < k` (earlier codebooks at same timestep)

### 2.3 Transformer Architecture

```python
class RVQTransformer(nn.Module):
    def __init__(
        self,
        num_quantizers: int = 6,      # K codebooks
        codebook_size: int = 512,     # Vocabulary per codebook
        dim: int = 512,               # Model dimension
        num_heads: int = 8,           # Attention heads
        num_layers: int = 12,         # Transformer depth
        hidden_scale: int = 4,        # FFN expansion
        dropout: float = 0.1,
        ...
    ):
        # Separate embedding per codebook
        self.motion_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, dim)  # +1 for special token
            for _ in range(num_quantizers)
        ])

        # Transformer backbone
        self.transformer = nn.TransformerEncoder(...)

        # Separate prediction head per codebook
        self.motion_heads = nn.ModuleList([
            nn.Linear(dim, codebook_size)
            for _ in range(num_quantizers)
        ])
```

### 2.4 Forward Pass

```python
def forward(self, motion_codes, condition_tensors):
    B, K, T = motion_codes.shape

    # 1. Pattern interleaving: [B, K, T] -> [B, K, S]
    pattern = self.pattern_provider.get_pattern(T)
    sequence_codes, _, mask = pattern.build_pattern_sequence(
        motion_codes, self.special_token_id
    )

    # 2. Additive embedding: sum across K codebooks
    input_emb = sum([
        self.motion_emb[k](sequence_codes[:, k, :])
        for k in range(K)
    ])  # [B, S, dim]

    # 3. Add text conditioning (cross-attention)
    input_emb = self.fuse_conditions(input_emb, condition_tensors)

    # 4. Transformer forward
    output = self.transformer(input_emb, mask=causal_mask)  # [B, S, dim]

    # 5. Per-codebook prediction
    logits = torch.stack([
        self.motion_heads[k](output)
        for k in range(K)
    ], dim=1)  # [B, K, S, codebook_size]

    # 6. Revert pattern: [B, K, S, card] -> [B, K, T, card]
    logits = pattern.revert_pattern_logits(logits)

    return logits, mask
```

### 2.5 Loss Computation

```python
def compute_loss(self, logits, targets, mask):
    """
    Per-codebook cross-entropy loss

    Args:
        logits: [B, K, T, codebook_size]
        targets: [B, K, T]
        mask: [B, K, T] valid positions
    """
    B, K, T, C = logits.shape

    total_loss = 0
    loss_per_codebook = []

    for k in range(K):
        logits_k = logits[:, k, :, :].reshape(-1, C)  # [B*T, C]
        targets_k = targets[:, k, :].reshape(-1)      # [B*T]
        mask_k = mask[:, k, :].reshape(-1)            # [B*T]

        # Only compute on valid positions
        ce_k = F.cross_entropy(
            logits_k[mask_k],
            targets_k[mask_k]
        )

        total_loss += ce_k
        loss_per_codebook.append(ce_k.detach())

    return total_loss / K, loss_per_codebook
```

---

## 3. File Structure

```
mGPT/
├── archs/
│   ├── tools/
│   │   ├── codebook_patterns.py      # NEW: Pattern & DelayedPatternProvider
│   │   └── ...
│   ├── rvq_transformer.py            # NEW: Main transformer architecture
│   └── ...
├── models/
│   ├── mgpt_rvq_transformer.py       # NEW: Training model (PyTorch Lightning)
│   └── ...
├── data/
│   └── humanml/
│       ├── dataset_rvq_token.py      # NEW: RVQ token dataset loader
│       └── ...
└── ...

configs/
├── h2s_rvq_transformer.yaml          # NEW: Configuration for training
└── ...
```

---

## 4. Implementation Steps

### Step 1: Codebook Patterns (`mGPT/archs/tools/codebook_patterns.py`)

Classes to implement:
- `LayoutCoord`: Named tuple (t, q) for timestep and codebook index
- `Pattern`: Core pattern class with build/revert methods
- `CodebooksPatternProvider`: Abstract base class
- `DelayedPatternProvider`: Delayed interleaving implementation

Key methods:
- `build_pattern_sequence()`: [B, K, T] -> [B, K, S]
- `revert_pattern_sequence()`: [B, K, S] -> [B, K, T]
- `revert_pattern_logits()`: [B, card, K, S] -> [B, card, K, T]

### Step 2: RVQ Transformer (`mGPT/archs/rvq_transformer.py`)

Classes to implement:
- `ScaledEmbedding`: Embedding with optional separate learning rate
- `RVQTransformerOutput`: Dataclass for output (logits, mask)
- `RVQTransformer`: Main model

Key components:
- `self.motion_emb`: ModuleList of K embeddings
- `self.transformer`: Standard transformer encoder
- `self.motion_heads`: ModuleList of K linear heads
- `self.pattern_provider`: DelayedPatternProvider
- Text conditioning via cross-attention (T5 encoder)

### Step 3: Training Model (`mGPT/models/mgpt_rvq_transformer.py`)

PyTorch Lightning module:
- Load pretrained RVQ-VAE (frozen)
- Load T5 encoder for text conditioning
- Training step with per-codebook loss
- Validation with motion generation
- Generation methods (autoregressive sampling)

### Step 4: Dataset Loader (`mGPT/data/humanml/dataset_rvq_token.py`)

Dataset class:
- Load pre-extracted RVQ tokens: [K, T] per sample
- Load text descriptions
- Padding/cropping to fixed length
- Return dict: {'motion_tokens': [K, T], 'text': str, 'length': int}

### Step 5: Configuration (`configs/h2s_rvq_transformer.yaml`)

```yaml
NAME: H2S_RVQ_Transformer
TRAIN:
  STAGE: lm_rvq
  BATCH_SIZE: 32
  END_EPOCH: 500

model:
  target: mGPT.models.mgpt_rvq_transformer.MotionRVQTransformer
  params:
    num_quantizers: 6
    codebook_size: 512
    dim: 512
    num_heads: 8
    num_layers: 12
    text_encoder: "google/flan-t5-base"

    # Pretrained RVQ-VAE (frozen)
    rvq_vae_config: ${vq.h2s_rvq}
    rvq_vae_ckpt: "path/to/pretrained_rvqvae.ckpt"
```

---

## 5. Key Differences from Original MotionGPT

| Aspect | Original MotionGPT | New RVQ Transformer |
|--------|-------------------|---------------------|
| Token representation | String (`<motion_id_X>`) | Integer tensor [B, K, T] |
| Backbone | T5/GPT2 | Custom Transformer |
| Codebook handling | Single (flattened) | Multiple (K separate) |
| Embedding | Single embedding table | K separate tables, summed |
| Prediction head | Single output layer | K separate heads |
| Loss | Single cross-entropy | Per-codebook cross-entropy |
| Sequence | Linear | Delayed pattern interleaved |

---

## 6. Training Pipeline

```
1. Load Data:
   - Pre-extracted RVQ tokens: [B, K=6, T]
   - Text descriptions: List[str]

2. Text Encoding:
   - T5 encoder: text -> [B, T_text, 768]

3. Forward Pass:
   - Pattern interleaving
   - Additive embedding
   - Transformer + cross-attention
   - Per-codebook prediction

4. Loss Computation:
   - Cross-entropy per codebook
   - Average across K codebooks

5. Logging:
   - Total loss
   - Per-codebook loss (for debugging)
   - Learning rate
```

---

## 7. Generation Pipeline

```
1. Input:
   - Text description: str
   - Duration: float (seconds)

2. Text Encoding:
   - T5 encoder: text -> condition_tensors

3. Autoregressive Generation:
   - Initialize: motion_tokens = empty
   - For each step s in range(S):
       - Build pattern sequence
       - Forward pass
       - Sample next token (top-k/top-p)
       - Update sequence

4. Post-processing:
   - Revert pattern sequence
   - Decode with RVQ-VAE decoder
   - Convert to joint positions

5. Output:
   - Motion tokens: [B, K, T]
   - Motion features: [B, T, D]
   - Joint positions: [B, T, J, 3]
```

---

## 8. Expected Results

### Training Metrics
- **Per-codebook loss convergence:**
  - Codebook 0 (coarse): Fastest convergence
  - Codebook 5 (fine): Slowest convergence
  - All should eventually converge to similar values

### Generation Quality
- Coarse motion structure from early codebooks
- Fine details from later codebooks
- Text-aligned motion via cross-attention conditioning

---

## 9. References

- UniMuMo: `/data/hwu/workspace/UniMuMo/`
  - Transformer: `unimumo/audio/audiocraft_/models/mm_lm.py`
  - Patterns: `unimumo/audio/audiocraft_/modules/codebooks_patterns.py`
  - Training: `unimumo/models/transformer_model.py`

- Original MotionGPT: Current codebase
  - RVQ-VAE: `mGPT/archs/mgpt_rvq.py`
  - Quantizer: `mGPT/archs/tools/residual_vq.py`

---

## 10. Next Steps

1. [ ] Implement `codebook_patterns.py`
2. [ ] Implement `rvq_transformer.py`
3. [ ] Implement `mgpt_rvq_transformer.py`
4. [ ] Implement `dataset_rvq_token.py`
5. [ ] Create configuration YAML
6. [ ] Test with small batch
7. [ ] Full training run
8. [ ] Evaluate generation quality
