# FSQ Integration Guide for MotionGPT

This guide explains how to use the newly integrated **Finite Scalar Quantization (FSQ)** architecture in your MotionGPT project.

## Table of Contents
1. [What is FSQ?](#what-is-fsq)
2. [Architecture Overview](#architecture-overview)
3. [Files Created](#files-created)
4. [Available Models](#available-models)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Advantages Over Traditional VQ-VAE](#advantages)
8. [Usage Examples](#usage-examples)

---

## What is FSQ?

**Finite Scalar Quantization (FSQ)** eliminates learned codebooks entirely, using a mathematically defined quantization scheme based on finite scalar levels. This provides several advantages over traditional VQ-VAE:

- ✅ **No codebook collapse** - All codes are mathematically guaranteed to exist
- ✅ **Simpler training** - No EMA updates or commitment losses needed
- ✅ **Stable training** - No hyperparameter tuning for codebook learning
- ✅ **Deterministic** - Reproducible results with fixed mathematical mapping
- ✅ **Memory efficient** - No learned codebook parameters to store

**Reference**: "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)

---

## Architecture Overview

### Input Data Format
Your project uses **SMPLX parameters** (133D after preprocessing):
- `smplx_root_pose`: 3D (1 joint)
- `smplx_body_pose`: 63D (21 joints, upper body only)
- `smplx_lhand_pose`: 45D (15 joints)
- `smplx_rhand_pose`: 45D (15 joints)
- `smplx_jaw_pose`: 3D (1 joint)
- `smplx_expr`: 10D (expression)

### FSQ Pipeline

```
Input SMPLX [B, T, 133]
    ↓
Encoder (Conv1D + ResNet)
    ↓
[B, 512, T']  (downsampled)
    ↓
FSQ Quantizer
    ↓
Discrete Tokens [B, T']  (range: 0-624 for 625 codes)
    ↓
Decoder (Conv1D + ResNet)
    ↓
Reconstructed SMPLX [B, T, 133]
```

---

## Files Created

### Core FSQ Implementation
- **`mGPT/archs/tools/vq/FSQ.py`** - FSQ quantizer module
- **`mGPT/archs/tools/vq/__init__.py`** - VQ tools package

### Model Architectures
- **`mGPT/archs/mgpt_fsq.py`** - FSQ-based VQ-VAE
- **`mGPT/archs/mgpt_fsq_rvq.py`** - FSQ-based Residual VQ-VAE
- **`mGPT/archs/__init__.py`** - Updated to export FSQ models

### Configurations
- **`configs/vq/fsq.yaml`** - FSQ VQ-VAE config (625 codes)
- **`configs/vq/fsq_rvq.yaml`** - FSQ Residual VQ-VAE config (3 stages)
- **`configs/vq/fsq_large.yaml`** - Large FSQ config (4096 codes)
- **`configs/deto_h2s_fsq.yaml`** - Full training config for How2Sign with FSQ
- **`configs/deto_h2s_fsq_rvq.yaml`** - Full training config for How2Sign with FSQ-RVQ

---

## Available Models

### 1. FSQVae
Basic FSQ-based VQ-VAE with single-stage quantization.

```python
from mGPT.archs import FSQVae

model = FSQVae(
    nfeats=133,                    # SMPLX parameters
    fsq_levels=[5, 5, 5, 5],      # 5^4 = 625 codes
    code_dim=4,                    # Latent dimension
    output_emb_width=512,
    down_t=2,
    stride_t=2,
    width=512,
    depth=3,
)
```

**Codebook size**: 625 codes (5^4)

**Use case**: Simple, stable quantization for motion tokens

### 2. FSQRVQVae
Residual FSQ-based VQ-VAE with hierarchical quantization.

```python
from mGPT.archs import FSQRVQVae

model = FSQRVQVae(
    nfeats=133,
    num_quantizers=3,              # Q0, Q1, Q2
    fsq_levels=[5, 5, 5, 5],      # 625 codes per stage
    code_dim=4,
    quantize_dropout_prob=0.2,     # 20% dropout
    quantize_dropout_cutoff_index=0,  # Keep Q0 always
    shared_codebook=False,         # Separate FSQ per stage
    output_emb_width=512,
    down_t=2,
)
```

**Codebook size**: 625 codes per stage (3 stages total)

**Use case**: Hierarchical representation, coarse-to-fine generation

---

## Configuration

### Basic FSQ Config (`configs/vq/fsq.yaml`)

```yaml
target: mGPT.archs.mgpt_fsq.FSQVae
params:
  fsq_levels: [5, 5, 5, 5]  # 625 codes
  code_dim: 4
  output_emb_width: 512
  down_t: 2
  stride_t: 2
  width: 512
  depth: 3
  nfeats: 133
```

### FSQ-RVQ Config (`configs/vq/fsq_rvq.yaml`)

```yaml
target: mGPT.archs.mgpt_fsq_rvq.FSQRVQVae
params:
  num_quantizers: 3          # Number of residual stages
  fsq_levels: [5, 5, 5, 5]  # 625 codes per stage
  code_dim: 4
  quantize_dropout_prob: 0.2
  quantize_dropout_cutoff_index: 0
  shared_codebook: false
  output_emb_width: 512
  down_t: 2
  nfeats: 133
```

### Codebook Size Options

| FSQ Levels | Codebook Size | Notes |
|------------|---------------|-------|
| [5, 5, 5, 5] | 625 | Default, good balance |
| [8, 8, 8, 8] | 4096 | Larger vocabulary, higher quality |
| [5, 5, 5, 5, 5] | 3125 | 5D latent, more expressive |
| [7, 7, 7] | 343 | Smaller, faster |

---

## Training

### Train FSQ VQ-VAE on How2Sign

```bash
python train.py \
  --cfg configs/deto_h2s_fsq.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --nodebug
```

### Train FSQ-RVQ on How2Sign

```bash
python train.py \
  --cfg configs/deto_h2s_fsq_rvq.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --nodebug
```

### Multi-GPU Training

```bash
python train.py \
  --cfg configs/deto_h2s_fsq_rvq.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --devices 0,1 \
  --nodebug
```

### Training Tips

1. **No commitment loss needed**: Set `LOSS.LAMBDA_COMMIT: 0.0` (FSQ has no commitment loss)

2. **Learning rate**: Start with `2e-4` (same as standard VQ-VAE)

3. **Batch size**: 64 works well for FSQ (less memory than standard VQ-VAE)

4. **Monitoring**: Watch perplexity - should be close to codebook size (e.g., ~625 for full utilization)

---

## Advantages

### 1. No Codebook Collapse

**Traditional VQ-VAE Problem**:
```python
# Only 150/2048 codes used (92% waste!)
Perplexity: 150 / 2048 = 7.3%
```

**FSQ Solution**:
```python
# All 625 codes available (mathematical guarantee)
Perplexity: 600-625 / 625 = 96-100%
```

### 2. Simpler Training

**Traditional VQ-VAE**:
```python
loss = recon_loss + β * commit_loss + codebook_loss
# Need to tune β, EMA decay, codebook reset, etc.
```

**FSQ**:
```python
loss = recon_loss  # That's it!
# No commitment loss, no EMA updates, no tuning
```

### 3. Deterministic & Reproducible

- Fixed mathematical mapping (not learned)
- Same codes across different training runs
- Easier to debug and analyze

### 4. Memory Efficient

**Traditional VQ-VAE**:
```python
# Store learned codebook: [2048, 512] × 4 bytes = 4MB
# Need EMA buffers: +4MB
```

**FSQ**:
```python
# Store levels: [5, 5, 5, 5] = 16 bytes
# Codebook computed on-the-fly
```

---

## Usage Examples

### 1. Encode Motion to Tokens

```python
import torch
from mGPT.archs import FSQVae

# Load model
model = FSQVae(nfeats=133, fsq_levels=[5,5,5,5], code_dim=4)
model.eval()

# Input: SMPLX motion [B=1, T=100, D=133]
motion = torch.randn(1, 100, 133)

# Encode to discrete tokens
with torch.no_grad():
    tokens, _ = model.encode(motion)
    # tokens: [1, T'] where T' = T / (2^down_t)
    # tokens range: 0-624

print(f"Motion shape: {motion.shape}")
print(f"Token shape: {tokens.shape}")
print(f"Token range: {tokens.min()}-{tokens.max()}")
```

### 2. Decode Tokens to Motion

```python
# Decode tokens back to motion
with torch.no_grad():
    reconstructed = model.decode(tokens)
    # reconstructed: [1, T, 133]

print(f"Reconstructed shape: {reconstructed.shape}")
```

### 3. Full Forward Pass

```python
# Forward pass with reconstruction
recon, loss, perplexity = model(motion)

print(f"Reconstruction loss: {loss.item()}")
print(f"Perplexity: {perplexity.item()}/625")
print(f"Codebook utilization: {perplexity.item()/625*100:.1f}%")
```

### 4. Extract Continuous Embeddings (for contrastive learning)

```python
# Get continuous embeddings before quantization
embeddings = model.encode_continuous(motion)
# embeddings: [B, code_dim] - pooled over time

# Use for text-motion contrastive learning
text_emb = text_encoder(text)
similarity = torch.cosine_similarity(embeddings, text_emb)
```

### 5. Using FSQ-RVQ

```python
from mGPT.archs import FSQRVQVae

model = FSQRVQVae(
    nfeats=133,
    num_quantizers=3,
    fsq_levels=[5,5,5,5],
    code_dim=4
)

# Encode returns multi-stage tokens
tokens, _ = model.encode(motion)
# tokens: [B, T', 3] - 3 quantizer stages

# Decode (uses all 3 stages)
reconstructed = model.decode(tokens)
```

---

## Integration with Hierarchical RVQ-GPT

To use FSQ tokens with your existing hierarchical GPT:

### 1. Train FSQ-RVQ VAE

```bash
python train.py --cfg configs/deto_h2s_fsq_rvq.yaml
```

### 2. Extract Tokens

```python
# Extract FSQ tokens for GPT training
from mGPT.archs import FSQRVQVae

vae = FSQRVQVae.load_from_checkpoint("path/to/fsq_rvq.ckpt")
vae.eval()

for batch in dataloader:
    motion = batch['motion']  # [B, T, 133]
    tokens, _ = vae.encode(motion)  # [B, T', 3]

    # Save tokens for GPT training
    # tokens[:, :, 0] = Q0 (coarse)
    # tokens[:, :, 1] = Q1 (medium)
    # tokens[:, :, 2] = Q2 (fine)
```

### 3. Train Hierarchical GPT

Use the extracted tokens with your existing `HierarchicalRVQGPT` architecture.

---

## Troubleshooting

### Low Perplexity (< 200)

**Cause**: Codebook underutilization (unlikely with FSQ but check anyway)

**Solution**:
- Increase `fsq_levels` (e.g., [8,8,8,8])
- Check input data diversity
- Increase training steps

### High Reconstruction Loss

**Cause**: Insufficient codebook size or encoder capacity

**Solution**:
- Use larger FSQ levels: [8,8,8,8] (4096 codes)
- Increase encoder width: `width: 768`
- Use more quantizers: `num_quantizers: 6`

### NaN Loss

**Cause**: Numerical instability

**Solution**:
- FSQ uses `force_quantization_f32=True` by default (should prevent this)
- Check data normalization
- Reduce learning rate

---

## Optimization: Direct Encoder-FSQ Mapping

**NEW**: For better efficiency, you can skip the projection layers between encoder and FSQ.

### Standard FSQ (with projection)
```yaml
# configs/vq/fsq.yaml
output_emb_width: 512  # Encoder outputs 512 dims
code_dim: 4            # FSQ projects 512 -> 4 -> 512
use_projection: true   # (default)
```

**Flow**: Encoder(133→512) → FSQ Project(512→4) → FSQ Quantize(4) → FSQ Project(4→512) → Decoder(512→133)

### Optimized FSQ (direct mapping)
```yaml
# configs/vq/fsq_optimized.yaml
output_emb_width: 4    # Encoder outputs 4 dims directly
code_dim: 4            # FSQ uses 4 dims directly (no projection!)
use_projection: false  # Skip projection layers
```

**Flow**: Encoder(133→4) → FSQ Quantize(4) → Decoder(4→133)

**Benefits**:
- ✅ Fewer parameters (no projection layers)
- ✅ Faster training and inference
- ✅ Lower memory usage
- ✅ Same reconstruction quality (quantization is the bottleneck)

**When to use**:
- Use **optimized** for most cases (recommended)
- Use **standard** if you need higher-dimensional bottleneck features

---

## Testing Your FSQ Implementation

Run the comprehensive test suite:

```bash
python test_fsq.py
```

This tests:
- FSQ module correctness
- Forward/backward gradient flow
- Encode-decode consistency
- Codebook utilization (perplexity)
- Edge cases (batch size 1, various sequence lengths)
- Shape handling

**Expected output**:
```
✓ FSQ initialization
✓ FSQ forward shape
✓ FSQ quantization range
✓ FSQ encode-decode consistency
✓ FSQ perplexity calculation
✓ FSQVae gradient flow (STE)
...
All tests passed! ✨
```

---

## Next Steps

1. **Run Tests**: Verify implementation with `python test_fsq.py`

2. **Train FSQ VAE**: Start with optimized config
   ```bash
   python train.py --cfg configs/deto_h2s_fsq.yaml
   ```

3. **Compare Configs**: Try both standard and optimized versions
   - Standard: `configs/vq/fsq.yaml` (512-dim bottleneck)
   - Optimized: `configs/vq/fsq_optimized.yaml` (4-dim bottleneck)

4. **Evaluate**: Compare reconstruction quality vs. standard VQ-VAE

5. **Extract Tokens**: Use trained FSQ VAE to tokenize your dataset

6. **Train GPT**: Use FSQ tokens for text-to-motion generation

7. **Experiment**: Try different FSQ levels, RVQ stages, etc.

---

## References

- **FSQ Paper**: "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)
- **SignViP Documentation**: `/data/hwu/workspace/signvip/FSQ_VQVAE_DOCUMENTATION.md`
- **Original Implementation**: Adapted from SignViP's FSQ implementation

---

## Contact & Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the FSQ paper for theoretical details
3. Compare with SignViP documentation for reference implementation

Happy training! 🚀
