# MoMask Training Guide for How2Sign

MoMask is a three-stage generative model for text-to-motion generation based on masked modeling and residual vector quantization.

## Architecture Overview

```
Text → [Stage 2: Masked Transformer] → Q0 tokens
     → [Stage 3: Residual Transformer] → Q1-Q2 tokens
     → [Stage 1: RVQ-VAE Decoder] → Motion
```

## Stage 1: RVQ-VAE Training

**Purpose**: Learn to compress motion into discrete tokens using 3-layer Residual Vector Quantization.

**Config**: `configs/momask_h2s_stage1.yaml`

**Training Command**:
```bash
python train.py --cfg configs/momask_h2s_stage1.yaml --nodebug
```

**Key Parameters**:
- `num_quantizers: 3` - Three layers of quantization (Q0, Q1, Q2)
- `code_num: 512` - 512 tokens per codebook
- `code_dim: 512` - 512-dimensional token embeddings
- Batch size: 32
- Epochs: 500
- Learning rate: 2e-4

**Expected Output**:
- Checkpoint: `experiments/mgpt/MoMask_H2S_Stage1/checkpoints/best.ckpt`
- Metrics: Motion reconstruction accuracy (MPJPE, FID, etc.)

**Validation**:
- Check reconstruction quality on validation set
- Monitor commitment loss and perplexity
- Ensure codebook utilization is high (>80%)

---

## Stage 2: Masked Transformer Training

**Purpose**: Learn to generate coarse motion tokens (Q0) from text using iterative masked prediction.

**Prerequisites**:
- Trained RVQ-VAE from Stage 1
- Update `PRETRAINED_VAE` path in config

**Config**: `configs/momask_h2s_stage2.yaml`

**Training Command**:
```bash
python train.py --cfg configs/momask_h2s_stage2.yaml --nodebug
```

**Key Parameters**:
- `latent_dim: 384` - Transformer hidden size
- `num_layers: 8` - 8 transformer layers
- `num_heads: 6` - 6 attention heads
- `cond_drop_prob: 0.1` - 10% CFG dropout
- Batch size: 32
- Epochs: 150
- Learning rate: 1e-4

**Training Strategy**:
- VAE is frozen (from Stage 1)
- BERT-style masking: 88% MASK, 10% random, 2% unchanged
- Cosine noise schedule for masking ratio
- Uses CLIP ViT-B/32 for text encoding

**Expected Output**:
- Checkpoint: `experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt`
- Metrics: Cross-entropy loss, token prediction accuracy

**Validation**:
- Monitor training loss and accuracy
- Should reach >70% token prediction accuracy
- Check generated Q0 tokens make sense

---

## Stage 3: Residual Transformer Training

**Purpose**: Refine motion by predicting residual tokens (Q1, Q2) conditioned on previous layers.

**Prerequisites**:
- Trained RVQ-VAE from Stage 1
- Update `PRETRAINED_VAE` path in config

**Config**: `configs/momask_h2s_stage3.yaml`

**Training Command**:
```bash
python train.py --cfg configs/momask_h2s_stage3.yaml --nodebug
```

**Key Parameters**:
- `latent_dim: 384` - Transformer hidden size
- `num_layers: 8` - 8 transformer layers
- `num_heads: 6` - 6 attention heads
- `cond_drop_prob: 0.2` - 20% CFG dropout (higher than Stage 2)
- `share_weight: true` - Share embedding/projection weights
- Batch size: 32
- Epochs: 150
- Learning rate: 1e-4

**Training Strategy**:
- VAE is frozen (from Stage 1)
- Randomly sample which quantizer layer to predict (Q1 or Q2)
- Condition on cumulative sum of previous layer embeddings
- Quantizer layer embedding added as input

**Expected Output**:
- Checkpoint: `experiments/mgpt/MoMask_H2S_Stage3/checkpoints/best.ckpt`
- Metrics: Motion reconstruction with GT Q0 (MPJPE, etc.)

**Validation**:
- Uses ground truth Q0, predicts Q1-Q2
- Should improve motion quality over Q0-only
- Monitor reconstruction metrics

---

## Full Inference Pipeline

**Purpose**: Generate motion from text using all three trained components.

**Prerequisites**:
- All three stages trained
- Update checkpoint paths in model loading code

**Config**: `configs/momask_h2s_inference.yaml`

**Inference Command**:
```bash
python test.py --cfg configs/momask_h2s_inference.yaml
```

**Generation Hyperparameters**:
```yaml
timesteps: 10              # Iterative unmasking steps
cond_scale_mask: 4.0      # CFG scale for Masked Transformer
cond_scale_res: 2.0       # CFG scale for Residual Transformer
temperature: 1.0          # Sampling temperature
topk_filter_thres: 0.9    # Top-k filtering
```

**Generation Pipeline**:
1. Encode text with CLIP
2. Start with all MASK tokens
3. Iteratively unmask over 10 steps → Q0 tokens
4. Auto-regressively predict Q1, Q2 from Q0
5. Decode all tokens with RVQ-VAE → motion

**Tuning Tips**:
- Higher `cond_scale_mask` → more text-aligned but less diverse
- Higher `temperature` → more random/diverse
- More `timesteps` → better quality but slower

---

## Training Timeline

**Total GPU Hours** (approximate, 3x A100):
- Stage 1: ~200 GPU hours (500 epochs)
- Stage 2: ~60 GPU hours (150 epochs)
- Stage 3: ~60 GPU hours (150 epochs)
- **Total**: ~320 GPU hours

**Recommended Schedule**:
1. Week 1-2: Stage 1 (VAE)
2. Week 3: Stage 2 (Masked Transformer)
3. Week 4: Stage 3 (Residual Transformer)
4. Week 5: Evaluation and tuning

---

## Model Loading for Inference

You need to manually load the checkpoints for each stage. Example:

```python
# Load Stage 1 VAE
vae_ckpt = torch.load('experiments/mgpt/MoMask_H2S_Stage1/checkpoints/best.ckpt')
model.vae.load_state_dict(vae_ckpt['state_dict'], strict=False)

# Load Stage 2 Masked Transformer
mask_ckpt = torch.load('experiments/mgpt/MoMask_H2S_Stage2/checkpoints/best.ckpt')
model.mask_transformer.load_state_dict(mask_ckpt['state_dict'], strict=False)

# Load Stage 3 Residual Transformer
res_ckpt = torch.load('experiments/mgpt/MoMask_H2S_Stage3/checkpoints/best.ckpt')
model.res_transformer.load_state_dict(res_ckpt['state_dict'], strict=False)
```

---

## Troubleshooting

### Stage 1 Issues
- **Low codebook utilization**: Increase commitment loss weight
- **Poor reconstruction**: Check data normalization, increase epochs
- **NaN loss**: Lower learning rate, check input data

### Stage 2 Issues
- **Low accuracy**: Check VAE is properly frozen and loaded
- **OOM errors**: Reduce batch size or sequence length
- **CLIP not loading**: Install with `pip install git+https://github.com/openai/CLIP.git`

### Stage 3 Issues
- **No improvement over Q0**: Check weight sharing settings
- **Training unstable**: Lower learning rate, increase dropout
- **Slow training**: Use gradient accumulation

### Inference Issues
- **Poor quality**: Tune CFG scales and temperature
- **Out of distribution**: Check text encoder matches training
- **Slow generation**: Reduce timesteps (trade quality for speed)

---

## Differences from Original MoMask

1. **Dataset**: Adapted for How2Sign (133 joints) vs HumanML3D (263 features)
2. **Quantizers**: 3 layers vs original 6 layers (faster training)
3. **Codebook size**: 512 vs original 1024 (smaller vocab)
4. **Unit length**: 4 frames/token vs original varies

---

## Citation

If you use MoMask, please cite:
```bibtex
@article{guo2024momask,
  title={MoMask: Generative Masked Modeling of 3D Human Motions},
  author={Guo, Chuan and Zou, Yuxuan and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
  journal={arXiv preprint arXiv:2312.00063},
  year={2024}
}
```
