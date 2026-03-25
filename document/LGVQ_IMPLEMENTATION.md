# LG-VQ: Language-Guided Codebook Learning for Motion RVQ-VAE

## Overview

Migration of **LG-VQ** (Language-Guided Codebook Learning, NeurIPS 2023) from image VQGAN to our 1D motion RVQ-VAE. The key idea is to make the quantized codebook semantically meaningful by supervising it with three text-guided alignment losses during VQ-VAE training.

**Source**: `/data/hwu/workspace/LG-VQ-language-guided-codebook-learning/`
**Reference**: `document/15265_MotionBind_Multi_Modal_H.pdf` (related), LG-VQ paper

## Architecture

```
Motion [B, T, 133]
    | preprocess
    v
[B, 133, T]
    | Encoder (Conv1D + ResNet1D, stride_t=2, down_t=2)
    v
[B, 512, T']     (T' = T/4)
    | ResidualVQ (3 quantizers, 512 codes x 512 dim)
    v
x_quantized [B, 512, T']  ──────────────────────────────────┐
    | Decoder                                                |
    v                                                        |
x_out [B, T, 133]  (reconstruction)                         |
                                                             |
Text ──> CLIP (frozen) ──> all_text_features [B, 30, 512]   |
                      └──> last_text_feature [B, 512]        |
                                                             v
                                              ┌──────────────────────────┐
                                              │  LG-VQ Alignment Modules │
                                              │                          │
                                              │  1. MaskTransformer1D    │
                                              │     encoder: quant→CLS+T │
                                              │     decoder: text×motion │
                                              │                          │
                                              │  2. NCE Loss (CLS↔EOS)  │
                                              │  3. Mask Prediction Loss │
                                              │  4. WRS Relation Loss    │
                                              └──────────────────────────┘
```

## Three Alignment Modules

### 1. NCE Loss (`global_infor_sup`)
- **Input**: Global motion CLS token [B, 512] from QuantTransformer1D, CLIP EOS text feature [B, 512]
- **Mechanism**: L2-normalize both, compute InfoNCE with all batch pairs as negatives
- **Purpose**: Align global motion representation with global text meaning
- **Weight**: `nce_weight = 0.001`

### 2. Mask Prediction (`mask_prediction`)
- **Input**: Quantized motion [B, 512, T'], CLIP per-token features [B, 30, 512]
- **Mechanism**:
  1. Randomly mask ~15% of valid text tokens (replace with learnable `mask_learned_parameter`)
  2. Encode motion via `QuantTransformer1D` → CLS + temporal tokens [B, T'+1, 512]
  3. Cross-attend: masked text queries attend to motion tokens via `MaskTransformerDecoder1D`
  4. Predict masked token indices via `MlmLayer` (FC→GELU→LN→matmul with CLIP vocabulary)
  5. Label-smoothed cross-entropy loss on masked positions only
- **Purpose**: Force codebook to capture fine-grained token-level semantics
- **Weight**: `mask_weight = 0.01`
- **Also produces**: `vision_global_token` (CLS) for NCE, `vision_tokens` for WRS

### 3. WRS Loss (`wrs_relation_sup`)
- **Input**: Vision temporal tokens [B, T', 512], quantized features [B, 512, T'], text tokens [B, 30, 512]
- **Mechanism**:
  1. Project quantized features to text space via `WrsLayer` (FC→GELU→LN, 512→512)
  2. Find nearest vision token for each text token (cosine similarity)
  3. Build text relation matrix Q = text_feat @ text_feat^T [B, 30, 30]
  4. Build code relation matrix P = matched_code_feat @ matched_code_feat^T [B, 30, 30]
  5. MSE(P, Q.detach()) — align code structure to text structure
- **Purpose**: Preserve pairwise semantic relationships in the codebook
- **Weight**: `wrs_weight = 0.0001`

## Key Adaptations from Image (2D) to Motion (1D)

| Aspect | LG-VQ (Image) | Ours (Motion) |
|--------|---------------|---------------|
| Input to quantizer | [B, 256, 16, 16] (2D spatial) | [B, 512, T'] (1D temporal) |
| Quantization | Single VQ (1024 codes, 256-dim) | Residual VQ (3 stages, 512 codes, 512-dim) |
| QuantTransformer | Fixed 16x16=256 spatial + 1 CLS positional embeddings | Variable T' temporal + 1 CLS, learnable pos embeddings up to `max_motion_len` |
| Transformer impl | Custom CLIP-style blocks | `nn.TransformerEncoder` / `nn.MultiheadAttention` (PyTorch native) |
| WRS gathering | Python for-loop over text positions | `torch.gather` (vectorized) |
| Text encoder | CLIP ViT-B/32 with custom `clip.tokenize(context_length=30)` | CLIP ViT-B/32 with standard `clip.tokenize(truncate=True)` + manual truncation to 30 with EOS preservation |
| Decoder mask | Custom ScaleDotProductAttention with mask | `nn.MultiheadAttention` with `key_padding_mask` |
| Discriminator | Has adversarial loss (GAN training) | No discriminator (pure reconstruction + LG losses) |
| LG-VQ loss applied to | Single quantized output | Summed RVQ output (after all quantizer stages) |

## File Structure

### New Files
- `mGPT/archs/mgpt_rvq_lgvq.py` — Main implementation (RVQVaeLGVQ + all sub-modules)
- `configs/vq/h2s_rvq_3_lgvq.yaml` — VQ architecture config
- `configs/deto_h2s_rvq_3_lgvq.yaml` — Full training config

### Modified Files
- `mGPT/archs/__init__.py` — Registered `RVQVaeLGVQ`
- `mGPT/losses/mgpt.py` — Added `vqstyle_lgvq` loss slot with `LAMBDA_LGVQ`
- `mGPT/models/mgpt.py` — Added LG-VQ branch in `train_vae_forward` (detected via `hasattr(self.vae, 'nce_weight')`)

## Module Details

### QuantTransformer1D
```
Input: [B, code_dim, T']
    → permute to [B, T', code_dim]
    → prepend CLS token [B, T'+1, code_dim]
    → add positional embeddings (learnable, max_motion_len+1)
    → LayerNorm
    → TransformerEncoder (2 layers, 4 heads, GELU, dim_feedforward=code_dim*4)
    → LayerNorm
    → Linear projection (code_dim → text_dim via learned matrix)
Output: [B, T'+1, text_dim] (return_all=True) or [B, text_dim] (CLS only)
```

### MaskTransformerDecoder1D
```
For each of 2 layers:
    1. Self-attention on text tokens (with key_padding_mask for padding)
    2. Cross-attention: text (Q) attends to motion encoder output (K, V)
    3. FFN: Linear → ReLU → Dropout → Linear
    Each with residual connection + LayerNorm + Dropout
```

### MlmLayer
```
Input: [B, N_text, 512]
    → FC(512, 512) → GELU → LayerNorm
    → matmul with CLIP word embeddings [vocab_size, 512]^T
    → + learnable bias [1, 1, vocab_size]
Output: [B, N_text, vocab_size] logits
```

## Loss Integration

Total VAE loss:
```
loss = LAMBDA_FEATURE * L1_smooth(reconstruction)
     + LAMBDA_COMMIT * commitment_loss
     + LAMBDA_LGVQ * (nce_weight * NCE + mask_weight * MLM + wrs_weight * WRS)
```

Default config weights:
- `LAMBDA_FEATURE = 1.0`
- `LAMBDA_COMMIT = 0.02`
- `LAMBDA_LGVQ = 1.0` (outer weight in loss registry)
- `nce_weight = 0.001`, `mask_weight = 0.01`, `wrs_weight = 0.0001` (inner weights in VQ config)

## CLIP Text Processing Details

1. Tokenize with `clip.tokenize(texts, truncate=True)` → [B, 77]
2. Truncate to 30 tokens, ensuring EOS (token 49407) is preserved at position 29 if cut off
3. Run CLIP text encoder manually (token_embedding → positional → transformer → ln_final) to get per-token features [B, 30, 512]
4. Extract EOS feature via `argmax` on token IDs → global text representation [B, 512]
5. During training: randomly mask ~15% of inner tokens (skip BOS at 0, EOS at end)

## Training

```bash
python train.py --config configs/deto_h2s_rvq_3_lgvq.yaml
```

- CLIP is frozen (no gradients)
- LG-VQ modules (MaskTransformer, WrsLayer, MlmLayer, mask_learned_parameter) are trained alongside encoder/decoder/quantizer
- Text losses only applied during training (not validation)
- Batch size > 1 required for NCE loss (contrastive needs negatives)

## Relation to Other Text-Aligned VAEs in This Codebase

| VAE Variant | Text Integration | Alignment Type |
|-------------|-----------------|----------------|
| `RVQVae` (base) | None | Pure reconstruction |
| `RVQVaeAlign` | CLIP global feature | Single InfoNCE (multi-scale encoder features) |
| `RVQVaeVQStyle` | Cluster labels (not text) | Contrastive on style codes + MI on content |
| **`RVQVaeLGVQ`** | **CLIP per-token features** | **NCE + Masked prediction + Relation alignment** |

The key difference from `RVQVaeAlign` is that LG-VQ operates at the **token level** (per-token cross-attention and relation matching), not just the global level. This provides finer-grained semantic guidance to the codebook.
