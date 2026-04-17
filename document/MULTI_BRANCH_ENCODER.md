# Multi-Branch Encoder for RVQ-VAE

Inspired by the **HoMi (Holistic Motion) Tokenizer** from MotionLLaMA
([arXiv:2411.17335](https://arxiv.org/html/2411.17335v1)), which uses separate
hand/torso encoders fused via MLP. We extend this to **4 body-part branches**
tailored for sign language, with multiple **lightweight attention-based fusion**
options.

---

## Motivation

Sign language motion is heterogeneous across body parts:

- **Hands**: high-frequency, fine-grained (finger articulation, handshape)
- **Torso/arms**: low-frequency, large-amplitude (signing space, posture)
- **Face/jaw**: grammatical markers (negation, questions, intensity)

A single shared encoder forces these very different dynamics into the same
latent space — high-frequency hand details can be drowned out by torso signal.
A part-specific multi-branch encoder lets each branch specialize.

---

## 133D Feature Layout (SMPL-X, upper body)

```
Original 179D SMPL-X:
  root_pose(3) + body_pose(63) + lhand(45) + rhand(45)
  + jaw(3) + shape(10) + expr(10)

Drop lower body (36D) + shape (10D) → 133D:

  ┌─────────────────┬──────────────────┬──────────────────┬──────────────┐
  │  body [0:30]    │  lhand [30:75]   │  rhand [75:120]  │ head[120:133]│
  │  30D            │  45D             │  45D             │ 13D          │
  │  10 joints × 3  │  15 joints × 3   │  15 joints × 3   │ jaw(3)+ex(10)│
  └─────────────────┴──────────────────┴──────────────────┴──────────────┘
```

| Branch | Slice          | Dim |
|--------|----------------|-----|
| body   | `[0:30]`       | 30  |
| lhand  | `[30:75]`      | 45  |
| rhand  | `[75:120]`     | 45  |
| head   | `[120:133]`    | 13  |

---

## Pipeline

```
                        Input Motion [B, T, 133]
                                │
                          permute(0,2,1)
                                │
                         [B, 133, T]
                                │
               ┌────────────────┼────────────────┐──────────┐
               │                │                │          │
         ┌─────┴─────┐   ┌─────┴─────┐   ┌──────┴──────┐  ┌──┴───┐
         │  body     │   │  lhand    │   │  rhand      │  │ head │
         │ [0:30]    │   │ [30:75]   │   │ [75:120]    │  │[120:]│
         │ 30D       │   │ 45D       │   │ 45D         │  │ 13D  │
         └─────┬─────┘   └─────┬─────┘   └──────┬──────┘  └──┬───┘
               │               │                │             │
         ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
         │  Encoder  │  │  Encoder  │  │  Encoder  │  │  Encoder  │
         │ Conv1d+   │  │ Conv1d+   │  │ Conv1d+   │  │ Conv1d+   │
         │ ResNet1D  │  │ ResNet1D  │  │ ResNet1D  │  │ ResNet1D  │
         │ 30→512    │  │ 45→512    │  │ 45→512    │  │ 13→512    │
         │ 4x↓ time  │  │ 4x↓ time  │  │ 4x↓ time  │  │ 4x↓ time  │
         └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
               │              │              │              │
          [B,512,T']     [B,512,T']     [B,512,T']     [B,512,T']
               │              │              │              │
               └──────────────┴──────┬───────┴──────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │   Fusion Module     │
                          │   (one of 3 types)  │
                          └──────────┬──────────┘
                                     │
                              [B, 512, T']
                                     │
                          ┌──────────┴──────────┐
                          │   ResidualVQ × 3    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────┴──────────┐
                          │   Decoder (单分支)  │
                          │   512 → 133, 4x↑    │
                          └──────────┬──────────┘
                                     │
                            Output [B, T, 133]
```

---

## Fusion Modules

### Option A: MLP Fusion (default, baseline)

```
concat(branches) [B, K*512, T']
    → permute → [B, T', K*512]
    → Linear(K*512 → 512)
    → ReLU
    → Linear(512 → 512)
    → permute → [B, 512, T']
```

- **Params** (K=4, width=512): ≈1.32M
- **Pros**: simple, no inductive bias
- **Cons**: heaviest, no part-aware reasoning

---

### Option B: Dynamic Temporal Branch Attention (方案 2)

> Per-timestep softmax weighting over branches.
> Inspired by Dynamic Convolution (Chen et al., CVPR 2020).

```
concat(branches) [B, K*C, T']
    → permute → [B, T', K*C]
    → FC1(K*C → d) + ReLU      # d = 32
    → FC2(d → K)
    → softmax(dim=-1)            # [B, T', K]   per-timestep branch weights
                ↓
    stack(branches, dim=-1) [B, C, T', K]
    Y = Σ_i  w_i(t) × F_i(t)    # [B, C, T']
```

**Params** (C=512, K=4, d=32):
- FC1: `2048 × 32 + 32 = 65,568`
- FC2: `32 × 4 + 4 = 132`
- **Total ≈ 65.7K**

**Story**: At each timestep the model dynamically routes attention to the most
informative body part — hands during active signing, face during grammatical
markers, torso during transitions. The temporal attention map is directly
interpretable and visualizable.

---

### Option C: Dual-Axis Selective Fusion (方案 4)

> Combines SK-Net-style channel attention + dynamic temporal attention.
> Inspired by Selective Kernel Networks (Li et al., CVPR 2019) +
> Dynamic Convolution (Chen et al., CVPR 2020).

```
                          branches: K × [B, C, T']
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                                                      │
   ── Channel branch (SK) ──                          ── Temporal branch ──
            │                                                      │
   U = Σ_i F_i      [B, C, T']                concat(branches)  [B, K*C, T']
            │                                                      │
   s = GAP(U)       [B, C]                       permute → [B, T', K*C]
            │                                                      │
   z = ReLU(W_down s)   [B, d]                  FC1(K*C → d) + ReLU
            │                                                      │
   ch_logits_i = W_up_i(z)  [B, C]                  FC2(d → K)
   stack →  [B, C, K]                              t_logits  [B, T', K]
            │                                                      │
            └──────────────┬───────────────────────────────────────┘
                           │
        ch_logits.unsqueeze(2) + t_logits.unsqueeze(1)
                       [B, C, T', K]
                           │
                    softmax(dim=-1)
                           │
        w[c, t, i] = "branch i's weight for channel c at time t"
                           │
        stack(branches, dim=-1) [B, C, T', K]
                           │
        Y = Σ_i w_i ⊙ F_i        [B, C, T']
```

**Params** (C=512, K=4, reduction=32, d=16):
- `ch_down`: `512 × 16 + 16 = 8,208`
- `ch_ups` (4): `4 × (16 × 512 + 512) = 34,816`
- `t_fc1`: `2048 × 16 + 16 = 32,784`
- `t_fc2`: `16 × 4 + 4 = 68`
- **Total ≈ 75.9K**

**Story**: Each `(channel, timestep)` pair gets its own branch weight.
- *Channel axis*: which feature channels prefer which body part?
- *Time axis*: which timesteps emphasize which body part?
- The two attention signals are added and softmax'd for joint normalization,
  giving a `[B, C, T', K]` attention tensor — the most expressive lightweight
  scheme.

---

## Comparison

| Fusion         | Params  | Channel-aware | Temporal-aware | Story strength |
|----------------|---------|---------------|----------------|----------------|
| `mlp` (A)      | ~1.32M  | implicit      | no             | weakest        |
| `temporal` (B) | ~66K    | no            | **yes**        | strong + viz   |
| `dual_axis` (C)| ~76K    | **yes**       | **yes**        | strongest      |

Both attention variants are **20× smaller** than the MLP fusion.

---

## Files Modified

| File | Change |
|------|--------|
| `mGPT/archs/mgpt_vq.py` | + `DynamicTemporalBranchAttention` class |
|                         | + `DualAxisSelectiveFusion` class |
|                         | + `MultiBranchEncoder` class with `fusion_type` arg |
| `mGPT/archs/mgpt_rvq.py`| + `multi_branch`, `branch_slices`, `fusion_type` params on `RVQVae` |

All downstream classes (`RVQVaeLGVQ`, `RVQVaeLGVQSpeech`) inherit `RVQVae` and
forward `**kwargs`, so no change is required there.

The decoder remains a single-branch `Decoder` (matching HoMi). The
`ResidualVQ` quantizer is unchanged.

---

## Usage

In any VQ config (e.g. `configs/vq/h2s_rvq_3_lgvq_whisper.yaml`):

```yaml
target: mGPT.archs.mgpt_rvq_lgvq_speech.RVQVaeLGVQSpeech
params:
  # ... existing params unchanged ...

  # Enable multi-branch encoder
  multi_branch: true

  # Pick a fusion mechanism:
  #   'mlp'       - 2-layer MLP on concat (default, ~1.32M params)
  #   'temporal'  - Dynamic Temporal Branch Attention (~66K params)
  #   'dual_axis' - Dual-Axis Selective Fusion (~76K params)
  fusion_type: dual_axis

  # branch_slices: optional. If omitted, uses the default 4-part layout above.
```

To override branch slices (e.g. merge head into body):

```yaml
params:
  multi_branch: true
  branch_slices:
    body:  [0, 30]
    lhand: [30, 75]
    rhand: [75, 120]
    head:  [120, 133]
```

---

## Validation Checklist

1. Output shape matches original `Encoder`: `[B, output_emb_width, T']`
2. `RVQVae.encode/decode/forward` all unchanged from caller's perspective
3. `RVQVaeLGVQ` / `RVQVaeLGVQSpeech` work without modification
4. Train and compare against single-branch baseline:
   - Overall MPJPE
   - Hand-specific MPJPE (key for sign language)
   - Codebook perplexity

---

## References

- **HoMi Tokenizer** (MotionLLaMA): Wu et al., *arXiv:2411.17335*, 2024 — dual-branch encoder for hand/torso.
- **Selective Kernel Networks**: Li et al., *CVPR 2019* — channel attention across kernels.
- **Dynamic Convolution**: Chen et al., *CVPR 2020* — input-dependent attention over kernels.
- **ParCo**: Part-Coordinated text-to-motion generation, *arXiv 2024* — per-part VQ + cross-attention coordination.
- **PST-Transformer**: Part-level Spatial-Temporal Transformer for skeleton recognition.
