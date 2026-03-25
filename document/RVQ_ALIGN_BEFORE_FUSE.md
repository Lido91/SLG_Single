# RVQ-VAE Cumulative Alignment: Align Before Fuse

## 1. Motivation

### 1.1 Problem: RVQ Codebook Lacks Semantic Structure

Standard RVQ-VAE training only optimizes reconstruction loss + commitment loss. The codebook entries are organized purely by **reconstruction error minimization**, not by semantic meaning. This means:

- Q0 tokens don't inherently carry text/gloss semantics
- The Hierarchical GPT must learn text→motion mapping from scratch
- Q0 decoder needs to simultaneously learn: (1) what each codebook entry means semantically, and (2) how to arrange them temporally

### 1.2 Insight: Align Codebook with Text During VAE Training

If we add a text-alignment loss during VAE training, the codebook entries become **semantically structured**:

- Similar text → similar Q0 codes
- The GPT decoder only needs to learn **temporal arrangement**, not semantic mapping
- This is analogous to ALBEF's "Align before Fuse" principle: align unimodal representations before multimodal fusion

### 1.3 Why Cumulative Alignment

RVQ produces a coarse-to-fine hierarchy:
```
Q0:           coarse reconstruction
Q0+Q1:        medium reconstruction
Q0+Q1+Q2:     fine reconstruction
```

We align **each cumulative level** with text, ensuring:
- Q0 alone carries coarse semantic information
- Adding Q1 refines the semantic alignment
- Adding Q2 achieves full semantic alignment

This directly maps to Hierarchical GPT's generation chain:
```
Q0 decoder:  text → Q0 tokens    (benefits from Q0 semantic alignment)
Q1 decoder:  text+Q0 → Q1 tokens (benefits from Q0+Q1 alignment)
Q2 decoder:  text+Q0+Q1 → Q2     (benefits from full alignment)
```

---

## 2. Method

### 2.1 Architecture

```
                        RVQ-VAE with Cumulative Alignment
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Motion [B,T,133]                     Text (list of strings)     │
│       │                                      │                   │
│       ▼                                      ▼                   │
│   Encoder                            Frozen CLIP Encoder         │
│       │                                      │                   │
│       ▼                                      ▼                   │
│  x_enc [B,512,T']                    text_emb [B,512]            │
│       │                                      │                   │
│       ▼                                      ▼                   │
│   ResidualVQ                           text_proj [B,256]         │
│   ┌─────────┐                                │                   │
│   │ Q0(x)→z0│── pool ── proj_0 ── ITC ──────┤                   │
│   │ r=x-z0  │                                │                   │
│   │ Q1(r)→z1│                                │                   │
│   │ z0+z1   │── pool ── proj_1 ── ITC ──────┤                   │
│   │ r=r-z1  │                                │                   │
│   │ Q2(r)→z2│                                │                   │
│   │ z0+z1+z2│── pool ── proj_2 ── ITC ──────┘                   │
│   └─────────┘                                                    │
│       │                                                          │
│       ▼                                                          │
│   Decoder                                                        │
│       │                                                          │
│       ▼                                                          │
│  Motion_recon [B,T,133]                                          │
│                                                                  │
│  Total Loss = L_recon + λ_commit * L_commit + λ_align * L_align  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 ITC Loss (Image-Text Contrastive)

Standard symmetric InfoNCE, same as CLIP:

```
Given: motion_emb [B, 256], text_emb [B, 256] (both L2-normalized)

Similarity matrix: S = motion_emb @ text_emb.T / tau    [B, B]

Labels: diagonal = positive pairs (each sample's own text)

Loss = ( CrossEntropy(S, labels) + CrossEntropy(S.T, labels) ) / 2
```

Key advantage over VQ-Style cluster-based contrastive:

| Property | VQ-Style (cluster) | ITC (CLIP-style) |
|----------|-------------------|-------------------|
| Positive pairs per batch | 0~31 (depends on cluster collision) | B (every sample has one) |
| Semantic granularity | 64 coarse clusters | Continuous text embedding |
| Offline preprocessing | Required (cluster_labels.json) | None |
| Negative pairs | Same-batch, same-cluster excluded | All other samples in batch |

### 2.3 Cumulative Alignment Loss

```python
L_align = (1/N) * Σ_{i=0}^{N-1} ITC(pool(Σ_{j=0}^{i} z_qj), text_emb)
```

Where:
- N = num_quantizers (3)
- z_qj = quantized output of j-th quantizer [B, 512, T']
- pool = mean over time dimension: [B, 512, T'] → [B, 512]
- Each level has its own projection head proj_i: 512 → 256

### 2.4 Total Loss

```
L_total = λ_recon * L_recon       (reconstruction, L1 Smooth)
        + λ_commit * L_commit     (VQ commitment)
        + λ_align * L_align       (cumulative text alignment)

Default: λ_recon=1.0, λ_commit=0.02, λ_align=0.01
```

---

## 3. How This Helps Hierarchical GPT

### 3.1 Without Alignment (Current)

```
VAE codebook: organized by reconstruction error only
  → Token 42 and Token 187 might encode similar hand shapes for different sentences
  → No semantic clustering in codebook space

GPT task: text → Q0 token sequence
  → Must learn BOTH: (1) text→semantics mapping AND (2) temporal dynamics
  → Hard optimization problem
```

### 3.2 With Cumulative Alignment (Proposed)

```
VAE codebook: organized by BOTH reconstruction AND text semantics
  → Tokens for "thank you" cluster together in codebook space
  → Tokens for "hello" cluster in a different region
  → Semantic structure is baked into the codebook

GPT task: text → Q0 token sequence
  → Semantics already encoded in codebook → just learn temporal arrangement
  → Easier optimization → faster convergence, better generation quality
```

### 3.3 Per-Level Benefits

| GPT Decoder | What It Generates | How Alignment Helps |
|-------------|-------------------|---------------------|
| Q0 decoder | Coarse motion tokens from text | Q0 codebook already text-aligned → decoder focuses on temporal ordering |
| Q1 decoder | Refinement tokens from text+Q0 | Q0+Q1 space aligned → Q1 tokens add semantically coherent detail |
| Q2 decoder | Fine tokens from text+Q0+Q1 | Full alignment → Q2 tokens add consistent fine-grained detail |

### 3.4 Quantitative Expectations

- **Faster GPT convergence**: Codebook semantic structure reduces the search space
- **Better text-motion consistency**: Generated motion more faithfully reflects input text
- **Improved partial decoding**: Even Q0-only decoding produces semantically meaningful motion (useful for real-time / low-latency generation)

---

## 4. Comparison with VQ-Style

| Aspect | VQ-Style | Cumulative Alignment |
|--------|----------|---------------------|
| Goal | Disentangle content (Q0) vs style (Q1+) | All levels carry text semantics |
| Q0 role | Text-independent (style/rhythm) | Coarse text-aligned content |
| Q1+ role | Text-dependent (content) | Refined text-aligned content |
| Loss on Q0 | MI minimization (push away from text) | ITC alignment (pull toward text) |
| Loss on Q1+ | Cluster-based contrastive | ITC alignment (cumulative) |
| Needs clustering | Yes (offline k-means) | No |
| Needs text encoder | No | Yes (frozen CLIP, ~400MB) |
| Positive pairs | Cluster collision dependent | Every sample (guaranteed) |
| Helps GPT? | Indirect (Q1+ has semantics) | Direct (all levels have semantics) |

---

## 5. Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `mGPT/archs/mgpt_rvq_vqstyle.py` | Rewrite: remove cluster logic, add CLIP encoder + projection heads + ITC loss |
| `mGPT/models/mgpt.py` | Pass `texts` instead of `names` to VAE |
| `mGPT/losses/mgpt.py` | Replace `vqstyle_con` + `vqstyle_mi` with single `vqstyle_align` |
| `configs/vq/h2s_rvq_3_vqstyle.yaml` | Update params: remove cluster, add align params |
| `configs/deto_h2s_rvq_3_vqstyle.yaml` | Update loss weights |

### Files Unchanged

| File | Reason |
|------|--------|
| `mGPT/archs/tools/residual_vq.py` | `return_per_quantizer` already implemented |
| `mGPT/archs/__init__.py` | Class name unchanged |
| `scripts/precompute_text_clusters.py` | No longer needed but kept for reference |

### New Hyperparameters

```yaml
# VQ-Style Alignment Parameters
lambda_align: 0.01          # Alignment loss weight
tau_align: 0.07             # InfoNCE temperature
align_proj_dim: 256         # Projection dimension for ITC
text_encoder_type: 'clip'   # Frozen text encoder
```

---

## 6. Training Pipeline

### Step 1: Train RVQ-VAE with Alignment

```bash
python train.py --cfg configs/deto_h2s_rvq_3_vqstyle.yaml
```

No preprocessing needed. CLIP encoder is loaded frozen inside the VAE.

### Step 2: Train Hierarchical GPT (unchanged)

```bash
python train.py --cfg configs/deto_h2s_rvq_hierarchical_3layer.yaml
```

Uses the aligned RVQ-VAE checkpoint. The GPT training code doesn't change — it just benefits from better codebook structure.

### Monitoring

| Wandb Metric | Meaning |
|-------------|---------|
| `recons/feature/train` | Reconstruction loss |
| `vq/commit/train` | Commitment loss |
| `vqstyle/align/train` | Cumulative ITC alignment loss |
| `total/train` | Total weighted loss |

---

## 7. Theoretical Justification

### Information-Theoretic View

Standard RVQ-VAE minimizes: `I(X; X_recon)` — mutual information between input and reconstruction.

With alignment, we additionally maximize: `I(Z_q; T)` — mutual information between quantized codes and text.

This creates a bottleneck where the codebook must encode **both** reconstruction-relevant and text-relevant information, resulting in semantically structured discrete representations.

### Connection to ALBEF

ALBEF's key finding: aligning unimodal representations before cross-modal fusion significantly improves downstream task performance. Our approach applies the same principle:

1. **Align** (VAE stage): Align motion codebook with text via ITC
2. **Fuse** (GPT stage): Fuse text conditioning with motion generation via cross-attention

The alignment stage ensures the motion "language" (codebook) is compatible with natural language, making the fusion stage more effective.

---

## 8. References

- **ALBEF**: Li et al., "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation", NeurIPS 2021
- **VQ-Style**: "VQ-Style: Disentangling Style from Content in Vector Quantized Representations", arXiv 2602.02334
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
