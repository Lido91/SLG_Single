# LaMP Integration Plan for MotionGPT

## Current State Analysis

### What Already Exists in MotionGPT
| Component | File | Status |
|-----------|------|--------|
| LaMP model skeleton | `mGPT/archs/lamp/lamp_model.py` | Implemented (429 lines) |
| QFormer base class | `mGPT/archs/lamp/qformer_base.py` | Implemented (146 lines) |
| Module init | `mGPT/archs/lamp/__init__.py` | Done |
| Loss registration | `mGPT/losses/mgpt.py` (lines 53-62, 140-149) | Done (lamp_ptc, lamp_ptm, lamp_lm, lamp_gen) |
| Training forward | `mGPT/models/mgpt.py:415-423` (`train_lamp_forward`) | Done |
| `allsplit_step` routing | `mGPT/models/mgpt.py:524-526` | Done (stage=="lamp") |
| Config file | `configs/lamp_h2s.yaml` | Done |
| VQ-VAE backend | `mGPT/archs/mgpt_rvq.py` (RVQVae) | Done (frozen encoder reused) |
| Data module | `mGPT/data/H2S.py` (H2SDataModule) | Done |

### What is MISSING
| Component | Source in LaMP | Target in MotionGPT | Priority |
|-----------|---------------|---------------------|----------|
| **QFormer.py** (BERT with cross-attention) | `LaMP/models/lamp/QFormer.py` | `mGPT/archs/lamp/QFormer.py` | **CRITICAL** |
| **QFormer_output.py** (output dataclass) | `LaMP/models/lamp/QFormer_output.py` | `mGPT/archs/lamp/QFormer_output.py` | **CRITICAL** |
| **basemodel.py** (gather utilities) | `LaMP/models/lamp/basemodel.py` | `mGPT/archs/lamp/basemodel.py` | **CRITICAL** |
| Validation forward for LaMP | N/A (new) | `mGPT/models/mgpt.py` | HIGH |
| LaMP-specific metrics | `LaMP/utils/metrics.py` | `mGPT/metrics/` | MEDIUM |
| BERT pretrained weights | HuggingFace `bert-base-uncased` | Auto-downloaded | LOW (auto) |

---

## Integration Plan

### Phase 1: Copy Missing Dependencies (CRITICAL)

These 3 files are imported by `lamp_model.py` and `qformer_base.py` but do not exist yet. Without them, the LaMP module will fail on import.

#### Step 1.1: Copy `QFormer.py`
- **Source**: `/home/student/hwu/Workplace/LaMP/models/lamp/QFormer.py`
- **Target**: `/home/student/hwu/Workplace/MotionGPT/mGPT/archs/lamp/QFormer.py`
- **Description**: Full BERT implementation with cross-attention layers. This is the core QFormer architecture (~800+ lines) adapted from Salesforce BLIP-2.
- **Key classes**: `BertConfig`, `BertLMHeadModel`, `BertSelfAttention` (with cross-attention), `BertEmbeddings`
- **Modifications needed**: None expected - this is a self-contained BERT implementation. Just ensure imports are correct (uses standard `transformers` and `torch`).

#### Step 1.2: Copy `QFormer_output.py`
- **Source**: `/home/student/hwu/Workplace/LaMP/models/lamp/QFormer_output.py`
- **Target**: `/home/student/hwu/Workplace/MotionGPT/mGPT/archs/lamp/QFormer_output.py`
- **Description**: Dataclass for QFormer outputs. Contains fields like `loss`, `loss_ptc`, `loss_ptm`, `loss_lm`, `loss_gen`.
- **Modifications needed**: None.

#### Step 1.3: Copy `basemodel.py`
- **Source**: `/home/student/hwu/Workplace/LaMP/models/lamp/basemodel.py`
- **Target**: `/home/student/hwu/Workplace/MotionGPT/mGPT/archs/lamp/basemodel.py`
- **Description**: Contains `all_gather_with_grad` and `concat_all_gather` utilities for distributed contrastive learning, plus the `BaseModel` class that `QFormer_Base` inherits from.
- **Modifications needed**: Verify that `BaseModel` here (LaMP's `nn.Module` subclass) does not conflict with MotionGPT's `BaseModel` (which is a `LightningModule`). The LaMP `BaseModel` is only used as a parent for `QFormer_Base` → `LaMP`, which is instantiated as `self.lm` inside `MotionGPT` (the Lightning module). **No conflict** — they live in different namespaces.

### Phase 2: Verify and Fix Import Chain

After copying, verify the import chain works:

```
mGPT/archs/lamp/__init__.py
  └── lamp_model.py (LaMP class)
        ├── qformer_base.py (QFormer_Base)
        │     ├── QFormer.py (BertConfig, BertLMHeadModel)
        │     └── basemodel.py (BaseModel for LaMP)
        ├── QFormer_output.py (QFormer_Output dataclass)
        └── basemodel.py (all_gather_with_grad, concat_all_gather)
```

**Verification**: After copying, run:
```python
from mGPT.archs.lamp import LaMP
```
If this imports without error, Phase 2 is complete.

### Phase 3: Verify VQ-VAE Encoder Compatibility

The LaMP model calls `self.vq_model.encoder` (line 79 of `lamp_model.py`) and `self.vq_model.encode()` (line 333).

**Current `lamp_model.py` assumptions**:
1. `self.vq_model.encoder(motion_input)` → returns `[B, code_dim, T']` (continuous features)
2. `self.vq_model.encode(motion)` → returns `(codes, _)` where codes are `[B, T', num_quantizers]`

**Check against `mGPT/archs/mgpt_rvq.py` (RVQVae)**:
- `RVQVae.encoder` is a `Conv1D`-based encoder → outputs `[B, code_dim, T']` ✓
- `RVQVae.encode(motion)` → need to verify return format matches `(codes, ...)` with shape `[B, T', num_quantizers]`

**Action**: Read `RVQVae.encode()` and confirm:
- Output shape matches expectations
- First quantizer codes (`[:, :, 0]`) are valid discrete token IDs in range `[0, 512)`

### Phase 4: Validate Training Pipeline Integration

The training pipeline is already wired up:

```
configs/lamp_h2s.yaml
  → TRAIN.STAGE: "lamp"
  → model.target: mGPT.models.mgpt.MotionGPT
  → lm: mGPT.archs.lamp.lamp_model.LaMP
  → motion_vae: mGPT.archs.mgpt_rvq.RVQVae

MotionGPT.__init__():
  → self.vae = instantiate(motion_vae)  # RVQVae
  → self.lm = instantiate(lm)           # LaMP (BUT: needs vq_model param)

allsplit_step("train"):
  → stage == "lamp" → train_lamp_forward(batch)
  → LaMP.forward(motion, text) → 4 losses

GPTLosses(stage="lamp"):
  → lamp_ptc, lamp_ptm, lamp_lm, lamp_gen
```

**Issue to resolve**: The config has `lm.lamp.params.vq_model: ${vq.h2s_rvq_3}` which would instantiate a *separate* VQ model for LaMP. But `MotionGPT.__init__()` already instantiates `self.vae`. The LaMP model needs the *same* frozen VQ model.

**Solution**: Modify `MotionGPT.__init__()` to pass the already-instantiated `self.vae` to LaMP, OR modify the config so LaMP receives the VQ model reference properly. The cleanest approach:

```python
# In MotionGPT.__init__(), after self.vae is created:
if self.hparams.stage == "lamp":
    self.lm = LaMP(
        vq_model=self.vae,
        **lm_params  # other params from config
    )
```

This avoids double-instantiating the VQ-VAE.

### Phase 5: Add Validation Logic for LaMP (HIGH)

Currently there is **no validation forward** for the `lamp` stage in `allsplit_step`. Lines 532-618 handle val/test for `vae`, `lm_*`, and `lm_masked_t2m`, but not `lamp`.

**Options for LaMP validation**:

#### Option A: Contrastive Retrieval Metrics (Recommended)
Evaluate motion-text alignment quality using:
- **R-Precision (R@1, R@2, R@3)**: Given a motion, retrieve correct text from candidates
- **Matching Score**: Average cosine similarity between paired motion-text features

Implementation:
```python
def val_lamp_forward(self, batch):
    """LaMP validation: compute retrieval metrics."""
    motion = batch["motion"]
    texts = batch["text"]

    # Get features from LaMP
    text_feats = self.lm.encode_text(texts)     # [B, embed_dim]
    motion_feats = self.lm.encode_motion(motion)  # [B, embed_dim]

    # Compute similarity matrix
    sim_matrix = motion_feats @ text_feats.T  # [B, B]

    # R-precision: for each motion, check if correct text is in top-k
    # Matching score: diagonal of sim_matrix
    return {
        "sim_matrix": sim_matrix,
        "text_feats": text_feats,
        "motion_feats": motion_feats
    }
```

#### Option B: Validation Loss Only (Simpler)
Just run the same forward pass as training and log val losses:
```python
elif self.hparams.stage == "lamp" and split in ["val"]:
    rs_set = self.train_lamp_forward(batch)
    loss = self._losses['losses_' + split].update(rs_set)
```

**Recommendation**: Start with Option B for simplicity, add Option A later.

### Phase 6: Verify Data Pipeline Compatibility

The `H2SDataModule` batch provides:
```python
batch = {
    "motion": tensor [B, T, 133],  # Normalized motion features
    "text": list[str],             # Text descriptions
    "length": list[int],           # Frame lengths
    "name": list[str],             # Sample identifiers
    "src": list[str],              # Source dataset
}
```

LaMP's `forward(motion, text)` expects:
- `motion`: `[B, T, nfeats]` where `nfeats=133` ✓
- `text`: list of strings ✓

**Data pipeline is compatible.** No changes needed.

---

## Implementation Checklist

### Must Do (Blocking)
- [ ] Copy `QFormer.py` from LaMP to `mGPT/archs/lamp/`
- [ ] Copy `QFormer_output.py` from LaMP to `mGPT/archs/lamp/`
- [ ] Copy `basemodel.py` from LaMP to `mGPT/archs/lamp/`
- [ ] Verify import chain works: `from mGPT.archs.lamp import LaMP`
- [ ] Fix VQ model double-instantiation issue in `MotionGPT.__init__()`
- [ ] Verify `RVQVae.encode()` returns compatible format for generation loss

### Should Do (Important)
- [ ] Add validation forward for LaMP stage in `allsplit_step`
- [ ] Add LaMP to validation branch in `allsplit_step` (even if just loss logging)
- [ ] Verify BERT tokenizer auto-downloads correctly

### Nice to Have
- [ ] Add R-Precision and Matching Score metrics for LaMP evaluation
- [ ] Add LaMP checkpoint saving with best retrieval accuracy
- [ ] Add `encode_text` / `encode_motion` utility methods for downstream use

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                 MotionGPT + LaMP Integration                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                    │
│  MotionGPT (LightningModule)                                     │
│  ├── self.vae = RVQVae (frozen for lamp stage)                   │
│  │     ├── encoder: Conv1D [133→512, T→T/4]                     │
│  │     ├── quantizer: ResidualVQ (6 codebooks × 512 codes)      │
│  │     └── decoder: Conv1D [512→133, T/4→T]                     │
│  │                                                                 │
│  ├── self.lm = LaMP (QFormer_Base)                               │
│  │     ├── motion_encoder = self.vae.encoder (frozen)            │
│  │     ├── motion_projection [512→1408]                          │
│  │     ├── Qformer (BERT + cross-attention)                      │
│  │     │     ├── query_tokens [1×32×768]                         │
│  │     │     └── cross_attention_freq = 2                        │
│  │     ├── text_proj [768→512]                                   │
│  │     ├── motion_proj [768→512]                                 │
│  │     ├── itm_head [768→2]                                      │
│  │     ├── motion_cls [768→512]                                  │
│  │     └── temp (learnable temperature)                          │
│  │                                                                 │
│  └── self._losses = GPTLosses(stage="lamp")                      │
│        ├── lamp_ptc: Contrastive loss                            │
│        ├── lamp_ptm: Matching loss                               │
│        ├── lamp_lm: Language modeling loss                       │
│        └── lamp_gen: Generation loss                             │
│                                                                    │
│  Training Flow:                                                   │
│  batch["motion"] [B,T,133] ──→ LaMP.forward(motion, text)       │
│  batch["text"]   [B strings]     ├── loss_ptc (contrastive)     │
│                                    ├── loss_ptm (matching)        │
│                                    ├── loss_lm  (M→T generation) │
│                                    └── loss_gen (T→M generation) │
│                                    = total_loss                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure After Integration

```
mGPT/archs/lamp/
├── __init__.py           # ✅ EXISTS
├── lamp_model.py         # ✅ EXISTS (LaMP main model)
├── qformer_base.py       # ✅ EXISTS (QFormer base class)
├── QFormer.py            # ❌ MISSING → Copy from LaMP
├── QFormer_output.py     # ❌ MISSING → Copy from LaMP
└── basemodel.py          # ❌ MISSING → Copy from LaMP
```

---

## Estimated Steps to Working LaMP Training

1. **Copy 3 missing files** (5 min)
2. **Verify imports** (2 min)
3. **Fix VQ model instantiation** (10 min)
4. **Verify RVQVae.encode compatibility** (5 min)
5. **Add val loss logging** (5 min)
6. **Test training launch** (user runs manually)

Total estimated code changes: ~30 lines of modification + 3 file copies.

---

---

## Implementation Status (Updated)

### Completed Changes

#### 1. Missing Dependencies Copied
- `mGPT/archs/lamp/QFormer.py` - Full BERT with cross-attention (from LaMP)
- `mGPT/archs/lamp/QFormer_output.py` - Output dataclasses (from LaMP)
- `mGPT/archs/lamp/basemodel.py` - Gather utilities + BaseModel for LaMP (from LaMP)

#### 2. VQ Model Instantiation Fixed (`mGPT/models/mgpt.py`)
- For `stage == "lamp"`, LaMP now receives the already-instantiated `self.vae` instead of creating a duplicate
- VAE is properly frozen for the `lamp` stage (same as `lm` stages)

#### 3. Validation Support Added (`mGPT/models/mgpt.py`)
- LaMP `allsplit_step` now runs on both `train` and `val` splits (was train-only)
- Val losses (lamp_ptc, lamp_ptm, lamp_lm, lamp_gen) are logged

### How to Run LaMP Training

```bash
python train.py --cfg configs/lamp_h2s.yaml
```

This will:
1. Load `H2SDataModule` (How2Sign data with 133D motion features)
2. Load pretrained RVQ-VAE from `TRAIN.PRETRAINED_VAE` checkpoint (frozen)
3. Create `LaMP` model with QFormer (32 query tokens, cross-attention every 2 layers)
4. Train with 4 proxy tasks:
   - **loss_ptc**: Motion-Text Contrastive (align representations)
   - **loss_ptm**: Motion-Text Matching (binary match/no-match with hard negatives)
   - **loss_lm**: Language Modeling (motion→text generation)
   - **loss_gen**: Token Generation (text→motion token prediction)
5. Log all 4 losses to WandB

### Training Flow

```
train.py
  → parse_args("train") → cfg from configs/lamp_h2s.yaml
  → build_data(cfg) → H2SDataModule (How2Sign, 133D features)
  → build_model(cfg, datamodule) → MotionGPT(stage="lamp")
      → self.vae = RVQVae(nfeats=133, ...) [frozen]
      → self.lm = LaMP(vq_model=self.vae, ...)
      → self._losses = GPTLosses(stage="lamp")
  → load_pretrained_vae(cfg, model) → loads RVQ-VAE weights
  → trainer.fit(model, datamodule)
      → training_step → allsplit_step("train")
          → train_lamp_forward(batch)
              → LaMP.forward(motion=[B,T,133], text=[B strings])
              → returns {outputs: QFormer_Output, text_feat, motion_feat}
          → GPTLosses.update(rs_set)
              → lamp_ptc + lamp_ptm + lamp_lm + lamp_gen = total_loss
      → validation_step → allsplit_step("val")
          → same as train (logs val losses)
```

### Complete File Structure

```
mGPT/archs/lamp/
├── __init__.py           # ✅ Exports LaMP class
├── lamp_model.py         # ✅ LaMP model (4 training objectives)
├── qformer_base.py       # ✅ QFormer base class (init_Qformer, init_tokenizer)
├── QFormer.py            # ✅ BERT + cross-attention implementation
├── QFormer_output.py     # ✅ Output dataclasses
└── basemodel.py          # ✅ Gather utilities for distributed training

mGPT/models/mgpt.py       # ✅ Modified: LaMP VQ instantiation + val support
mGPT/losses/mgpt.py       # ✅ Already has LaMP loss registration
configs/lamp_h2s.yaml     # ✅ Training config
```

*Plan created: 2026-02-23*
*Implementation completed: 2026-02-23*
*Workspace: /home/student/hwu/Workplace/MotionGPT*
*Reference: /home/student/hwu/Workplace/LaMP/ARCHITECTURE_DOCUMENTATION.md*
