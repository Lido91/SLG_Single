# Per-Time Corrupted RVQ Augmentation

## Source
T2M-HiFiGPT paper (arXiv:2312.10628), Section 3.4

## What was added

### Problem: Exposure Bias
During training, each decoder's autoregressive input is always **clean GT tokens**. But during inference, the input comes from the model's own predictions (which may have errors). This train/test discrepancy is called **exposure bias**.

### Solution: Per-Time Corruption
During training, randomly corrupt `corrupt_ratio` fraction of **timesteps** by replacing the **entire row** (all quantizers Q0, Q1, Q2) with random codes. The loss is still computed against **clean GT targets**.

```
GT codes (T=6, R=3):                Corrupted (corrupt_ratio=0.5):

t=1: [23, 45, 67]                   t=1: [23, 45, 67]  ← kept
t=2: [12, 89, 34]        ──►        t=2: [██, ██, ██]  ← replaced with random
t=3: [56, 78, 90]                   t=3: [56, 78, 90]  ← kept
t=4: [11, 22, 33]                   t=4: [██, ██, ██]  ← replaced with random
t=5: [44, 55, 66]                   t=5: [44, 55, 66]  ← kept
t=6: [77, 88, 99]                   t=6: [██, ██, ██]  ← replaced with random
```

**Why per-time (not per-code)?** Per-code corruption would break the RVQ sum at each timestep, making every position noisy. Per-time keeps some timesteps fully clean so the decoder can still learn temporal patterns.

Paper ablation results (HumanML3D):

| Strategy | FID↓ | MM-Dist↓ | Top-1↑ |
|----------|------|----------|--------|
| No corruption | 0.258 | 3.846 | 0.374 |
| Per-code corruption | 0.149 | 3.133 | 0.476 |
| **Per-time corruption** | **0.134** | **3.077** | **0.486** |

## Changes Made

### 1. `mGPT/archs/mgpt_rvq_hierarchical.py` — `HierarchicalRVQGPT` class

**New parameter in `__init__`:**
```python
corrupt_ratio=0.5  # τ: fraction of timesteps to corrupt (0.0 = disabled)
```

**New method `_corrupt_per_time(motion_codes)`:**
- Input: `(B, T, num_q)` clean GT codes
- For each sample, each timestep: with probability `corrupt_ratio`, replace all quantizer codes with random codes from `[0, num_vq)`
- Output: `(B, T, num_q)` corrupted codes
- Only used during training

**Modified `forward()` method:**
```
Before:
  motion_codes → split → Q0 decoder(GT) → Q1 decoder(GT) → Q2 decoder(GT)
  Loss targets: GT

After:
  motion_codes → corrupt copy → split → Q0 decoder(corrupted) → Q1 decoder(corrupted) → Q2 decoder(corrupted)
  Loss targets: still clean GT (unchanged)
  pkeep GT branch: also uses clean GT (not corrupted)
```

Data flow:
- `input_codes` = corrupted copy (fed to decoders as autoregressive input)
- `target_q0/q1/q2` = clean GT (used for CE loss, never corrupted)
- pkeep scheduled sampling: GT branch uses `target_q0/q1` (clean), not `motion_q0/q1` (corrupted)

### 2. Config files updated

**`configs/deto_h2s_rvq_hierarchical_3layer_lgvq_h2s.yaml`** (primary):
```yaml
corrupt_ratio: 0.5  # Per-time RVQ corruption
```

**`configs/deto_h2s_rvq_hierarchical_3layer.yaml`**:
```yaml
corrupt_ratio: 0.5
```

**`configs/lm/rvq_hierarchical.yaml`** (shared LM config):
```yaml
corrupt_ratio: 0.5
```

### 3. Default behavior
- `corrupt_ratio=0.5` in `__init__` signature → **enabled by default** for all configs
- Other hierarchical 3layer configs that don't explicitly set `corrupt_ratio` will automatically use the default (0.5)
- Set `corrupt_ratio: 0.0` in yaml to disable

## Interaction with existing pkeep

These are **two independent mechanisms** addressing different aspects of exposure bias:

| Mechanism | Where it acts | What it does |
|-----------|--------------|--------------|
| `corrupt_ratio` (new) | Each decoder's own autoregressive input | Replaces timesteps with random codes |
| `pkeep` (existing) | Cross-decoder conditioning (Q0→Q1, Q1→Q2) | Mixes GT vs predicted tokens for conditioning |

Both can (and should) be used simultaneously. With `corrupt_ratio=0.5, pkeep=0.5`:
- Each decoder sees corrupted input tokens (50% timesteps random)
- Q1 gets Q0 conditioning that's 50% GT / 50% predicted
- Q2 gets Q1 conditioning that's 50% GT / 50% predicted
