# Progressive Contrastive Grounding (PCG) — Implementation Plan

## Overview

Add training-only per-level semantic alignment on top of the existing `RVQVae` tokenizer.  
The main motion path (`encoder -> ResidualVQ -> decoder`) is **completely unchanged**.  
All new modules are auxiliary, training-only, and disabled at inference.

---

## Files to Create

| File | Purpose |
|------|---------|
| `mGPT/archs/tools/pcg_modules.py` | Module A (`SemanticHead`) and Module B (`AttentionGroundingBlock`) |
| `mGPT/archs/mgpt_rvq_pcg.py` | `RVQVaePCG` — subclasses `RVQVae`, adds PCG aux path |
| `configs/deto_h2s_rvq_3_pcg_whisper_h2s.yaml` | Training config for speech mode on How2Sign |

## Files to Modify

| File | Change |
|------|--------|
| `mGPT/losses/mgpt.py` | Add `pcg_q0`, `pcg_q1`, `vq_geo` loss terms to `GPTLosses` |
| `mGPT/models/mgpt.py` | Add PCG branch in `train_vae_forward` |

---

## Architecture Diagram

```
motion [B, T, 133]
    |
    v  ===================== MAIN PATH (unchanged) =========================
    Encoder  ->  z_e [B, 512, T']
    ResidualVQ
        Q0: z_q0 [B, 512, T']
        Q1: z_q1 [B, 512, T']    (via return_per_quantizer=True)
        Q2: z_q2 [B, 512, T']
    ->  x_quantized = z_q0 + z_q1 + z_q2
    Decoder  ->  x_out [B, T, 133]
    =====================================================================
    |
    v  ============= TRAINING-ONLY AUX PATH =============================

    z0  = z_q0                        [B, 512, T']
    z01 = z_q0 + z_q1                 [B, 512, T']

    +-- Module A: Semantic Heads ------------------------------------+
    |  head_q0(z0)   -> mean-pool -> MLP -> L2-norm -> h0  [B, 512] |
    |  head_q1(z01)  -> mean-pool -> MLP -> L2-norm -> h1  [B, 512] |
    +----------------------------------------------------------------+

    Frozen condition encoder  ->  g_cond [B, D_cond]  +  E [B, S, D_cond]
    cond_global_proj(g_cond)  -> L2-norm -> g_proj     [B, 512]

    +-- Module B: AttentionGroundingBlock ---------------------------+
    |  input:  z01.detach() [B, 512, T']  and  E [B, S, D_cond]     |
    |  query = motion_proj(z01) -> [B, T', 512]                      |
    |  kv    = cond_proj(E)    -> [B, S,  512]                       |
    |  cross_attn(q, kv) -> residual+norm -> mean-pool -> out_proj   |
    |  output: a  [B, 512]  L2-normalized                            |
    +----------------------------------------------------------------+

    Losses:
      L_pcg_q0 = InfoNCE(h0, g_proj)       <- Q0 vs global condition
      L_pcg_q1 = InfoNCE(h1, a)            <- Q1 vs grounded target
    =====================================================================
```

**Q2 gets NO contrastive loss** — left purely for reconstruction / geometric refinement.

---

## Loss Formula

```
L_total = L_rec
        + lambda_commit * L_commit
        + lambda_geo    * L_geo           # joint-space MSE
        + lambda_pcg_q0 * L_pcg_q0       # Q0 InfoNCE
        + lambda_pcg_q1 * L_pcg_q1       # Q1 InfoNCE
```

Where:
- `L_rec` = SmoothL1 on 133D motion features (existing `recons_feature`)
- `L_commit` = VQ commitment loss (existing)
- `L_geo` = MSE on 3D joint positions (`joints_rst` vs `joints_ref`)
- `L_pcg_q0` = Symmetric InfoNCE between `h0` and `g_proj`
- `L_pcg_q1` = Symmetric InfoNCE between `h1` and `a` (or `g_proj` in ablation)

---

## Ablation Modes (`pcg_mode` config flag)

| Mode | L_pcg_q0 | L_pcg_q1 | Grounding Block | Notes |
|------|----------|----------|-----------------|-------|
| `baseline` | off | off | off | Pure reconstruction baseline |
| `q0_only` | on | off | off | Only Q0 contrastive |
| `q0_q1` | on | on (vs `a`) | on | Full PCG |
| `all_levels_contrast` | on | on (vs `a`) | on | Alias of q0_q1 for ablation tables |
| `q1_without_grounding_block` | on | on (vs `g_proj`) | off | Q1 uses global cond, no cross-attn |

---

## Staged Training (`training_stage` config flag)

| Stage | Active losses | Purpose |
|-------|---------------|---------|
| `stage1` | `L_rec + L_commit + L_geo` | Warm up reconstruction |
| `stage2` | `+ L_pcg_q0` | Ground Q0 on global condition |
| `stage3` | `+ L_pcg_q0 + L_pcg_q1` | Ground Q1 on cross-attended target |

Stages are config-driven: change `training_stage` in YAML and resume from checkpoint.

---

## Gradient Flow Design

### Q0 contrastive
```
InfoNCE(h0, g_proj)
  - grad -> h0 -> head_q0 -> z_q0 -> quantizer   (shapes Q0 codebook)
  - grad -> g_proj -> cond_global_proj             (learns condition projection)
  - g_cond is detached (frozen encoder)
```

### Q1 contrastive (with grounding block)
```
InfoNCE(h1, a)
  - grad -> h1 -> head_q1 -> z01 -> quantizer     (shapes Q0+Q1 codebooks)
  - grad -> a -> grounding_block -> E_cond         (trains grounding block)
  - z01 is DETACHED before entering grounding_block (prevents circular grad)
  - a is NOT fed back into decoder/quantizer/main path
```

### Q1 contrastive (without grounding block)
```
InfoNCE(h1, g_proj)
  - grad -> h1 -> head_q1 -> z01 -> quantizer     (shapes Q0+Q1 codebooks)
  - grad -> g_proj -> cond_global_proj             (same as Q0)
  - Same global target as Q0 — ablation to test grounding block value
```

---

## ResidualVQ Hook

`ResidualVQ.forward()` already supports `return_per_quantizer=True` (residual_vq.py:96):

```python
x_quantized, all_indices, loss_commit, perplexity, all_z_q, all_residuals = \
    self.quantizer(x_encoder, return_per_quantizer=True)
# all_z_q[i]: [B, code_dim, T'] -- i-th quantizer's contribution (with STE grad)
```

This is only called during training when PCG is active. Zero overhead at inference.

---

## Module Details

### SemanticHead (`pcg_modules.py`)
```
Input:  z [B, C, T']  (e.g. z0 or z01)
        mean-pool over T' -> [B, C]
        Linear(C, hidden_dim) -> ReLU -> Linear(hidden_dim, proj_dim)
        L2 normalize
Output: h [B, proj_dim]
```

### AttentionGroundingBlock (`pcg_modules.py`)
```
Input:  z01 [B, C, T']          (detached)
        E   [B, S, cond_dim]    (condition sequence)
        E_mask [B, S]           (1=valid, 0=pad, or None)

        q  = motion_proj(z01.permute(0,2,1))   [B, T', proj_dim]
        kv = cond_proj(E)                       [B, S,  proj_dim]
        attn_out = MHA(q, kv, kv)               [B, T', proj_dim]
        x = LayerNorm(q + attn_out)             residual + norm
        x = mean-pool over T'                   [B, proj_dim]
        a = out_proj(x) -> L2 normalize
Output: a [B, proj_dim]
```

### RVQVaePCG (`mgpt_rvq_pcg.py`)
```
Subclasses: RVQVae

New __init__ params:
  pcg_mode:            str = 'q0_q1'
  training_stage:      str = 'stage3'
  condition_modality:  str = 'speech'    # or 'text'
  speech_encoder_type: str = 'whisper-medium'
  proj_dim:            int = 512
  grounding_n_head:    int = 8
  grounding_dropout:   float = 0.1
  infonce_temp:        float = 0.07

New submodules:
  self.cond_encoder        # frozen SpeechEncoder or CLIP
  self.head_q0             # SemanticHead
  self.head_q1             # SemanticHead
  self.cond_global_proj    # Linear + LayerNorm
  self.grounding_block     # AttentionGroundingBlock (or None)

Sentinel attr:
  self.pcg_mode            # used by train_vae_forward to detect PCG VAE

forward(features, batch=None):
  Inference (batch=None or not self.training):
    returns: x_out, loss_commit, perplexity
    -- identical to RVQVae.forward()

  Training with batch:
    returns: x_out, loss_commit, perplexity, loss_pcg_q0, loss_pcg_q1

_get_condition(batch):
  @torch.no_grad (frozen encoder)
  Speech mode: batch['speech_feats'] or batch['audio'] -> SpeechEncoder
    g_cond = mean-pooled [B, cond_dim]
    E_cond = sequence    [B, S, cond_dim]
    E_mask = mask         [B, S]
  Text mode: batch['text'] -> CLIP
    g_cond = CLS          [B, 512]
    E_cond = unsqueezed   [B, 1, 512]
    E_mask = None
```

---

## Integration Points

### `mGPT/models/mgpt.py` — `train_vae_forward`

New branch added **before** existing `hasattr` checks:

```python
elif hasattr(self.vae, 'pcg_mode'):
    result = self.vae(feats_ref, batch=batch)
    if len(result) == 5:
        feats_rst, loss_commit, perplexity, loss_pcg_q0, loss_pcg_q1 = result
    else:
        feats_rst, loss_commit, perplexity = result
        loss_pcg_q0 = torch.tensor(0.0, device=feats_ref.device)
        loss_pcg_q1 = torch.tensor(0.0, device=feats_ref.device)
    # ... zero out unused loss terms (loss_con, loss_mi, loss_align, etc.)

# After computing joints:
if hasattr(self.vae, 'pcg_mode'):
    loss_geo = F.mse_loss(joints_rst.float(), joints_ref.float())
else:
    loss_geo = torch.tensor(0.0, device=feats_ref.device)

rs_set = {
    ...existing keys...,
    "loss_pcg_q0": loss_pcg_q0,
    "loss_pcg_q1": loss_pcg_q1,
    "loss_geo": loss_geo,
}
```

### `mGPT/losses/mgpt.py` — `GPTLosses`

New loss entries under `stage == "vae"`:

```python
# PCG losses
losses.append("pcg_q0")
params['pcg_q0'] = cfg.LOSS.get('LAMBDA_PCG_Q0', 0.0)

losses.append("pcg_q1")
params['pcg_q1'] = cfg.LOSS.get('LAMBDA_PCG_Q1', 0.0)

# Geometric (joint-space) loss
losses.append("vq_geo")
params['vq_geo'] = cfg.LOSS.get('LAMBDA_GEO', 0.0)
```

Loss function routing — add `'pcg'` to the CommitLoss prefix set and `'geo'` to the suffix set:

```python
elif loss.split('_')[0] in ['lamp', 'masked', 'vqstyle', 'pcg']:
    losses_func[loss] = CommitLoss
elif loss.split('_')[1] in [..., 'geo']:
    losses_func[loss] = CommitLoss
```

Update method additions:

```python
if 'loss_pcg_q0' in rs_set:
    total += self._update_loss("pcg_q0", rs_set['loss_pcg_q0'], rs_set['loss_pcg_q0'])
if 'loss_pcg_q1' in rs_set:
    total += self._update_loss("pcg_q1", rs_set['loss_pcg_q1'], rs_set['loss_pcg_q1'])
if 'loss_geo' in rs_set:
    total += self._update_loss("vq_geo", rs_set['loss_geo'], rs_set['loss_geo'])
```

W&B log names (via `loss2logname`):
- `pcg_q0` -> `pcg/q0/{split}`
- `pcg_q1` -> `pcg/q1/{split}`
- `vq_geo` -> `vq/geo/{split}`

---

## Config Structure

```yaml
NAME: DETO_RVQ_wholebody_3_pcg_whisper_h2s

TRAIN:
  STAGE: vae
  BATCH_SIZE: 32
  END_EPOCH: 500

DATASET:
  target: mGPT.data.H2S.H2SDataModule
  H2S:
    USE_SPEECH: true
    AUDIO_DIR: /data/hwu/slg_data/How2Sign

LOSS:
  LAMBDA_FEATURE: 1.0
  LAMBDA_VELOCITY: 0.0
  LAMBDA_COMMIT: 0.25
  LAMBDA_GEO: 0.1
  LAMBDA_PCG_Q0: 0.1
  LAMBDA_PCG_Q1: 0.1
  ABLATION:
    RECONS_LOSS: "l1_smooth"

model:
  target: mGPT.models.mgpt.MotionGPT
  params:
    condition: "text"
    task: "t2m"
    lm: ${lm.default}
    motion_vae: ${vq.h2s_rvq_3_pcg_whisper}

vq:
  h2s_rvq_3_pcg_whisper:
    target: mGPT.archs.mgpt_rvq_pcg.RVQVaePCG
    params:
      # --- RVQVae base params ---
      nfeats: 133
      num_quantizers: 3
      quantizer: 'ema_reset'
      quantize_dropout_prob: 0.2
      quantize_dropout_cutoff_index: 0
      shared_codebook: false
      code_num: 512
      code_dim: 512
      output_emb_width: 512
      down_t: 2
      stride_t: 2
      width: 512
      depth: 3
      dilation_growth_rate: 3
      norm: None
      activation: 'relu'
      # --- PCG-specific params ---
      pcg_mode: q0_q1              # ablation flag
      training_stage: stage3       # current stage
      condition_modality: speech
      speech_encoder_type: whisper-medium
      proj_dim: 512
      grounding_n_head: 8
      grounding_dropout: 0.1
      infonce_temp: 0.07
```

---

## What Does NOT Change

- `RVQVae.encode()` — used by LM-stage tokenization, unchanged
- `RVQVae.decode()` / `decode_partial()` — unchanged
- `ResidualVQ.forward()` default behavior (without `return_per_quantizer`) — unchanged
- `HierarchicalRVQGPT` — unchanged
- All existing VAE variants (`lgvq`, `vqstyle`, `align`) — unchanged
- Inference / export: returns `(x_out, loss_commit, perplexity)` — unchanged

---

## Inference / Export

When `batch=None` or `not self.training`:
- `forward()` returns the standard 3-tuple `(x_out, loss_commit, perplexity)`
- Module A and Module B are never called
- `encode()` and `decode()` are inherited directly from `RVQVae` with no changes
