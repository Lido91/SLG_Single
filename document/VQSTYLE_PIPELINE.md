# VQ-Style RVQ-VAE Training Pipeline

## Overview

VQ-Style adds two regularization losses to standard RVQ-VAE training:
- **Contrastive loss** on Q1+ (style quantizers) — clusters by text semantics
- **MI loss** on Q0 (content quantizer) — removes text info from Q0

This disentangles Q0 (signer style) from Q1+ (text-predictable content).

---

## Step 1: Precompute Text Cluster Labels

```bash
python scripts/precompute_text_clusters.py \
    --data_root /data/hwu/slg_data/How2Sign \
    --num_clusters 64 \
    --output cluster_labels.json
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `--data_root` | (required) | How2Sign data root |
| `--num_clusters` | 64 | K-means cluster count. 64 balances batch collision rate with granularity |
| `--output` | `cluster_labels.json` | Output path |
| `--seed` | 42 | Random seed for reproducibility |

**Requirements:** `clip`, `sklearn`, `pandas`

```bash
pip install git+https://github.com/openai/CLIP.git scikit-learn pandas
```

**Output:** JSON file mapping `{sample_name: cluster_id}`

---

## Step 2: Set Cluster Labels Path in Config

Edit `configs/vq/h2s_rvq_3_vqstyle.yaml`, set `cluster_labels_path`:

```yaml
cluster_labels_path: '/path/to/cluster_labels.json'
```

Or override via command line (see Step 3).

---

## Step 3: Train VQ-Style RVQ-VAE

```bash
python train.py --cfg configs/deto_h2s_rvq_3_vqstyle.yaml \
    DATASET.H2S.ROOT /data/hwu/slg_data/How2Sign \
    model.params.motion_vae.params.cluster_labels_path /path/to/cluster_labels.json
```

**Key config differences from standard RVQ-VAE (`deto_h2s_rvq_3.yaml`):**

| Config | Standard | VQ-Style |
|--------|----------|----------|
| VAE class | `RVQVae` | `RVQVaeVQStyle` |
| Batch size | 32 | 64 (more contrastive pairs) |
| `LAMBDA_CONTRASTIVE` | — | 1.0 |
| `LAMBDA_MI` | — | 1.0 |
| `lambda_con` (inside VAE) | — | 0.005 |
| `lambda_mi` (inside VAE) | — | 0.02 |

---

## Step 4: Monitor Training

Wandb logged metrics:

| Metric | Meaning |
|--------|---------|
| `recons/feature/train` | Reconstruction loss |
| `vq/commit/train` | Commitment loss |
| `vqstyle/con/train` | Contrastive loss (Q1+ style) |
| `vqstyle/mi/train` | Mutual information loss (Q0 content) |
| `total/train` | Sum of all weighted losses |

**Sanity checks:**
- `vqstyle/con` should be > 0 if batch has positive pairs (same cluster). If always 0, check cluster labels path or increase `num_clusters`
- `vqstyle/mi` should be > 0 if batch has multiple clusters. If always 0, check labels
- `recons/feature` should not degrade significantly vs baseline RVQ-VAE

---

## Hyperparameter Tuning

| Param | Location | Default | Notes |
|-------|----------|---------|-------|
| `lambda_con` | `configs/vq/h2s_rvq_3_vqstyle.yaml` | 0.005 | From VQ-Style Table 4. Increase if Q1+ not clustering |
| `lambda_mi` | same | 0.02 | From VQ-Style Table 4. Increase if Q0 still encodes text |
| `tau_con` | same | 0.07 | Contrastive temperature. Lower = sharper |
| `tau_mi` | same | 1.0 | MI soft assignment temperature |
| `num_clusters` | same + clustering script | 64 | Trade-off: fewer = more batch collisions, coarser semantics |
| `content_cutoff` | same | 1 | Number of content quantizers (Q0). With 3 quantizers: 1 content + 2 style |
| Batch size | `configs/deto_h2s_rvq_3_vqstyle.yaml` | 64 | Larger = more positive pairs per batch |

---

## File Structure

```
scripts/precompute_text_clusters.py     # Step 1: Generate cluster labels
configs/vq/h2s_rvq_3_vqstyle.yaml      # VQ-Style VAE config
configs/deto_h2s_rvq_3_vqstyle.yaml     # Training config
mGPT/archs/mgpt_rvq_vqstyle.py         # RVQVaeVQStyle implementation
mGPT/archs/tools/residual_vq.py        # Modified: return_per_quantizer support
mGPT/losses/mgpt.py                    # Modified: vqstyle_con/mi losses
mGPT/models/mgpt.py                    # Modified: passes names to VAE
```
