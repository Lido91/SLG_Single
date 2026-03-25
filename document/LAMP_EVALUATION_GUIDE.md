# LaMP Evaluation Guide: Contrastive Learning Metrics

**Date:** 2026-02-10
**Task:** How to Evaluate LaMP Pretraining Performance
**Key Insight:** LaMP is **contrastive learning**, NOT generation - evaluation is completely different

---

## Table of Contents

1. [Why LaMP Evaluation is Different](#why-lamp-evaluation-is-different)
2. [Evaluation Metrics Overview](#evaluation-metrics-overview)
3. [Implementation Guide](#implementation-guide)
4. [Step-by-Step Evaluation](#step-by-step-evaluation)
5. [Code Examples](#code-examples)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## Why LaMP Evaluation is Different

### ❌ **What LaMP Does NOT Do**
- Does NOT generate motion sequences
- Does NOT decode VQ tokens to motion
- Does NOT use T2M metrics (FID, Diversity, etc.)

### ✅ **What LaMP DOES**
- **Learns aligned embeddings** between text and motion
- Text embedding: [B, 512] from QFormer
- Motion embedding: [B, 512] from QFormer
- Creates a **shared embedding space** where:
  - Similar text/motion pairs are close
  - Dissimilar pairs are far apart

### Evaluation Philosophy

```
Traditional T2M:  Text → Model → Motion → Metrics (FID, MPJPE, etc.)
                  ✅ Evaluate generated motion quality

LaMP:            Text → Embedding [512D]
                 Motion → Embedding [512D]
                 ✅ Evaluate embedding alignment (retrieval tasks)
```

---

## Evaluation Metrics Overview

LaMP is evaluated using **retrieval metrics**, similar to CLIP (text-image retrieval):

### 1. **R-Precision (Recall @ K)**

**What it measures:** Given a text query, can we retrieve the correct motion from a set of candidates?

```
Query: "A person walks forward"
Candidates: [motion1, motion2, ..., motion100]

Similarity: cosine(text_embedding, motion_embeddings)
Rank candidates by similarity
Check if ground-truth is in top-K

R@1: Ground-truth in top-1 (most similar)
R@5: Ground-truth in top-5
R@10: Ground-truth in top-10
```

**Higher is better** (0.0 to 1.0)

### 2. **Matching Score**

**What it measures:** Average distance between paired text-motion embeddings

```
For each (text_i, motion_i) pair:
    distance = euclidean_distance(text_embedding_i, motion_embedding_i)

Matching Score = mean(distances)
```

**Lower is better** (measures alignment quality)

### 3. **Diversity**

**What it measures:** Variance in the learned motion embedding space

```
diversity = variance(motion_embeddings)
```

**Higher is better** (embeddings cover diverse motions, not collapsed)

### 4. **Text-to-Motion Retrieval (T2M)**

**Direction:** Text query → Retrieve motion

```python
# Given text embedding [1, 512]
# Given motion embedding database [N, 512]

similarities = cosine_similarity(text_emb, motion_embs)  # [N]
top_k_indices = similarities.argsort()[-K:][::-1]

# Check if ground-truth index in top_k_indices
recall_at_k = ground_truth_idx in top_k_indices
```

### 5. **Motion-to-Text Retrieval (M2T)**

**Direction:** Motion query → Retrieve text

```python
# Given motion embedding [1, 512]
# Given text embedding database [N, 512]

similarities = cosine_similarity(motion_emb, text_embs)  # [N]
top_k_indices = similarities.argsort()[-K:][::-1]

recall_at_k = ground_truth_idx in top_k_indices
```

---

## Implementation Guide

### Architecture for Evaluation

Since we adapted LaMP to use MotionGPT's RVQ-VAE, we need to add evaluation methods to our `LaMP` class:

```python
# In mGPT/archs/lamp/lamp_model.py

class LaMP(QFormer_Base):
    # ... existing code ...

    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode text to normalized embeddings for retrieval.

        Args:
            texts: List[str] text descriptions

        Returns:
            text_embeddings: [B, embed_dim] normalized embeddings
        """
        device = next(self.parameters()).device

        text_tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        # CLS token embedding + projection + normalization
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]),
            dim=-1
        )  # [B, 512]

        return text_feat

    @torch.no_grad()
    def encode_motion(self, motion):
        """
        Encode motion to normalized embeddings for retrieval.

        Args:
            motion: [B, T, nfeats] motion features

        Returns:
            motion_embeddings: [B, embed_dim] normalized embeddings
        """
        device = motion.device
        B = motion.shape[0]

        # Motion encoding through frozen RVQ-VAE encoder
        motion_input = motion.permute(0, 2, 1)  # [B, nfeats, T]
        motion_embeds = self.motion_encoder(motion_input)  # [B, code_dim, T']
        motion_embeds = motion_embeds.permute(0, 2, 1)  # [B, T', code_dim]

        # Project to QFormer dimension
        motion_embeds = motion_embeds @ self.motion_projection  # [B, T', 1408]
        motion_atts = torch.ones(motion_embeds.size()[:-1], dtype=torch.long).to(device)

        # QFormer cross-attention
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=motion_embeds,
            encoder_attention_mask=motion_atts,
            use_cache=True,
            return_dict=True,
        )

        # Motion queries + projection + pooling + normalization
        motion_feats = F.normalize(
            self.motion_proj(query_output.last_hidden_state),
            dim=-1
        )  # [B, 32, 512]

        motion_feats_pooled = F.normalize(
            torch.mean(motion_feats, dim=1),
            dim=-1
        )  # [B, 512]

        return motion_feats_pooled
```

---

## Step-by-Step Evaluation

### Step 1: Extract Embeddings from Dataset

```python
import torch
import torch.nn.functional as F
from mGPT.config import instantiate_from_config
from mGPT.data.H2S import H2SDataModule

# Load trained LaMP model
cfg = load_config("configs/lamp_h2s.yaml")
model = MotionGPT(cfg, ...)
model.load_state_dict(torch.load("lamp_checkpoint.ckpt"))
model.eval()
model.cuda()

# Load validation dataset
datamodule = H2SDataModule(cfg)
datamodule.setup()
val_loader = datamodule.val_dataloader()

# Extract embeddings
text_embeddings = []
motion_embeddings = []
texts_list = []
names_list = []

for batch in val_loader:
    texts = batch["text"]
    motion = batch["motion"].cuda()  # [B, T, 133]
    names = batch["name"]

    # Encode
    text_emb = model.lm.encode_text(texts)  # [B, 512]
    motion_emb = model.lm.encode_motion(motion)  # [B, 512]

    text_embeddings.append(text_emb.cpu())
    motion_embeddings.append(motion_emb.cpu())
    texts_list.extend(texts)
    names_list.extend(names)

# Concatenate all batches
text_embeddings = torch.cat(text_embeddings, dim=0)  # [N, 512]
motion_embeddings = torch.cat(motion_embeddings, dim=0)  # [N, 512]

print(f"Extracted {len(text_embeddings)} text-motion pairs")
```

### Step 2: Compute Similarity Matrix

```python
# Compute cosine similarity matrix
# [N, 512] @ [512, N] = [N, N]
similarity_matrix = torch.mm(text_embeddings, motion_embeddings.t())

# similarity_matrix[i, j] = similarity between text_i and motion_j
# Diagonal = similarity between paired samples
# Off-diagonal = cross-modal similarity
```

### Step 3: Compute R-Precision

```python
def calculate_R_precision(text_embs, motion_embs, top_k=10):
    """
    Calculate R-Precision for text-to-motion retrieval.

    Args:
        text_embs: [N, 512] text embeddings
        motion_embs: [N, 512] motion embeddings
        top_k: list of K values, e.g., [1, 2, 3, 5, 10]

    Returns:
        R_precision: [top_k] recall at each K
    """
    N = text_embs.shape[0]

    # Compute similarity: [N, N]
    similarity = torch.mm(text_embs, motion_embs.t())

    # For each text query, rank all motions by similarity
    # argsort returns indices from lowest to highest
    # We want highest to lowest, so reverse
    ranked_indices = similarity.argsort(dim=1, descending=True)  # [N, N]

    # Ground truth: for text_i, the correct motion is motion_i
    # Check if i is in the top-K of ranked_indices[i]
    recalls = []

    if isinstance(top_k, int):
        top_k = [top_k]

    for K in top_k:
        correct = 0
        for i in range(N):
            # Top-K retrieved motion indices for text_i
            top_k_motions = ranked_indices[i, :K]

            # Check if ground-truth index (i) is in top-K
            if i in top_k_motions:
                correct += 1

        recall_at_k = correct / N
        recalls.append(recall_at_k)

    return recalls

# Calculate
R_precision = calculate_R_precision(
    text_embeddings,
    motion_embeddings,
    top_k=[1, 2, 3, 5, 10]
)

print(f"R@1:  {R_precision[0]:.4f}")
print(f"R@2:  {R_precision[1]:.4f}")
print(f"R@3:  {R_precision[2]:.4f}")
print(f"R@5:  {R_precision[3]:.4f}")
print(f"R@10: {R_precision[4]:.4f}")
```

### Step 4: Compute Matching Score

```python
def calculate_matching_score(text_embs, motion_embs):
    """
    Calculate average distance between paired embeddings.

    Lower is better (embeddings are closer).
    """
    N = text_embs.shape[0]

    # Euclidean distance between paired samples
    distances = torch.norm(text_embs - motion_embs, dim=1)  # [N]

    matching_score = distances.mean().item()

    return matching_score

matching_score = calculate_matching_score(text_embeddings, motion_embeddings)
print(f"Matching Score: {matching_score:.4f}")
```

### Step 5: Compute Diversity

```python
def calculate_diversity(embeddings, num_samples=300):
    """
    Calculate diversity of embeddings.

    Higher is better (more diverse representations).
    """
    N = embeddings.shape[0]

    if N > num_samples:
        # Sample random subset
        indices = torch.randperm(N)[:num_samples]
        embeddings = embeddings[indices]

    # Compute pairwise distances
    # [num_samples, 512] → [num_samples, num_samples]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # Average pairwise distance (excluding diagonal)
    mask = ~torch.eye(dist_matrix.shape[0], dtype=torch.bool)
    diversity = dist_matrix[mask].mean().item()

    return diversity

diversity = calculate_diversity(motion_embeddings, num_samples=300)
print(f"Diversity: {diversity:.4f}")
```

### Step 6: Visualization (Optional)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot similarity heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix[:50, :50].numpy(),  # First 50 samples
    cmap='YlOrRd',
    square=True,
    cbar=True
)
plt.title("Text-Motion Similarity Matrix")
plt.xlabel("Motion Index")
plt.ylabel("Text Index")
plt.savefig("similarity_heatmap.png", dpi=300)
plt.close()

# Plot embedding distribution (t-SNE)
from sklearn.manifold import TSNE

combined = torch.cat([text_embeddings, motion_embeddings], dim=0)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(combined.numpy())

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:len(text_embeddings), 0],
           embeddings_2d[:len(text_embeddings), 1],
           c='blue', alpha=0.5, label='Text')
plt.scatter(embeddings_2d[len(text_embeddings):, 0],
           embeddings_2d[len(text_embeddings):, 1],
           c='red', alpha=0.5, label='Motion')
plt.legend()
plt.title("Text-Motion Embedding Space (t-SNE)")
plt.savefig("embedding_tsne.png", dpi=300)
plt.close()
```

---

## Code Examples

### Complete Evaluation Script

```python
# eval_lamp_retrieval.py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from mGPT.config import parse_args, instantiate_from_config
from mGPT.models.mgpt import MotionGPT

@torch.no_grad()
def evaluate_lamp_retrieval(model, dataloader, device='cuda'):
    """
    Evaluate LaMP on text-motion retrieval tasks.

    Returns:
        results: dict with R-precision, matching score, diversity
    """
    model.eval()

    text_embs_list = []
    motion_embs_list = []

    print("Extracting embeddings...")
    for batch in tqdm(dataloader):
        texts = batch["text"]
        motion = batch["motion"].to(device)

        # Encode using LaMP
        text_emb = model.lm.encode_text(texts)
        motion_emb = model.lm.encode_motion(motion)

        text_embs_list.append(text_emb.cpu())
        motion_embs_list.append(motion_emb.cpu())

    # Concatenate
    text_embs = torch.cat(text_embs_list, dim=0)  # [N, 512]
    motion_embs = torch.cat(motion_embs_list, dim=0)  # [N, 512]

    N = text_embs.shape[0]
    print(f"\nEvaluating on {N} samples...")

    # ===== R-Precision (Text-to-Motion) =====
    similarity = torch.mm(text_embs, motion_embs.t())  # [N, N]
    ranked_indices = similarity.argsort(dim=1, descending=True)

    top_ks = [1, 2, 3, 5, 10]
    R_precision_t2m = []

    for K in top_ks:
        correct = sum(i in ranked_indices[i, :K] for i in range(N))
        R_precision_t2m.append(correct / N)

    # ===== R-Precision (Motion-to-Text) =====
    similarity_m2t = torch.mm(motion_embs, text_embs.t())
    ranked_indices_m2t = similarity_m2t.argsort(dim=1, descending=True)

    R_precision_m2t = []
    for K in top_ks:
        correct = sum(i in ranked_indices_m2t[i, :K] for i in range(N))
        R_precision_m2t.append(correct / N)

    # ===== Matching Score =====
    distances = torch.norm(text_embs - motion_embs, dim=1)
    matching_score = distances.mean().item()

    # ===== Diversity =====
    num_samples = min(300, N)
    indices = torch.randperm(N)[:num_samples]
    motion_subset = motion_embs[indices]
    dist_matrix = torch.cdist(motion_subset, motion_subset, p=2)
    mask = ~torch.eye(num_samples, dtype=torch.bool)
    diversity = dist_matrix[mask].mean().item()

    # ===== Results =====
    results = {
        'R_precision_t2m': {f'R@{k}': r for k, r in zip(top_ks, R_precision_t2m)},
        'R_precision_m2t': {f'R@{k}': r for k, r in zip(top_ks, R_precision_m2t)},
        'matching_score': matching_score,
        'diversity': diversity,
        'num_samples': N
    }

    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "="*70)
    print(f"{'LaMP Retrieval Evaluation Results':^70}")
    print("="*70)
    print(f"\n{'Num Samples:':<30} {results['num_samples']}")
    print(f"\n{'Text-to-Motion Retrieval:':<30}")
    for k, v in results['R_precision_t2m'].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"\n{'Motion-to-Text Retrieval:':<30}")
    for k, v in results['R_precision_m2t'].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"\n{'Matching Score:':<30} {results['matching_score']:.4f}")
    print(f"{'Diversity:':<30} {results['diversity']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Load config
    cfg = parse_args(phase="test")
    cfg.TEST.CHECKPOINTS = "experiments/mgpt/LaMP_H2S_Pretrain/checkpoints/last.ckpt"

    # Build datamodule
    from mGPT.data.H2S import H2SDataModule
    datamodule = H2SDataModule(cfg)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    # Build model
    model = MotionGPT(cfg, datamodule)

    # Load checkpoint
    ckpt = torch.load(cfg.TEST.CHECKPOINTS, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.cuda()
    model.eval()

    # Evaluate
    results = evaluate_lamp_retrieval(model, val_loader, device='cuda')

    # Print
    print_results(results)

    # Save
    import json
    output_dir = os.path.dirname(cfg.TEST.CHECKPOINTS)
    output_file = os.path.join(output_dir, "eval_retrieval_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
```

---

## Expected Results

### Benchmark Comparison

Based on the original LaMP paper and HumanML3D dataset:

| Metric | Random Baseline | CLIP Baseline | LaMP (Expected) | Our Target (How2Sign) |
|--------|----------------|---------------|-----------------|----------------------|
| **R@1 (T2M)** | 0.01 | 0.35 | 0.45-0.50 | 0.40-0.45 |
| **R@5 (T2M)** | 0.05 | 0.65 | 0.75-0.80 | 0.70-0.75 |
| **R@10 (T2M)** | 0.10 | 0.80 | 0.85-0.90 | 0.80-0.85 |
| **R@1 (M2T)** | 0.01 | 0.33 | 0.43-0.48 | 0.38-0.43 |
| **Matching Score** | ~50.0 | ~15.0 | ~8.0-10.0 | ~10.0-12.0 |
| **Diversity** | ~10.0 | ~11.0 | ~11.5-12.0 | ~11.0-12.0 |

**Notes:**
- How2Sign is harder than HumanML3D (full-body + hands vs body-only)
- Expect slightly lower retrieval accuracy
- Matching score may be higher (worse) due to increased complexity

### Training Progression

Monitor these metrics during training (every 10 epochs):

| Epoch | R@1 (T2M) | R@5 (T2M) | Matching Score | Status |
|-------|----------|----------|----------------|--------|
| 10 | 0.05-0.10 | 0.20-0.30 | ~30.0 | Early training |
| 30 | 0.15-0.25 | 0.45-0.55 | ~20.0 | Learning alignment |
| 50 | 0.25-0.35 | 0.60-0.70 | ~15.0 | Converging |
| 70 | 0.35-0.40 | 0.70-0.75 | ~12.0 | Near optimal |
| 100 | 0.40-0.45 | 0.75-0.80 | ~10.0 | **Final** |

**What to look for:**
- ✅ R@1 should increase steadily
- ✅ Matching score should decrease steadily
- ✅ Diversity should stabilize around 11-12
- ❌ If R@1 < 0.20 after 50 epochs → training issue
- ❌ If matching score > 25 after 50 epochs → poor alignment

---

## Troubleshooting

### Issue 1: Low R-Precision (< 0.10 after 50 epochs)

**Symptoms:**
```
R@1: 0.05
R@5: 0.15
R@10: 0.25
```

**Possible Causes:**

1. **Temperature not learning:**
```python
# Check temperature value
print(f"Temperature: {model.lm.temp.exp().item():.4f}")

# Should be around 10-20
# If stuck at 1.0 or exploded to 100+, there's a problem
```

2. **QFormer not processing motion correctly:**
```python
# Check motion embedding variance
motion_emb_var = motion_embeddings.var(dim=0).mean()
print(f"Motion embedding variance: {motion_emb_var:.4f}")

# Should be around 0.01-0.1
# If < 0.001, embeddings are collapsed
# If > 1.0, embeddings are too spread out
```

3. **Text encoding issues:**
```python
# Check if text embeddings are normalized
text_norms = torch.norm(text_embeddings, dim=1)
print(f"Text embedding norms: mean={text_norms.mean():.4f}, std={text_norms.std():.4f}")

# Should be close to 1.0 (normalized)
```

**Solutions:**
- Reduce learning rate for temperature: `lr=1e-5`
- Check that motion_projection is being updated
- Verify QFormer is not frozen during LaMP training

---

### Issue 2: High Matching Score (> 30.0 after 50 epochs)

**Symptoms:**
```
Matching Score: 35.2
```

**Possible Causes:**

1. **Projection heads not training:**
```python
# Check gradient flow
for name, param in model.lm.named_parameters():
    if 'proj' in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

2. **PTC loss not effective:**
```python
# Check loss values during training
# lamp/ptc/train should decrease from ~6.0 to ~0.5-1.0
```

**Solutions:**
- Increase PTC loss weight: `LAMBDA_PTC: 2.0`
- Check that `concat_all_gather` works correctly for distributed training
- Verify embedding normalization is applied

---

### Issue 3: Embedding Collapse (Diversity < 5.0)

**Symptoms:**
```
Diversity: 3.2
All embeddings very similar
```

**Possible Causes:**

1. **All embeddings converging to same vector:**
```python
# Check embedding std
print(f"Motion embedding std: {motion_embeddings.std(dim=0).mean():.4f}")

# Should be > 0.05
# If < 0.01, collapse occurred
```

**Solutions:**
- Reduce contrastive loss weight
- Add diversity loss term
- Check for exploding/vanishing gradients

---

### Issue 4: Diagonal Dominance in Similarity Matrix

**Symptoms:**
```
Similarity matrix has very strong diagonal
Off-diagonal values are near zero
```

This is actually **GOOD** - it means the model is learning correct alignments!

**Visualization:**
```python
# Plot similarity matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.heatmap(similarity_matrix[:50, :50].numpy(), cmap='YlOrRd', square=True)
plt.title("Similarity Matrix - Good alignment shows strong diagonal")
plt.savefig("similarity_good.png")
```

Expected pattern:
- **Bright diagonal** (high similarity for paired samples)
- **Dark off-diagonal** (low similarity for unpaired samples)

---

## Integration with MotionGPT Training

### Add Retrieval Evaluation to Validation

Modify `mGPT/models/mgpt.py`:

```python
def val_lamp_forward(self, batch):
    """
    Validation for LaMP stage.

    Unlike generation stages, we don't generate motion.
    Instead, we extract embeddings for retrieval evaluation.
    """
    feats_ref = batch["motion"]
    texts = batch["text"]

    # Extract embeddings
    text_emb = self.lm.encode_text(texts)
    motion_emb = self.lm.encode_motion(feats_ref)

    return {
        'text_emb': text_emb,
        'motion_emb': motion_emb,
        'texts': texts
    }
```

Add to `allsplit_step()`:

```python
elif self.hparams.stage == "lamp":
    if split == "train":
        rs_set = self.train_lamp_forward(batch)
        loss = self._losses['losses_' + split].update(rs_set)
    elif split == "val":
        rs_set = self.val_lamp_forward(batch)
        # Accumulate embeddings for end-of-epoch retrieval eval
        self.val_text_embs.append(rs_set['text_emb'].cpu())
        self.val_motion_embs.append(rs_set['motion_emb'].cpu())
```

Add to `on_validation_epoch_end()`:

```python
if self.hparams.stage == "lamp":
    # Compute retrieval metrics
    text_embs = torch.cat(self.val_text_embs, dim=0)
    motion_embs = torch.cat(self.val_motion_embs, dim=0)

    # R-Precision
    similarity = torch.mm(text_embs, motion_embs.t())
    ranked = similarity.argsort(dim=1, descending=True)
    R1 = sum(i in ranked[i, :1] for i in range(len(text_embs))) / len(text_embs)
    R5 = sum(i in ranked[i, :5] for i in range(len(text_embs))) / len(text_embs)

    # Log
    self.log("Metrics/R@1", R1)
    self.log("Metrics/R@5", R5)

    # Clear
    self.val_text_embs = []
    self.val_motion_embs = []
```

---

## Summary

### Key Takeaways

1. **LaMP ≠ Generation**
   - Evaluates embedding alignment, not motion quality
   - Use retrieval metrics, not FID/MPJPE

2. **Main Metrics**
   - R-Precision (R@1, R@5, R@10): Higher is better
   - Matching Score: Lower is better
   - Diversity: Higher is better (but not too high)

3. **Typical Values**
   - R@1: 0.40-0.45 (good alignment)
   - R@5: 0.70-0.75
   - Matching Score: 10-12
   - Diversity: 11-12

4. **Evaluation Script**
   - Extract embeddings for entire validation set
   - Compute similarity matrix
   - Calculate retrieval metrics
   - Visualize alignment (heatmaps, t-SNE)

5. **Integration**
   - Add `encode_text()` and `encode_motion()` to LaMP model
   - Run retrieval eval every N epochs
   - Log to wandb for monitoring

---

## Next Steps

After evaluating LaMP:

1. **If retrieval metrics are good (R@1 > 0.35):**
   - ✅ Proceed to Stage 3 (Masked T2M)
   - Use frozen QFormer for text encoding

2. **If retrieval metrics are poor (R@1 < 0.20):**
   - ❌ Debug LaMP training
   - Check loss curves, embedding distributions
   - May need longer training or hyperparameter tuning

3. **Use LaMP for downstream tasks:**
   - Text-to-motion retrieval
   - Motion-to-text retrieval
   - Motion captioning
   - Cross-modal search

---

**Document Version:** 1.0
**Last Updated:** 2026-02-10
**Status:** Complete ✅

LaMP evaluation is fundamentally different from T2M generation - it's all about learning good embeddings for retrieval!
