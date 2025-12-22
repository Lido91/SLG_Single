# Text-to-Motion Transformer Implementation Report
## UniMuMo Architecture Adaptation for MotionGPT

---

## 1. Implementation Checklist

### ✅ Checklist 1: Compatibility with `get_motion_code.py` Tokens

| Item | Status | Details |
|------|--------|---------|
| Token Format | ✅ PASS | Expects `[B, T', num_quantizers]` from RVQ, our model expects `[B, K, T]` where K=num_quantizers |
| Codebook Count | ✅ PASS | `n_q=6` matches MotionGPT's 6-quantizer RVQ |
| Vocabulary Size | ✅ PASS | `card=512` matches MotionGPT's codebook size |
| Token Range | ✅ PASS | Tokens in range `[0, 511]`, special token is `512` |
| Data Shape Conversion | ⚠️ NOTE | `get_motion_code.py` outputs `[B, T', K]`, model expects `[B, K, T]` - need `.permute(0, 2, 1)` |

**Required Data Transformation:**
```python
# From get_motion_code.py output: [B, T', num_quantizers] = [B, T, 6]
# To model input: [B, K, T] = [B, 6, T]
codes = codes.permute(0, 2, 1)  # [B, T, K] -> [B, K, T]
```

---

### ✅ Checklist 2: Implementation Comparison with UniMuMo

| Component | UniMuMo (`mm_lm.py`) | Our Implementation | Match |
|-----------|---------------------|-------------------|-------|
| **Per-Codebook Embeddings** | `self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim) for _ in range(n_q)])` (L115) | `self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])` (L324) | ✅ |
| **Embedding Summation** | `sum([self.emb[k](sequence[:, k]) for k in range(K)])` (L187) | `sum([self.emb[k](codes[:, k]) for k in range(K)])` (L405) | ✅ |
| **Per-Codebook Output Heads** | `self.linears = nn.ModuleList([nn.Linear(dim, self.card) for _ in range(n_q)])` (L129) | `self.linears = nn.ModuleList([nn.Linear(dim, card, bias=bias_proj) for _ in range(n_q)])` (L343) | ✅ |
| **Output Logits Stacking** | `torch.stack([self.linears[k](out) for k in range(K)], dim=1)` (L216) | `torch.stack([self.linears[k](out) for k in range(K)], dim=1)` (L426) | ✅ |
| **ScaledEmbedding Class** | Lines 73-82 | Lines 68-78 | ✅ |
| **Special Token ID** | `self.card` (L168-169) | `self.card` (L374-376) | ✅ |
| **Weight Initialization** | `get_init_fn`, `init_layer` (L29-70) | `get_init_fn`, `init_layer` (L24-65) | ✅ |
| **CFG Dropout** | `ClassifierFreeGuidanceDropout(p=cfg_dropout)` (L104) | `CFGDropout(p=cfg_dropout)` in conditioner | ✅ |
| **CFG Inference** | `logits = uncond + cfg_coef * (cond - uncond)` (L383-384) | `logits = uncond + cfg_coef * (cond - uncond)` (L335, tm_model.py) | ✅ |
| **Codebook Pattern** | `DelayedPatternProvider` with `build_pattern_sequence` | `DelayedPatternProvider` with `build_pattern_sequence` | ✅ |
| **Pattern Revert** | `revert_pattern_logits` (L269-278) | `revert_pattern_logits` (L198-201, tm_model.py) | ✅ |
| **Causal Mask** | `get_self_attn_mask` (L331-347) | `get_causal_mask` (L430-434) | ✅ |
| **Cross-Attention** | T5 embeddings via `ConditionFuser` | T5 embeddings via `ConditionProvider` | ✅ |
| **T5 Encoder** | `T5EncoderModel` (conditioners.py L409) | `T5EncoderModel` (tm_conditioner.py L88) | ✅ |
| **Output Projection** | `self.output_proj = nn.Linear(dim, output_dim)` | `self.output_proj = nn.Linear(dim, output_dim)` | ✅ |

#### Key Differences (Intentional):

| Aspect | UniMuMo | Our Implementation | Reason |
|--------|---------|-------------------|--------|
| Modalities | Music + Motion (dual) | Motion only | MotionGPT is motion-only |
| Music Embeddings | `self.emb` + `self.motion_emb` | `self.emb` only | No music modality |
| Music Output Heads | `self.linears` + `self.motion_linears` | `self.linears` only | No music modality |
| Sequence Splitting | `S//2` for music/motion | Full sequence | Single modality |
| Default n_q | 4 (MusicGen) | 6 (MotionGPT RVQ) | Different VQ-VAE |
| Default card | 1024 | 512 | Different codebook size |

---

## 2. File Structure

```
mGPT/
├── archs/
│   ├── __init__.py                 # Updated with new exports
│   ├── tm_transformer.py           # Core transformer architecture
│   ├── tm_conditioner.py           # T5 conditioner and CFG dropout
│   ├── tm_codebook_patterns.py     # Delayed pattern provider
│   └── tm_model.py                 # Complete Text-to-Motion model
├── models/
│   ├── __init__.py                 # Updated with new exports
│   └── tm_lightning.py             # PyTorch Lightning training wrapper
└── configs/
    └── tm_transformer.yaml         # Training configuration
```

---

## 3. Core Components Implementation Details

### 3.1 Per-Codebook Embeddings (`tm_transformer.py:324-327`)

```python
# Per-codebook embedding tables
self.emb = nn.ModuleList([
    ScaledEmbedding(embed_dim, dim, lr=emb_lr, **factory_kwargs)
    for _ in range(n_q)
])
```

**UniMuMo Reference** (`mm_lm.py:115`):
```python
self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
```

### 3.2 Embedding Summation (`tm_transformer.py:404-405`)

```python
# Sum embeddings from all codebooks
input_ = sum([self.emb[k](codes[:, k]) for k in range(K)])  # [B, T, dim]
```

**UniMuMo Reference** (`mm_lm.py:187-189`):
```python
music_input = sum([self.emb[k](sequence[:, k, :S//2]) for k in range(K)])
motion_input = sum([self.motion_emb[k](sequence[:, k, S//2:]) for k in range(K)])
input_ = torch.cat((music_input, motion_input), dim=1)
```

### 3.3 Per-Codebook Output Heads (`tm_transformer.py:343-346`)

```python
# Per-codebook output heads
self.linears = nn.ModuleList([
    nn.Linear(dim, card, bias=bias_proj, **factory_kwargs)
    for _ in range(n_q)
])
```

**UniMuMo Reference** (`mm_lm.py:129-131`):
```python
self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
self.motion_linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
```

### 3.4 Output Logits Computation (`tm_transformer.py:425-426`)

```python
# Per-codebook output projections
logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, T, card]
```

**UniMuMo Reference** (`mm_lm.py:216-217`):
```python
music_logits = torch.stack([self.linears[k](out[:, :S // 2]) for k in range(K)], dim=1)
motion_logits = torch.stack([self.motion_linears[k](out[:, S // 2:]) for k in range(K)], dim=1)
```

### 3.5 T5 Cross-Attention (`tm_conditioner.py:83-97`)

```python
def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    mask = inputs['attention_mask']
    with torch.set_grad_enabled(self.finetune):
        with self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
    embeds = self.output_proj(embeds.to(self.output_proj.weight.dtype))
    embeds = embeds * mask.unsqueeze(-1)
    return embeds, mask
```

**UniMuMo Reference** (`conditioners.py:448-454`):
```python
def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
    mask = inputs['attention_mask']
    with torch.set_grad_enabled(self.finetune), self.autocast:
        embeds = self.t5(**inputs).last_hidden_state
    embeds = self.output_proj(embeds.to(self.output_proj.weight))
    embeds = (embeds * mask.unsqueeze(-1))
    return embeds, mask
```

### 3.6 CFG Dropout (`tm_conditioner.py:109-128`)

```python
class CFGDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, condition, mask):
        if not self.training or self.p == 0:
            return condition, mask
        B = condition.shape[0]
        drop_mask = torch.rand(B, device=condition.device) < self.p
        condition = condition * (~drop_mask).float().view(B, 1, 1)
        mask = mask * (~drop_mask).float().view(B, 1)
        return condition, mask
```

**UniMuMo Reference** (`conditioners.py:543-576`):
```python
class ClassifierFreeGuidanceDropout(DropoutModule):
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples):
        if not self.training:
            return samples
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples
        # Nullify ALL conditions
        samples = deepcopy(samples)
        ...
```

### 3.7 CFG Inference (`tm_model.py:332-335`)

```python
# Apply CFG
if cfg_coef > 1.0:
    cond_logits, uncond_logits = logits.split(B, dim=0)
    logits = uncond_logits + cfg_coef * (cond_logits - uncond_logits)
```

**UniMuMo Reference** (`mm_lm.py:380-387`):
```python
if condition_tensors:
    music_cond_logits, music_uncond_logits = music_all_logits.split(B, dim=0)
    motion_cond_logits, motion_uncond_logits = motion_all_logits.split(B, dim=0)
    music_logits = music_uncond_logits + (music_cond_logits - music_uncond_logits) * cfg_coef
    motion_logits = motion_uncond_logits + (motion_cond_logits - motion_uncond_logits) * cfg_coef
```

### 3.8 Delayed Pattern Provider (`tm_codebook_patterns.py:168-214`)

```python
class DelayedPatternProvider(CodebooksPatternProvider):
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None, ...):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))  # [0, 1, 2, 3, 4, 5] for n_q=6
        self.delays = delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)
```

**UniMuMo Reference** (`codebooks_patterns.py:303-353`) - Identical logic.

### 3.9 Cross-Entropy Loss Per Codebook (`tm_model.py:206-247`)

```python
def compute_loss(self, codes, texts):
    output = self.compute_predictions(codes, texts, apply_cfg_dropout=True)
    B, K, T = codes.shape
    total_loss = torch.zeros([], device=codes.device)
    loss_per_codebook = []

    for k in range(K):
        logits_k = output.logits[:, k].contiguous().view(-1, self.card)
        targets_k = codes[:, k].contiguous().view(-1)
        mask_k = output.mask[:, k].contiguous().view(-1)
        valid_logits = logits_k[mask_k]
        valid_targets = targets_k[mask_k]
        if valid_targets.numel() > 0:
            loss_k = F.cross_entropy(valid_logits, valid_targets)
            total_loss = total_loss + loss_k
            loss_per_codebook.append(loss_k.detach())
    total_loss = total_loss / K
    return total_loss, loss_per_codebook
```

**UniMuMo Reference** (`transformer_model.py:343-362`) - Same per-codebook loss computation logic.

---

## 4. Data Flow Diagram

```
Input: Text descriptions + Motion codes [B, K, T]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    TextToMotionLM                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ConditionProvider                        │   │
│  │  ┌───────────────┐    ┌────────────────┐             │   │
│  │  │ T5Conditioner │ -> │  CFGDropout    │             │   │
│  │  │ (T5Encoder +  │    │ (10% dropout)  │             │   │
│  │  │  projection)  │    └───────┬────────┘             │   │
│  │  └───────────────┘            │                      │   │
│  │                               ▼                      │   │
│  │                    T5 Embeddings [B, L, D]           │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            PatternProvider (Delayed)                  │   │
│  │  codes [B, K, T] -> pattern_sequence [B, K, S]       │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    MotionLM                           │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Per-Codebook Embeddings (6 × ScaledEmbedding)  │  │   │
│  │  │ codes[:, k] -> emb[k] -> SUM over k            │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                         │                            │   │
│  │                         ▼                            │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │           MotionTransformer                    │  │   │
│  │  │  12 × TransformerLayer:                        │  │   │
│  │  │    - Self-Attention (causal)                   │  │   │
│  │  │    - Cross-Attention (T5 embeddings)           │  │   │
│  │  │    - FFN                                       │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                         │                            │   │
│  │                         ▼                            │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Per-Codebook Output Heads (6 × Linear)         │  │   │
│  │  │ out -> linears[k] -> STACK over k              │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│              Logits [B, K, S, 512]                           │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Pattern Revert (if training)                │   │
│  │  logits [B, K, S, 512] -> [B, K, T, 512]             │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│               Cross-Entropy Loss (per codebook)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Generation Flow (Inference)

```
Input: Text descriptions
          │
          ▼
    ┌─────────────┐
    │  Prepare    │
    │  CFG Cond   │
    │  [cond;     │
    │   uncond]   │
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │  Initialize │
    │  gen_codes  │
    │  [B,K,T]    │
    │  = special  │
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │  Build      │
    │  Pattern    │
    │  Sequence   │
    └─────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  For offset in 1..gen_sequence_len: │
    │    1. Get curr_sequence[:offset]    │
    │    2. Duplicate for CFG             │
    │    3. Forward through MotionLM      │
    │    4. Get last position logits      │
    │    5. Apply CFG:                    │
    │       logits = uncond +             │
    │         cfg_coef*(cond-uncond)      │
    │    6. Sample next token             │
    │       (temp, top_k, top_p)          │
    │    7. Update gen_sequence           │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────┐
    │  Revert     │
    │  Pattern    │
    │  Sequence   │
    └─────────────┘
          │
          ▼
    Output: Motion codes [B, K, T]
```

---

## 6. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_q` | 6 | Number of RVQ codebooks |
| `card` | 512 | Codebook vocabulary size |
| `dim` | 768 | Transformer hidden dimension |
| `num_heads` | 12 | Attention heads |
| `num_layers` | 12 | Transformer layers |
| `hidden_scale` | 4 | FFN multiplier (dim × 4 = 3072) |
| `dropout` | 0.1 | Dropout rate |
| `t5_name` | "google/flan-t5-base" | T5 model name |
| `t5_finetune` | False | Whether to finetune T5 |
| `cfg_dropout` | 0.1 | CFG dropout probability |
| `cfg_coef` | 3.0 | CFG coefficient at inference |
| `pattern_type` | "delayed" | Codebook pattern type |
| `lr` | 1e-4 | Learning rate |
| `warmup_steps` | 2000 | LR warmup steps |
| `max_steps` | 200000 | Total training steps |

---

## 7. Usage Example

### Training
```python
from mGPT.models import TextToMotionLightning
import pytorch_lightning as pl

# Create model
model = TextToMotionLightning(
    n_q=6,
    card=512,
    dim=768,
    num_layers=12,
    t5_name="google/flan-t5-base",
    cfg_dropout=0.1,
)

# Training step
batch = {
    'motion_codes': codes,  # [B, K, T] = [B, 6, T]
    'texts': ["A person walks forward", "A person waves hand"],
}
loss = model.training_step(batch, batch_idx=0)
```

### Generation
```python
# Generate motion codes from text
model.eval()
texts = ["A person walks forward slowly"]
generated_codes = model.generate(
    texts=texts,
    max_gen_len=256,
    use_sampling=True,
    temp=1.0,
    top_k=250,
    cfg_coef=3.0,
)
# generated_codes: [B, K, T] = [1, 6, 256]

# Then decode with VQ-VAE:
# motion = vae.decode(generated_codes)
```

---

## 8. Training Instructions

### 8.1 File Structure (Complete)

```
MotionGPT/
├── mGPT/
│   ├── archs/
│   │   ├── __init__.py                 # Updated with new exports
│   │   ├── tm_transformer.py           # Core transformer architecture
│   │   ├── tm_conditioner.py           # T5 conditioner and CFG dropout
│   │   ├── tm_codebook_patterns.py     # Delayed pattern provider
│   │   └── tm_model.py                 # Complete Text-to-Motion model
│   ├── models/
│   │   ├── __init__.py                 # Updated with new exports
│   │   └── tm_lightning.py             # PyTorch Lightning training wrapper
│   │                                   # + tm_collate_fn for H2S data format
│   └── data/
│       ├── H2S.py                      # H2SDataModule (uses Text2MotionDatasetCB)
│       └── humanml/
│           ├── dataset_t2m_cb.py       # Text2MotionDatasetCB (loads tokens)
│           └── load_data.py            # load_h2s_sample, load_csl_sample, etc.
├── configs/
│   ├── tm_h2s.yaml                     # Training config for How2Sign
│   └── tm_transformer.yaml             # General training config
├── train_tm_transformer.py             # Training script
└── get_motion_code.py                  # Token extraction script
```

### 8.2 Prerequisites

1. **Pre-compute motion tokens** using the VQ-VAE:
   ```bash
   python get_motion_code.py --config configs/deto_h2s_rvq.yaml
   ```

   This will save tokens to:
   - `{data_root}/TOKENS_h2s_csl_phoenix/how2sign_whole/how2sign/{sample_name}.npy`
   - Shape: `[1, T, K]` where K=6 codebooks

2. **Verify token files exist**:
   ```bash
   ls ../SOKE/data/How2Sign/TOKENS_h2s_csl_phoenix/how2sign_whole/how2sign/
   # Should show .npy files for each sample
   ```

3. **Check T5 model availability**:
   ```bash
   # Either download from HuggingFace (requires internet)
   # Or use local path: "flan-t5-base/"
   ```

### 8.3 Training Command

```bash
python train_tm_transformer.py --config configs/tm_h2s.yaml
```

### 8.4 Config Options (configs/tm_h2s.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TM_MODEL.N_Q` | 6 | Number of codebooks (must match VQ-VAE) |
| `TM_MODEL.CARD` | 512 | Vocabulary size per codebook |
| `TM_MODEL.DIM` | 768 | Transformer hidden dimension |
| `TM_MODEL.NUM_LAYERS` | 12 | Number of transformer layers |
| `TM_MODEL.NUM_HEADS` | 12 | Number of attention heads |
| `TM_MODEL.HIDDEN_SCALE` | 4 | FFN hidden dim = DIM × HIDDEN_SCALE |
| `TM_MODEL.DROPOUT` | 0.1 | Dropout rate |
| `TM_MODEL.T5_NAME` | "google/flan-t5-base" | T5 model for conditioning |
| `TM_MODEL.T5_FINETUNE` | false | Whether to finetune T5 encoder |
| `TM_MODEL.CFG_DROPOUT` | 0.1 | CFG dropout rate (10%) |
| `TM_MODEL.CFG_COEF` | 3.0 | CFG coefficient at inference |
| `TM_MODEL.PATTERN_TYPE` | "delayed" | Codebook pattern type |
| `TM_MODEL.LR` | 1e-4 | Learning rate |
| `TM_MODEL.WARMUP_STEPS` | 2000 | Linear warmup steps |
| `TM_MODEL.MAX_STEPS` | 200000 | Max training steps |
| `DATASET.CODE_PATH` | `TOKENS_h2s.../how2sign_whole` | Path to pre-computed tokens |
| `TRAIN.BATCH_SIZE` | 32 | Training batch size |
| `TRAIN.END_EPOCH` | 100 | Number of training epochs |

### 8.5 Data Flow (Training)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Pipeline                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  H2SDataModule (configs/tm_h2s.yaml)                                        │
│       │                                                                     │
│       ▼                                                                     │
│  Text2MotionDatasetCB (mGPT/data/humanml/dataset_t2m_cb.py)                │
│       │                                                                     │
│       │  __getitem__ returns tuple:                                         │
│       │  (caption, m_tokens[T,K], m_length, name, None, None, None,        │
│       │   all_captions, tasks, src)                                         │
│       │                                                                     │
│       ▼                                                                     │
│  tm_collate_fn (mGPT/models/tm_lightning.py:244-277)                       │
│       │                                                                     │
│       │  - Extract captions and tokens from batch tuple                     │
│       │  - Pad tokens to max length                                         │
│       │  - Transpose: [T, K] -> [K, T]                                      │
│       │                                                                     │
│       ▼                                                                     │
│  Output: {'motion_codes': [B, K, T], 'texts': List[str]}                   │
│       │                                                                     │
│       ▼                                                                     │
│  TextToMotionLightning.training_step                                        │
│       │                                                                     │
│       ▼                                                                     │
│  TextToMotionLM.forward (computes per-codebook CE loss)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.6 Token Shape Conversion

```python
# Token loading in load_h2s_sample (load_data.py:67-71):
code = np.load(fname)[0]  # Shape: [T, K] (from [1, T, K])

# Dataset returns: (caption, m_tokens, ...)
# m_tokens shape: [T, K] (after torch.from_numpy)

# tm_collate_fn conversion (tm_lightning.py:267-272):
# Pad and transpose: [T, K] -> [K, T]
padded_codes = torch.zeros(len(batch), K, max_len, dtype=torch.long)
for i, tokens in enumerate(tokens_list):
    T = tokens.shape[0]
    # tokens is [T, K], transpose to [K, T]
    padded_codes[i, :, :T] = tokens.permute(1, 0)

# Output: [B, K, T] where K=6 codebooks
```

### 8.7 Expected Output

- **Checkpoints**: `experiments/tm_transformer/TM-H2S/checkpoints/`
  - `epoch={epoch}-step={step}-val_loss={val_loss}.ckpt`
  - `last.ckpt`
- **Logs**: Weights & Biases (if configured)
- **Metrics logged**:
  - `train/loss` - Total training loss
  - `train/loss_cb{0-5}` - Per-codebook loss
  - `val/loss` - Total validation loss
  - `val/loss_cb{0-5}` - Per-codebook validation loss

### 8.8 Multi-Dataset Training

To train on multiple datasets (How2Sign + CSL + Phoenix):

```yaml
# In configs/tm_h2s.yaml:
DATASET:
  H2S:
    DATASET_NAME: how2sign_csl_phoenix  # Change from 'how2sign'
    # ... other settings remain the same
```

### 8.9 Troubleshooting

#### Codebook Count Mismatch Error

If you see an error like:
```
ValueError: Input codes have X codebooks, but model expects 6.
```

This means `TM_MODEL.N_Q` in your config doesn't match the number of codebooks in your pre-computed tokens.

**Solution:**
1. Check the token shape printed at startup:
   ```
   [tm_collate_fn] First token shape: torch.Size([T, K]) (expected [T, K])
   ```
2. Update `TM_MODEL.N_Q` in `configs/tm_h2s.yaml` to match `K`

#### Data Path Issues

If tokens are not found, verify:
1. `DATASET.CODE_PATH` points to the correct directory
2. Token files exist at: `{ROOT}/{CODE_PATH}/{src}/{name}.npy`
3. The `src` subdirectory matches your dataset (e.g., `how2sign/`)

#### Validation with On-the-Fly VAE Decoding

The training script now supports validation with on-the-fly VAE decoding (similar to `mgpt.py`):

**How it works:**
1. **Training**: Uses pre-computed tokens from `Text2MotionDatasetCB` with `tm_collate_fn`
2. **Validation**: Uses raw motion from `Text2MotionDatasetEval` with `tm_val_collate_fn`
3. **Validation flow**:
   - Generate motion tokens from text using TM Transformer
   - Decode tokens to motion using VQ-VAE (`self.vae.decode()`)
   - Compare against reference motion

**Requirements for validation:**
1. Set `TRAIN.PRETRAINED_VAE` to your VQ-VAE checkpoint path
2. Include `vq.default` config in your YAML (matching the VAE architecture)
3. The VAE checkpoint should match the one used to generate pre-computed tokens

**Config example:**
```yaml
TRAIN:
  PRETRAINED_VAE: experiments/mgpt/DETO_RVQ_wholebody/checkpoints/your_vae.ckpt

vq:
  default:
    target: mGPT.archs.mgpt_rvq.RVQVae
    params:
      num_quantizers: 6
      code_num: 512
      # ... other params matching your VAE
```

**If VAE is not configured:**
- Validation will be automatically disabled (`limit_val_batches=0`)
- Training runs without validation epochs
- Log message: "Validation DISABLED (no VAE configured)"

---

## 9. Conclusion

The implementation successfully adapts UniMuMo's transformer architecture for motion-only generation in MotionGPT. All core components (per-codebook embeddings, per-codebook output heads, T5 cross-attention, CFG, delayed codebook pattern) are implemented following the UniMuMo design patterns, with appropriate modifications for the single-modality (motion-only) use case.

The model is compatible with the motion tokens exported by `get_motion_code.py` after a simple dimension permutation (`[B, T, K] -> [B, K, T]`), which is handled automatically by `tm_collate_fn`.

### Key Files for Training:
1. **Training script**: `train_tm_transformer.py`
2. **Config file**: `configs/tm_h2s.yaml`
3. **Collate function**: `mGPT/models/tm_lightning.py:tm_collate_fn`
4. **Model**: `mGPT/models/tm_lightning.py:TextToMotionLightning`
5. **Core model**: `mGPT/archs/tm_model.py:TextToMotionLM`

### Recent Updates:
- Added codebook count validation in `tm_model.py:compute_predictions()` with clear error message
- Added debug output in `tm_collate_fn` to print first token shape for verification
- Training script follows same pattern as `train.py` for compatibility with existing pipeline
- **Added on-the-fly VAE decoding for validation** (like `mgpt.py`):
  - `TextToMotionLightning` now accepts `vae_cfg` parameter
  - Validation uses `Text2MotionDatasetEval` (raw motion) + VAE decoding
  - Added `tm_val_collate_fn` for validation data format
  - Validation automatically enabled when `TRAIN.PRETRAINED_VAE` and `vq.default` are configured
