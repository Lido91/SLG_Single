# UniMuMo Transformer Architecture Adaptation Plan for MotionGPT
## Enriched with Implementation Details from UniMuMo Codebase

---

## 1. Key Source Files in UniMuMo

| File Path | Purpose |
|-----------|---------|
| `unimumo/audio/audiocraft_/models/mm_lm.py` | **Main LM Model** - Per-codebook embeddings, per-codebook output heads, CFG, generation |
| `unimumo/audio/audiocraft_/modules/transformer.py` | **Transformer Architecture** - StreamingTransformer, cross-attention, custom attention masks |
| `unimumo/audio/audiocraft_/modules/conditioners.py` | **T5 Conditioning** - T5Conditioner, ConditionFuser, CFG dropout |
| `unimumo/audio/audiocraft_/modules/codebooks_patterns.py` | **Codebook Patterns** - DelayedPatternProvider for interleaving codebooks |
| `unimumo/audio/audiocraft_/models/builders.py` | **Model Construction** - Factory functions to build LM, conditioners, pattern providers |
| `unimumo/audio/audiocraft_/models/loaders.py` | **Model Loading** - Loading pretrained weights, initializing motion components |
| `unimumo/audio/audiocraft_/quantization/core_vq.py` | **RVQ Implementation** - ResidualVectorQuantization class |
| `unimumo/models/transformer_model.py` | **Training Wrapper** - MusicMotionTransformer PyTorch Lightning module |
| `unimumo/models/motion_vqvae.py` | **Motion VQVAE** - Encoder/decoder with shared RVQ codebook |

---

## 2. Per-Codebook Embeddings (Critical Implementation)

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:115-116`

```python
# UniMuMo uses SEPARATE embedding tables for each codebook (n_q codebooks)
# Each embedding table has size (card + 1) x dim, where +1 is for special token

self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
self.motion_emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
```

**Key Details:**
- `embed_dim = self.card + 1` (codebook size + 1 for special token)
- `dim` = transformer hidden dimension (1024 for musicgen-small)
- `n_q` = number of codebooks (4 in UniMuMo)
- Separate embedding tables for music and motion

**Embedding Summation (Forward Pass):**

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:187-189`

```python
# Embeddings from all codebooks are SUMMED together
music_input = sum([self.emb[k](sequence[:, k, :S//2]) for k in range(K)])  # [B, S//2, dim]
motion_input = sum([self.motion_emb[k](sequence[:, k, S//2:]) for k in range(K)])  # [B, S//2, dim]
input_ = torch.cat((music_input, motion_input), dim=1)  # [B, S, dim]
```

**For MotionGPT Adaptation (6 codebooks):**
```python
self.motion_emb = nn.ModuleList([
    ScaledEmbedding(513, hidden_dim, lr=emb_lr)  # 512 + 1 special token
    for _ in range(6)  # 6 codebooks
])

# In forward:
motion_input = sum([self.motion_emb[k](codes[:, k]) for k in range(6)])
```

---

## 3. Per-Codebook Output Heads (Critical Implementation)

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:129-131`

```python
# Separate linear projection for each codebook
self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
self.motion_linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
```

**Output Logits Computation:**

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:216-217`

```python
# Stack logits from all codebook heads
music_logits = torch.stack([self.linears[k](out[:, :S // 2]) for k in range(K)], dim=1)  # [B, K, S/2, card]
motion_logits = torch.stack([self.motion_linears[k](out[:, S // 2:]) for k in range(K)], dim=1)  # [B, K, S/2, card]
```

**For MotionGPT Adaptation (6 codebooks x 512 vocab):**
```python
self.motion_linears = nn.ModuleList([
    nn.Linear(hidden_dim, 512, bias=True)  # 512 codebook size
    for _ in range(6)  # 6 codebooks
])

# In forward:
motion_logits = torch.stack([
    self.motion_linears[k](transformer_out)
    for k in range(6)
], dim=1)  # [B, 6, T, 512]
```

---

## 4. Cross-Attention Conditioning with T5

### 4.1 T5 Conditioner

**File:** `unimumo/audio/audiocraft_/modules/conditioners.py:354-455`

```python
class T5Conditioner(TextConditioner):
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        # ...
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
        t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
        # ...
        self.output_proj = nn.Linear(dim, output_dim)  # Project T5 dim to transformer dim

    def tokenize(self, x: tp.List[tp.Optional[str]], device) -> tp.Dict[str, torch.Tensor]:
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(device)
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask  # (condition_tensor, attention_mask)
```

### 4.2 Condition Fuser

**File:** `unimumo/audio/audiocraft_/modules/conditioners.py:775-864`

```python
class ConditionFuser(StreamingModule):
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]],
                 cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        # fuse2cond example: {'cross': ['description'], 'prepend': [], 'sum': [], 'input_interpolate': []}
        self.fuse2cond = fuse2cond

    def forward(self, input: torch.Tensor, conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == 'cross':
                cross_attention_output = cond  # T5 embeddings for cross-attention
            # ...
        return input, cross_attention_output
```

### 4.3 Cross-Attention in Transformer Layer

**File:** `unimumo/audio/audiocraft_/modules/transformer.py:573-586`

```python
# In StreamingTransformerLayer.__init__:
if cross_attention:
    self.cross_attention = StreamingMultiheadAttention(
        cross_attention=True, qk_layer_norm=qk_layer_norm_cross,
        **attn_kwargs, **factory_kwargs)
    self.dropout_cross = nn.Dropout(dropout)
    self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
```

**Cross-Attention Forward:**

**File:** `unimumo/audio/audiocraft_/modules/transformer.py:596-603`

```python
def _cross_attention_block(self, src: torch.Tensor,
                           cross_attention_src: torch.Tensor,
                           cross_attn_mask: torch.Tensor) -> torch.Tensor:
    # queries are from src (transformer hidden states)
    # keys and values from cross_attention_src (T5 embeddings)
    x = self.cross_attention(
        src, cross_attention_src, cross_attention_src,
        need_weights=False, cross_attn_mask=cross_attn_mask)[0]
    return self.dropout_cross(x)
```

---

## 5. Classifier-Free Guidance (CFG) Implementation

### 5.1 CFG Dropout (Training)

**File:** `unimumo/audio/audiocraft_/modules/conditioners.py:543-579`

```python
class ClassifierFreeGuidanceDropout(DropoutModule):
    """All attributes are dropped with the same probability."""
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p  # e.g., 0.1 = 10% dropout

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        if not self.training:
            return samples

        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # Nullify ALL conditions
        samples = deepcopy(samples)
        for condition_type in ["wav", "text"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples
```

### 5.2 CFG at Inference

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:349-418`

```python
def _sample_next_token(
    self,
    music_sequence: torch.LongTensor,
    motion_sequence: torch.LongTensor,
    cfg_conditions: CFGConditions,
    mode: str,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    cfg_coef: tp.Optional[float] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    B = music_sequence.shape[0]
    cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

    # ...

    # KEY: Duplicate sequence for conditional and unconditional
    if condition_tensors:
        sequence = torch.cat([sequence, sequence], dim=0)

    music_all_logits, motion_all_logits = self(
        sequence, conditions=[], condition_tensors=condition_tensors,
        src_mask=src_mask, cross_attn_mask=cross_attn_mask
    )

    # CFG formula: logits = uncond + cfg_coef * (cond - uncond)
    if condition_tensors:
        music_cond_logits, music_uncond_logits = music_all_logits.split(B, dim=0)
        motion_cond_logits, motion_uncond_logits = motion_all_logits.split(B, dim=0)
        music_logits = music_uncond_logits + (music_cond_logits - music_uncond_logits) * cfg_coef
        motion_logits = motion_uncond_logits + (motion_cond_logits - motion_uncond_logits) * cfg_coef
```

**CFG Condition Preparation:**

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:449-478`

```python
# In generate():
if conditions:
    music_conditions = conditions[:num_samples]
    motion_conditions = conditions[num_samples:]

    # Create null conditions by dropping everything
    null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
    null_music_conditions = null_conditions[:num_samples]
    null_motion_conditions = null_conditions[num_samples:]

    # Concatenate: [conditional, unconditional]
    music_conditions = music_conditions + null_music_conditions
    motion_conditions = motion_conditions + null_motion_conditions

    # Encode conditions
    tokenized_music = self.condition_provider.tokenize(music_conditions, device)
    condition_tensor_music = self.condition_provider(tokenized_music)
    # ...

    # Merge for cross-attention
    condition_tensor = torch.cat([
        condition_tensor_music['description'][0],
        condition_tensor_motion['description'][0]
    ], dim=1)  # [B*2, L_music+L_motion, D]
```

---

## 6. Custom Attention Masks

### 6.1 Self-Attention Mask

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:331-347`

```python
def get_self_attn_mask(self, section_1: int, section_2: int, mode: str) -> torch.Tensor:
    device = next(iter(self.parameters())).device
    mask = torch.zeros((section_1 + section_2, section_1 + section_2), dtype=torch.bool, device=device)

    if mode in ['music_caption', 'motion_caption']:
        # Bidirectional attention within each modality, no cross-modal attention
        mask[:section_1, :section_1] = True
        mask[section_1:, section_1:] = True
    else:
        assert mode in ['music_motion', 'music2motion', 'motion2music']
        # Causal attention mask (lower triangular)
        mask[:section_1, :section_1] = ~torch.ones((section_1, section_1), dtype=torch.bool, device=device).triu(1)
        mask[section_1:section_1 + section_2, :section_1] = ~torch.ones((section_2, section_1), dtype=torch.bool, device=device).triu(1)
        mask[:section_1, section_1:section_1 + section_2] = ~torch.ones((section_1, section_2), dtype=torch.bool, device=device).triu(1)
        mask[section_1:section_1 + section_2, section_1:section_1 + section_2] = ~torch.ones((section_2, section_2), dtype=torch.bool, device=device).triu(1)

    mask = torch.where(mask, 0., float('-inf'))
    return mask
```

### 6.2 Cross-Attention Mask (Music/Motion to Condition)

**File:** `unimumo/audio/audiocraft_/modules/transformer.py:34-66`

```python
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, cross_attn_mask=None
) -> torch.Tensor:
    # ...
    attn_bias = attn_bias[None, ...]  # [L_feature, L_cond] -> [1, L_feature, L_cond]
    if cross_attn_mask is not None:
        assert attn_mask is None
        attn_bias = attn_bias.repeat(query.shape[0], 1, 1)  # [B, L_feature, L_cond]
        feature_len = attn_bias.shape[1]
        # First half attends to music condition, second half to motion condition
        attn_bias[:, :feature_len//2, :] += cross_attn_mask[:, :1, :]
        attn_bias[:, feature_len//2:, :] += cross_attn_mask[:, 1:, :]
    # ...
```

**Cross-Attention Mask Construction:**

**File:** `unimumo/models/transformer_model.py:404-413`

```python
# Cross-attention mask: [B, 2, L_music+L_motion]
# Row 0: mask for music tokens (attend to music condition only)
# Row 1: mask for motion tokens (attend to motion condition only)
condition_mask = torch.zeros(
    (music_condition_mask.shape[0], 2, music_condition_mask.shape[-1] + motion_condition_mask.shape[-1]),
    dtype=torch.bool, device=music_condition_mask.device
)
condition_mask[:, 0, :music_condition_mask.shape[-1]] = music_condition_mask.bool()
condition_mask[:, 1, music_condition_mask.shape[-1]:] = motion_condition_mask.bool()
```

---

## 7. Codebook Pattern Provider (Delayed Pattern)

**File:** `unimumo/audio/audiocraft_/modules/codebooks_patterns.py:303-353`

```python
class DelayedPatternProvider(CodebooksPatternProvider):
    """Codebooks are delayed in the sequence.

    Example with timesteps=4 and n_q=3, delays=None:
        Input:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]

        Output (S = special token):
        [[S, 1, 2, 3, 4],
         [S, S, 1, 2, 3],
         [S, S, S, 1, 2]]
    """
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))  # [0, 1, 2, 3] for 4 codebooks
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

**Pattern Usage in LMModel:**

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:242-249`

```python
# Map codes [B, K, T] into pattern sequence [B, K, S]
music_pattern = self.pattern_provider.get_pattern(T_music)
motion_pattern = self.pattern_provider.get_pattern(T_motion)
music_sequence_codes, _, _ = music_pattern.build_pattern_sequence(
    music_codes, self.special_token_id, keep_only_valid_steps=True
)
motion_sequence_codes, _, _ = motion_pattern.build_pattern_sequence(
    motion_codes, self.special_token_id, keep_only_valid_steps=True
)
```

---

## 8. Separate FFN for Motion (Modality-Specific Processing)

**File:** `unimumo/audio/audiocraft_/modules/transformer.py:591-594`

```python
# In StreamingTransformerLayer.__init__:
# Add separate MLP for motion modality
self.linear1_motion = nn.Linear(d_model, dim_feedforward, bias=bias_ff, **factory_kwargs)
self.linear2_motion = nn.Linear(dim_feedforward, d_model, bias=bias_ff, **factory_kwargs)
self.norm1_motion = create_norm_fn(norm, d_model, **factory_kwargs)
self.norm2_motion = create_norm_fn(norm, d_model, **factory_kwargs)
```

**Forward with Separate FFN:**

**File:** `unimumo/audio/audiocraft_/modules/transformer.py:640-644`

```python
# In forward, split and process music/motion separately through FFN
x_music = x[:, :S//2]
x_motion = x[:, S//2:]
x_music = x_music + self.layer_scale_2(self._ff_block(self.norm2(x_music)))
x_motion = x_motion + self.layer_scale_2(self._ff_block_motion(self.norm2_motion(x_motion)))
x = torch.cat((x_music, x_motion), dim=1)
```

---

## 9. Weight Initialization from Pretrained MusicGen

**File:** `unimumo/audio/audiocraft_/models/loaders.py:106-127`

```python
def load_mm_lm_model(...):
    # ...

    # Initialize motion embeddings from music embeddings
    for k in my_model_dict.keys():
        if k.startswith('motion_emb.'):
            music_emb_key = k.replace('motion_', '')
            new_dict[k] = pretrained_dict[music_emb_key].clone()
            print(f'Init {k} with {music_emb_key}')

    # Initialize motion MLP from music MLP
    for k in my_model_dict.keys():
        if 'linear1_motion' in k or 'linear2_motion' in k or 'norm1_motion' in k or 'norm2_motion' in k:
            original_key_name = k.replace('_motion', '')
            new_dict[k] = pretrained_dict[original_key_name].clone()
            print(f'Init {k} with {original_key_name}')
```

---

## 10. ScaledEmbedding Class

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:73-82`

```python
class ScaledEmbedding(nn.Embedding):
    """Embedding with optional separate learning rate."""
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group
```

---

## 11. LMOutput and Loss Computation

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:85-88`

```python
@dataclass
class LMOutput:
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor    # [B, K, T]
```

**Cross-Entropy Loss per Codebook:**

**File:** `unimumo/models/transformer_model.py:343-362`

```python
def compute_cross_entropy(
    self, logits: torch.Tensor, targets: torch.LongTensor, mask: torch.Tensor
) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
    B, K, T = targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook: tp.List[torch.Tensor] = []

    for k in range(K):
        logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))
        targets_k = targets[:, k, ...].contiguous().view(-1)
        mask_k = mask[:, k, ...].contiguous().view(-1)
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())

    # Average across codebooks
    ce = ce / K
    return ce, ce_per_codebook
```

---

## 12. Generation Loop

**File:** `unimumo/audio/audiocraft_/models/mm_lm.py:499-538`

```python
# Autoregressive generation
for offset in tqdm(range(1, gen_sequence_len), desc=f"Generating..."):
    music_curr_sequence = music_gen_sequence[..., 0:offset]
    motion_curr_sequence = motion_gen_sequence[..., 0:offset]

    # Sample next token
    music_next_token, motion_next_token = self._sample_next_token(
        music_curr_sequence, motion_curr_sequence, cfg_conditions, mode,
        use_sampling, temp, top_k, top_p, cfg_coef=cfg_coef
    )

    # Apply valid mask (for delayed pattern)
    music_valid_mask = music_mask[..., offset:offset+1].expand(B, -1, -1)
    music_next_token[~music_valid_mask] = self.special_token_id

    # Update sequence
    music_gen_sequence[..., offset:offset+1] = torch.where(
        music_gen_sequence[..., offset:offset+1] == unknown_token,
        music_next_token, music_gen_sequence[..., offset:offset+1]
    )
```

---

## 13. Configuration Reference

**File:** `configs/train_music_motion.yaml`

```yaml
model:
  target: unimumo.models.transformer_model.MusicMotionTransformer
  params:
    name: 'facebook/musicgen-small'
    motion_weight: 0.15  # Weight for motion loss vs music loss
    length_single_modal: 500
    feature_frame_rate: 50  # 50 Hz for both music and motion codes

    generation_params:
      use_sampling: True
      temp: 1.
      top_k: 250
      top_p: 0.0
      cfg_coef: 4.0  # CFG coefficient at inference
      duration: 10
```

---

## 14. Summary: What MotionGPT Needs to Implement

| Component | UniMuMo Implementation | MotionGPT Adaptation |
|-----------|----------------------|---------------------|
| **Per-codebook Embeddings** | `nn.ModuleList` of 4 `ScaledEmbedding(1025, 1024)` | `nn.ModuleList` of 6 `ScaledEmbedding(513, hidden_dim)` |
| **Per-codebook Output Heads** | `nn.ModuleList` of 4 `nn.Linear(1024, 1024)` | `nn.ModuleList` of 6 `nn.Linear(hidden_dim, 512)` |
| **Embedding Aggregation** | Sum across codebooks | Same |
| **T5 Conditioning** | T5EncoderModel + projection layer | Same |
| **Cross-Attention** | StreamingMultiheadAttention | Standard cross-attention |
| **CFG Training** | 10% condition dropout | Same |
| **CFG Inference** | `logits = uncond + cfg * (cond - uncond)` | Same |
| **Pattern Provider** | DelayedPatternProvider with delays [0,1,2,3] | DelayedPatternProvider with delays [0,1,2,3,4,5] |
| **Attention Mask** | Custom causal mask | Same logic |

---

## 15. RVQVAE Settings Comparison

| Parameter            | UniMuMo     | MotionGPT (Current) | Action Needed                                |
|---------------------|-------------|-----------------------|-----------------------------------------------|
| Num Quantizers      | 4           | 6                     | Transformer must handle **6 codebooks**       |
| Codebook Size       | 1024        | 512                   | Use **512 vocabulary**                        |
| Embedding Dim       | 128         | 512                   | Keep 512                                      |
| Temporal Compression| ~1.2x (60→50Hz) | 4x               | Already set                                   |
| Input Features      | 263 (HumanML3D) | 133 (How2Sign)  | Already set                                   |

---

This document provides all the critical implementation details from UniMuMo needed to adapt the architecture to MotionGPT. The person implementing this should refer to the specific file paths and code snippets provided above.
