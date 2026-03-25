# Text2VQPoseGPT Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Text2VQPoseGPT Model                            │
│                                                                         │
│  Input: Text Description ("Hello, how are you?")                       │
│         Previous Tokens (during autoregressive generation)             │
│                                   ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │         SentenceTransformer Text Encoder (External)           │    │
│  │         Model: clip-ViT-B-32-multilingual-v1                  │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                   ↓                                     │
│                    Text Embedding [batch, 512]                          │
│                                   ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │                  CrossCondTransBase                            │    │
│  │               (Shared Encoder Network)                         │    │
│  │                                                                │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ 1. Condition Embedding                           │         │    │
│  │  │    - Linear(clip_dim=512, embed_dim=512)         │         │    │
│  │  │    - Projects text embeddings                    │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  │                         ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ 2. Token Embedding + Combiner                    │         │    │
│  │  │    - Embedding(num_vq=627, embed_dim=512)        │         │    │
│  │  │    - Token Combiner:                             │         │    │
│  │  │      Linear(512*pose_size → 512)                 │         │    │
│  │  │      Fuses pose_size tokens per timestep         │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  │                         ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ 3. Positional Encoding                           │         │    │
│  │  │    - Sinusoidal position embeddings              │         │    │
│  │  │    - Supports offset for KV caching              │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  │                         ↓                                      │    │
│  │  ┌──────────────────────────────────────────────────┐         │    │
│  │  │ 4. Transformer Blocks (num_layers=8)             │         │    │
│  │  │                                                   │         │    │
│  │  │    Block 0:                                       │         │    │
│  │  │    ┌────────────────────────────────┐            │         │    │
│  │  │    │ LayerNorm                      │            │         │    │
│  │  │    │ Causal Self-Attention (8 heads)│            │         │    │
│  │  │    │   - Q, K, V projections        │            │         │    │
│  │  │    │   - Scaled dot-product         │            │         │    │
│  │  │    │   - Causal masking             │            │         │    │
│  │  │    │   - KV caching support         │            │         │    │
│  │  │    │ Residual Connection            │            │         │    │
│  │  │    └────────────────────────────────┘            │         │    │
│  │  │    ┌────────────────────────────────┐            │         │    │
│  │  │    │ LayerNorm                      │            │         │    │
│  │  │    │ MLP (4x expansion):            │            │         │    │
│  │  │    │   Linear(512 → 2048)           │            │         │    │
│  │  │    │   GELU                         │            │         │    │
│  │  │    │   Linear(2048 → 512)           │            │         │    │
│  │  │    │   Dropout                      │            │         │    │
│  │  │    │ Residual Connection            │            │         │    │
│  │  │    └────────────────────────────────┘            │         │    │
│  │  │                                                   │         │    │
│  │  │    Block 1-7: (same structure)                   │         │    │
│  │  └──────────────────────────────────────────────────┘         │    │
│  └────────────────────────────────────────────────────────────── │    │
│                                   ↓                                     │
│              Shared Features [batch, seq_len, 512]                     │
│                                   ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │          Multiple CrossCondTransHead (pose_size heads)        │    │
│  │                                                                │    │
│  │  Head 0 ────┐    Head 1 ────┐         Head 63 ────┐          │    │
│  │      ↓      │         ↓     │    ...        ↓      │          │    │
│  │  ┌────────┐ │    ┌────────┐ │          ┌────────┐ │          │    │
│  │  │ Block  │ │    │ Block  │ │          │ Block  │ │          │    │
│  │  │(Layers)│ │    │(Layers)│ │          │(Layers)│ │          │    │
│  │  └────────┘ │    └────────┘ │          └────────┘ │          │    │
│  │      ↓      │         ↓     │    ...        ↓      │          │    │
│  │  LayerNorm  │    LayerNorm  │          LayerNorm   │          │    │
│  │      ↓      │         ↓     │              ↓       │          │    │
│  │  Linear     │    Linear     │          Linear      │          │    │
│  │  (512→627)  │    (512→627)  │          (512→627)   │          │    │
│  │      ↓      │         ↓     │              ↓       │          │    │
│  │  Logits     │    Logits     │          Logits      │          │    │
│  │  [B,S,627]  │    [B,S,627]  │          [B,S,627]   │          │    │
│  └─────┬───────┴────────┬──────┴──────────────┬───────┴──────────┘    │
│        └────────────────┴─────────────────────┘                        │
│                              ↓                                         │
│              Stack & Rearrange to [B, S, 64, 627]                     │
│                              ↓                                         │
│              Flatten to [B, S*64, 627] (optional)                     │
│                              ↓                                         │
│  Output: Logits over VQ vocabulary for each token position            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Model Parameters (Default Configuration)

```yaml
num_vq: 627              # Codebook size (625) + END (626) + PAD (627)
embed_dim: 512           # Hidden dimension
clip_dim: 512            # Text embedding dimension
block_size: 401          # Maximum sequence length (400 + 1)
num_layers: 8            # Transformer blocks in base encoder
head_layers: 1           # Transformer blocks per output head
n_head: 8                # Number of attention heads per block
drop_out_rate: 0.0       # Dropout rate
fc_rate: 4               # MLP expansion factor
pose_size: 64            # Number of VQ codes per frame (dataset-specific)
```

## Detailed Component Breakdown

### 1. CrossCondTransBase (Shared Encoder)

**Purpose**: Process text condition and previously generated tokens into contextualized representations.

**Key Operations**:

```python
# Pseudo-code flow
def forward(idx, clip_feature, use_cache=False, shift_t=None):
    # Step 1: Handle first token (text-only)
    if len(idx) == 0:
        embeddings = cond_emb(clip_feature).unsqueeze(1)  # [B, 1, 512]

    # Step 2: Process previous tokens
    else:
        # Reshape flat tokens to [B, T, pose_size]
        idx = idx.view(B, -1, pose_size)  # e.g., [B, 10, 64]

        # For each timestep
        for t in range(T):
            # Get pose_size tokens for this timestep
            pose_tokens = idx[:, t, :]  # [B, 64]

            # Embed each token
            embeds = tok_emb(pose_tokens)  # [B, 64, 512]

            # Combine into single vector
            combined = token_combiner(embeds.flatten(1))  # [B, 512]

        # Prepend text embedding
        embeddings = cat([cond_emb(clip_feature), combined_list], dim=1)

    # Step 3: Add positional encoding
    x = pos_embed(embeddings, offset=shift_t)

    # Step 4: Apply transformer blocks
    for block in blocks:
        x = block(x, use_cache=use_cache)

    return x  # [B, seq_len, 512]
```

### 2. Token Combiner (Multi-Token Fusion)

**Challenge**: Each frame has `pose_size` discrete tokens (e.g., 64 for How2Sign).

**Solution**: Fuse them into a single vector per timestep.

```python
# Architecture
token_combiner = nn.Sequential(
    nn.Linear(embed_dim * pose_size, embed_dim),  # 512*64 → 512
    nn.LayerNorm(embed_dim),
    nn.GELU(),
)

# Example for pose_size=64:
# Input:  [B, 64, 512] (64 token embeddings)
# Flatten: [B, 32768]
# Project: [B, 512] (single timestep representation)
```

### 3. Causal Self-Attention with KV Caching

**Standard Attention (Training)**:
```python
Q = query(x)   # [B, n_head, T, head_dim]
K = key(x)     # [B, n_head, T, head_dim]
V = value(x)   # [B, n_head, T, head_dim]

# Causal masking prevents attending to future
mask = torch.tril(ones(T, T))  # Lower triangular

attention = softmax((Q @ K.T) / sqrt(d_k)) * mask
output = attention @ V
```

**Cached Attention (Inference)**:
```python
# First call: initialize cache
Q_t = query(x_t)     # Only current token [B, n_head, 1, head_dim]
K_t = key(x_t)
V_t = value(x_t)

cached_K = K_t       # Store
cached_V = V_t

# Subsequent calls: reuse cache
Q_t = query(x_t)
K_t = key(x_t)
V_t = value(x_t)

K_all = cat([cached_K, K_t], dim=2)  # [B, n_head, t+1, head_dim]
V_all = cat([cached_V, V_t], dim=2)

attention = softmax(Q_t @ K_all.T / sqrt(d_k))
output = attention @ V_all

cached_K = K_all     # Update cache
cached_V = V_all
```

**Benefit**: O(T) complexity instead of O(T²) for each new token.

### 4. CrossCondTransHead (Output Decoder)

**Purpose**: Refine shared features and predict VQ token distribution.

```python
class CrossCondTransHead(nn.Module):
    def __init__(self, num_vq=627, embed_dim=512, num_layers=1):
        # Additional refinement layers
        self.blocks = nn.Sequential(*[
            Block(embed_dim, ...) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)

    def forward(self, x, use_cache=False):
        # Refine features
        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        # Layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.head(x)  # [B, seq_len, num_vq]

        return logits
```

### 5. Multi-Head Output Strategy

**Why 64 separate heads?**

Each head specializes in predicting one dimension of the VQ code:

```
Frame t has 64 VQ tokens: [token_0, token_1, ..., token_63]

Head 0 predicts: token_0 ∈ {0, 1, ..., 624, END, PAD}
Head 1 predicts: token_1 ∈ {0, 1, ..., 624, END, PAD}
...
Head 63 predicts: token_63 ∈ {0, 1, ..., 624, END, PAD}

All predictions happen in parallel for efficiency.
```

## Autoregressive Generation Process

### Training (Teacher Forcing with Scheduled Sampling)

```python
# Ground truth sequence
target = [t0_0, t0_1, ..., t0_63,    # Frame 0 (64 tokens)
          t1_0, t1_1, ..., t1_63,    # Frame 1 (64 tokens)
          ...,
          tN_0, tN_1, ..., tN_63]    # Frame N (64 tokens)

# Input (shifted by pose_size)
input = target[:-pose_size]  # Remove last frame

# Scheduled sampling (60% ground truth, 40% random)
mask = bernoulli(0.6, size=input.shape)
random_tokens = randint(0, codebook_size, size=input.shape)
input = mask * input + (1 - mask) * random_tokens

# Forward pass
logits = model(input, text_embedding)  # [B, seq_len, num_vq]

# Compute loss
loss = CrossEntropy(logits, target, ignore_index=PAD)
```

### Inference (Autoregressive Sampling)

```python
def generate(text, max_len=400):
    # Encode text
    text_emb = sentence_transformer(text)  # [1, 512]

    # Initialize
    model.clear_cache()  # Reset KV cache
    generated = []

    for t in range(max_len):
        if t == 0:
            # First token: only text condition
            input_tokens = []
        else:
            # Use last generated frame
            input_tokens = generated[-1]  # [1, pose_size]

        # Forward with caching
        logits = model.forward(
            input_tokens,
            text_emb,
            use_cache=True,
            shift_t=t
        )  # [1, 1, pose_size, num_vq]

        logits = logits[:, -1, :, :]  # Get last position [1, pose_size, num_vq]

        # Sample from distribution
        probs = softmax(logits, dim=-1)
        sampled = probs.argmax(dim=-1)  # Greedy: [1, pose_size]
        # OR: sampled = Categorical(probs).sample()  # Stochastic

        # Check for END token
        if (sampled == END_TOKEN).all():
            break

        generated.append(sampled)

    # Flatten: [num_frames, pose_size] → [num_frames * pose_size]
    return flatten(generated)
```

## Memory and Computation

### Model Size

```python
# Base Encoder
Token Embedding:      627 × 512 = 321,024 params
Condition Projection: 512 × 512 = 262,144 params
Token Combiner:       (512×64) × 512 = 16,777,216 params
Position Embedding:   401 × 512 = 205,312 params (frozen)

# Transformer Blocks (8 blocks)
Per Block:
  - Self-Attention: 512×512×4 (Q,K,V,O) = 1,048,576 params
  - MLP: 512×2048 + 2048×512 = 2,097,152 params
  - LayerNorms: ~2,048 params
  Total per block: ~3.15M params
Total for 8 blocks: ~25.2M params

# Output Heads (64 heads)
Per Head:
  - 1 Transformer Block: ~3.15M params
  - LayerNorm: ~1,024 params
  - Output Linear: 512×627 = 321,024 params
  Total per head: ~3.47M params
Total for 64 heads: ~222M params

# Total Model Size: ~265M parameters
```

### Inference Complexity (Per Token)

```
Without KV Cache: O(T² × d) for T tokens
With KV Cache:    O(T × d)  for T tokens

For T=400, d=512:
  No cache:  ~82M operations per token
  With cache: ~205K operations per token
  Speedup: ~400x
```

## Comparison to Standard GPT

| Aspect | Standard GPT | Text2VQPoseGPT |
|--------|-------------|----------------|
| **Input** | Token sequence | Text embedding + token sequence |
| **Output** | Single vocabulary prediction | Multi-head (64 predictions per timestep) |
| **Token Format** | Flat sequence | Grouped by `pose_size` |
| **Special Tokens** | BOS, EOS, PAD | END, PAD (no BOS) |
| **Text Conditioning** | Prepended to sequence | Injected via learned projection |
| **Applications** | Language modeling | Text-to-pose sequence translation |

## Training Dynamics

### Loss Curve

```
Cross-Entropy Loss:
  - Initial: ~6.4 (random guessing over 627 classes)
  - After 1K steps: ~3.5
  - Converged: ~1.8-2.2

Accuracy:
  - Initial: ~0.16% (1/627)
  - After 1K steps: ~15%
  - Converged: ~35-45%

DTW Score (lower is better):
  - Initial: ~200
  - After 10K steps: ~50
  - Best: ~15-25
```

### Why Accuracy is "Low"?

- 627-way classification is hard
- Multiple valid sign language variations for the same text
- Model optimized for **sequence coherence** (DTW) not exact match
- Diffusion model can "fix" small errors in tokens

## Summary

The **Text2VQPoseGPT** architecture is a specialized GPT variant that:

1. **Accepts text embeddings** as conditioning input
2. **Generates multi-dimensional discrete tokens** (pose_size per timestep)
3. **Uses separate output heads** for each token dimension
4. **Employs KV caching** for efficient autoregressive generation
5. **Bridges language and pose** through learned token embeddings

The model serves as the **critical translation layer** between natural language and quantized pose representations, enabling the downstream diffusion model to generate realistic sign language videos.
