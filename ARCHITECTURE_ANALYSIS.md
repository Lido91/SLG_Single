# MotionGPT: Architecture & Training Pipeline Analysis

**Analysis Date**: 2025-10-17
**Repository**: MotionGPT - Human Motion as a Foreign Language

---

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [3-Stage Training Pipeline](#3-stage-training-pipeline)
4. [ASCII Pipeline Diagrams](#ascii-pipeline-diagrams)
5. [Instruction Templates](#instruction-templates)
6. [Technical Specifications](#technical-specifications)
7. [File Structure](#file-structure)

---

## Overview

MotionGPT is a **unified motion-language model** that treats human motion as a "foreign language." It uses:
- **Vector Quantization (VQ-VAE)** to tokenize continuous motion into discrete codes
- **T5 Language Model** extended with motion tokens to process both modalities
- **3-Stage Training** to progressively learn motion vocabulary, motion-text alignment, and multi-task capabilities

**Key Innovation**: Motion sequences are converted to discrete tokens (like words), enabling language models to understand and generate both text and motion in a unified framework.

---

## Model Architecture

### Core Components

#### 1. Motion Tokenizer (VQ-VAE)
**File**: [mGPT/archs/mgpt_vq.py](mGPT/archs/mgpt_vq.py)

```
Input Motion: [Batch, Time, 263 features]
           ↓
    ┌──────────────┐
    │   Encoder    │  Conv1D + ResNet1D blocks
    │  Downsample  │  Temporal downsampling by 4x
    └──────┬───────┘  (20fps → 5fps)
           ↓
    [Batch, 512, T/4]
           ↓
    ┌──────────────┐
    │  Quantizer   │  EMA-Reset algorithm
    │ Codebook:512 │  Maps to discrete codes
    └──────┬───────┘
           ↓
    Discrete Codes: [Batch, T/4]
           ↓
    ┌──────────────┐
    │   Decoder    │  ResNet1D + Upsample
    │  Upsample 4x │  Reconstruct motion
    └──────┬───────┘
           ↓
Reconstructed: [Batch, Time, 263]
```

**Parameters**:
- Codebook size: 512 tokens
- Code dimension: 512
- Downsampling factor: 4
- Width: 512, Depth: 3
- Quantizer: EMA-Reset (μ=0.99)

#### 2. Language Model (T5-Based)
**File**: [mGPT/archs/mgpt_lm.py](mGPT/archs/mgpt_lm.py)

```
Original T5 Vocabulary (32k tokens)
           +
Motion Tokens (515 new tokens)
           ↓
    Extended Vocabulary
    ┌─────────────────────────────┐
    │ <motion_id_0> ... <motion_id_511>  │  Codebook tokens
    │ <motion_id_512>                     │  Motion start
    │ <motion_id_513>                     │  Motion end
    │ <motion_id_514>                     │  Mask token
    └─────────────────────────────┘
           ↓
    T5 Encoder-Decoder (770M params)
    - 12 encoder layers
    - 12 decoder layers
    - d_model = 768
    - Unified text + motion embeddings
```

**Special Tokens**:
- `<motion_id_0>` to `<motion_id_511>`: Motion vocabulary (512 codes)
- `<motion_id_512>`: Motion sequence start marker
- `<motion_id_513>`: Motion sequence end marker
- `<motion_id_514>`: Masked motion token (for in-between task)

#### 3. Unified MotionGPT Model
**File**: [mGPT/models/mgpt.py](mGPT/models/mgpt.py)

```python
class MotionGPT:
    def __init__(self):
        self.vae = VQVae(...)           # Motion tokenizer
        self.lm = MLM(...)              # Language model
        self._losses = GPTLosses(...)   # Task-specific losses

    # Stage-specific forward methods
    - train_vae_forward()      # Stage 1: VQ-VAE training
    - train_lm_forward()       # Stage 2 & 3: LM training
    - val_t2m_forward()        # Text-to-Motion inference
    - val_m2t_forward()        # Motion-to-Text inference
    - val_m2m_forward()        # Motion-to-Motion (pred/inbetween)
```

---

## 3-Stage Training Pipeline

### Stage 1: Motion Tokenizer Training
**Config**: [configs/config_h3d_stage1.yaml](configs/config_h3d_stage1.yaml)

**Objective**: Learn discrete motion vocabulary

```yaml
TRAIN:
  STAGE: vae
  BATCH_SIZE: 256
  OPTIM:
    lr: 2e-4

LOSS:
  LAMBDA_FEATURE: 1.0      # Reconstruction loss
  LAMBDA_VELOCITY: 0.5     # Velocity loss
  LAMBDA_COMMIT: 0.02      # Commitment loss
  ABLATION:
    RECONS_LOSS: 'l1_smooth'
```

**Training Process**:
1. Input raw motion sequences (263-dim features)
2. VQ-VAE encodes → quantizes → decodes
3. Compute reconstruction + velocity + commitment losses
4. Update encoder, decoder, and codebook

**Output**:
- Trained VQ-VAE checkpoint
- Motion codebook with 512 discrete tokens

---

### Stage 2: Motion-Language Pre-training
**Config**: [configs/config_h3d_stage2.yaml](configs/config_h3d_stage2.yaml)

**Objective**: Learn motion-text semantic alignment

```yaml
TRAIN:
  STAGE: lm_pretrain
  BATCH_SIZE: 16
  PRETRAINED_VAE: <stage1_checkpoint>  # Frozen VQ-VAE
  OPTIM:
    lr: 2e-4

DATASET:
  CODE_PATH: TOKENS  # Pre-extracted motion tokens
```

**Training Process**:
1. Pre-extract motion tokens using frozen VQ-VAE (for efficiency)
2. Sample text-motion pairs from HumanML3D dataset
3. Randomly choose training strategy:
   - **Supervised** (75%): Text → Motion or Motion → Text
   - **Self-supervised** (25%): Text → Text or Motion → Motion (with span masking)
4. Train T5 with teacher forcing

**Training Strategies**:
```python
# From mgpt_lm.py:95-129
condition = random.choice(['supervised', 'supervised', 'supervised'])

if condition == 'text':
    # Self-supervised: Masked language modeling on text only
    inputs = texts
    outputs = texts (with span masking)

elif condition == 'motion':
    # Self-supervised: Masked motion modeling
    inputs = motion_strings
    outputs = motion_strings (with span masking)

else:  # 'supervised'
    # Supervised: Cross-modal generation
    inputs, outputs = template_fulfill(tasks, lengths, motion_strings, texts)
    # Examples:
    # - "Generate motion: walking" → <motion_tokens>
    # - "Describe motion: <motion_tokens>" → "walking forward"
```

**Output**:
- Pre-trained MotionGPT with motion-language alignment

---

### Stage 3: Instruction Tuning
**Config**: [configs/config_h3d_stage3.yaml](configs/config_h3d_stage3.yaml)

**Objective**: Multi-task instruction following

```yaml
TRAIN:
  STAGE: lm_instruct
  BATCH_SIZE: 16
  PRETRAINED: <stage2_checkpoint>
  OPTIM:
    lr: 1e-4  # Lower LR for fine-tuning

DATASET:
  CODE_PATH: TOKENS
```

**Training Tasks**:
1. **Text-to-Motion (T2M)**: Generate motion from text
2. **Motion-to-Text (M2T)**: Generate caption from motion
3. **Motion Prediction**: Predict future motion given initial frames
4. **Motion In-between**: Fill masked motion gaps

**Training Process**:
1. Load pre-trained MotionGPT from Stage 2
2. Mix all task types in each batch
3. Randomly sample instruction templates for diversity
4. Fine-tune entire model with task-specific losses

**Output**:
- Final MotionGPT model capable of multi-task motion-language generation

---

## ASCII Pipeline Diagrams

### Complete Training Pipeline

```
╔═════════════════════════════════════════════════════════════════════════╗
║                        STAGE 1: MOTION TOKENIZER                         ║
╚═════════════════════════════════════════════════════════════════════════╝

    Raw Motion Data (HumanML3D)
    [N, T, 263] features
           │
           ▼
    ┌──────────────┐
    │ Data Loader  │ ← Normalize with mean/std
    └──────┬───────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │      VQ-VAE ENCODER             │
    │  - Conv1D layers                │
    │  - ResNet1D blocks              │
    │  - Downsample 4x (stride=2)     │
    └──────────┬──────────────────────┘
               │ [N, 512, T/4]
               ▼
    ┌─────────────────────────────────┐
    │    VECTOR QUANTIZER             │
    │  - Codebook: 512 tokens         │
    │  - Code dim: 512                │
    │  - EMA-Reset algorithm          │
    └──────────┬──────────────────────┘
               │ Discrete codes [N, T/4]
               ▼
    ┌─────────────────────────────────┐
    │      VQ-VAE DECODER             │
    │  - Upsample 4x                  │
    │  - ResNet1D blocks              │
    │  - Conv1D layers                │
    └──────────┬──────────────────────┘
               │
               ▼
    Reconstructed Motion [N, T, 263]
               │
               ▼
    ┌─────────────────────────────────┐
    │         LOSS FUNCTIONS          │
    │  - L1 Smooth (Reconstruction)   │
    │  - Commitment Loss (λ=0.02)     │
    │  - Velocity Loss (λ=0.5)        │
    └──────────┬──────────────────────┘
               │
               ▼
          Trained VQ-VAE ✓


╔═════════════════════════════════════════════════════════════════════════╗
║                  STAGE 2: MOTION-LANGUAGE PRETRAINING                    ║
╚═════════════════════════════════════════════════════════════════════════╝

    Text + Motion Pairs
    "A person walks" + <motion_data>
           │
           ├─────────────────┬──────────────────┐
           │                 │                  │
           ▼                 ▼                  ▼
    ┌──────────┐   ┌─────────────────┐   ┌─────────────┐
    │   Text   │   │ Frozen VQ-VAE   │   │  Template   │
    │ "walk"   │   │     Encoder     │   │ Instructions│
    └─────┬────┘   └────────┬────────┘   └──────┬──────┘
          │                 │                    │
          │        Motion → Tokens               │
          │        [<m_512><m_23>...]           │
          │                 │                    │
          └─────────────────┴────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │   T5 TOKENIZER (Extended)    │
              │ Vocab: 32k + 515 motion IDs  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      T5 ENCODER-DECODER      │
              │  - 12 encoder layers         │
              │  - 12 decoder layers         │
              │  - d_model = 768             │
              │  - Unified embeddings        │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    TRAINING STRATEGIES       │
              │ 1. Supervised: Text→Motion   │
              │ 2. Self-sup: Text/Motion→Self│
              │ 3. Random span masking       │
              │ 4. Template fulfillment      │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       CROSS ENTROPY LOSS     │
              └──────────────┬───────────────┘
                             │
                             ▼
                  Pretrained MotionGPT ✓


╔═════════════════════════════════════════════════════════════════════════╗
║                    STAGE 3: INSTRUCTION TUNING                           ║
╚═════════════════════════════════════════════════════════════════════════╝

    Task Templates (JSON)
    {
      "Text-to-Motion": "Generate motion: <Caption>",
      "Motion-to-Text": "Generate text: <Motion>",
      "Prediction": "Predict motion: <Motion_s1>",
      "In-between": "Complete masked: <Motion_Masked>"
    }
           │
           ▼
    ┌─────────────────────────────────┐
    │   MULTI-TASK DATA LOADER        │
    │  - Mix all tasks                │
    │  - Random template sampling     │
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │   PRETRAINED MOTIONGPT          │
    │  (From Stage 2)                 │
    │  - Frozen VQ-VAE                │
    │  - Trainable T5                 │
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │     INSTRUCTION FORWARD         │
    │  Input: "Generate motion: walk" │
    │  Output: <m_512><m_23>...<m_513>│
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │      TASK-SPECIFIC LOSSES       │
    │  - Language modeling loss       │
    │  - Lower LR (1e-4)              │
    └──────────┬──────────────────────┘
               │
               ▼
         Final MotionGPT ✓


╔═════════════════════════════════════════════════════════════════════════╗
║                           INFERENCE PIPELINE                             ║
╚═════════════════════════════════════════════════════════════════════════╝

    User Input: "A person jumps high"
           │
           ▼
    ┌──────────────────────────────────┐
    │  Template Fulfillment            │
    │  → "Generate motion: A person    │
    │     jumps high"                  │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │  T5 Encoder-Decoder              │
    │  - Encode text                   │
    │  - Decode motion tokens          │
    │  → [23, 45, 89, 12, ...]         │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │  VQ-VAE Decoder                  │
    │  - Dequantize codes              │
    │  - Upsample 4x                   │
    │  → [T, 263] features             │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │  Feats2Joints Transform          │
    │  → [T, 22, 3] 3D joints          │
    └──────────┬───────────────────────┘
               │
               ▼
         SMPL Rendering / Visualization
```

---

## Instruction Templates

MotionGPT uses **~575+ diverse instruction templates** during training to enable robust multi-task learning.

### 1. Text-to-Motion (T2M)

**144+ variations** for basic text-to-motion generation:

```
Input Examples:
- "Generate motion: walking forward"
- "Give me a gesture that corresponds to jumping"
- "Create a motion that represents running quickly"
- "Show me a motion that captures the essence of dancing"
- "I need a motion that represents the power of kicking. Can you generate it for me?"
- "Demonstrate a sequence of movements that symbolizes the sentiment of waving"
- "Can you show me that a person is sitting down"

Output:
<motion_id_512><motion_id_23><motion_id_45>...<motion_id_513>
```

**48+ variations** with frame length specification:

```
Input Examples:
- "Give me a motion that lasts for approximately 120 frames. The caption is: walking"
- "Show me a human motion that represents 'running' for 100 frames"
- "I need a motion that lasts for 150 frames and is inspired by the phrase 'dancing'"

Output:
<motion_id_512><motion_id_23>... (120 frames worth of tokens) ...<motion_id_513>
```

### 2. Motion-to-Text (M2T)

**80+ variations** for motion captioning:

```
Input Examples:
- "Describe the motion represented by <Motion_Placeholder> using plain English"
- "What kind of motion is displayed in <Motion_Placeholder>? Describe it in text"
- "Explain the motion shown in <Motion_Placeholder> using natural language"
- "Give me a brief summary of the motion displayed in <Motion_Placeholder>"
- "Can you tell me what is happening in <Motion_Placeholder> using natural language?"
- "Provide a textual description of <Motion_Placeholder>"

Output:
"A person walks forward and then turns around"
```

### 3. Motion Prediction

**1 template** for motion forecasting:

```
Input:
"Predict motion: <Motion_Placeholder_s1>"

Example:
"Predict motion: <motion_id_512><motion_id_23><motion_id_45><motion_id_513>"
(First 20% of the motion sequence)

Output:
<motion_id_512><motion_id_67><motion_id_89>...<motion_id_513>
(Remaining 80% of the sequence)
```

### 4. Motion In-between

**2 variations** for motion interpolation:

```
Input Examples:
- "Complete the masked motion: <Motion_Placeholder_Masked>"
- "Here is a masked motion sequence <Motion_Placeholder_Masked>, complete it"

Example:
Input: <motion_id_512><motion_id_23><motion_id_514><motion_id_514><motion_id_89><motion_id_513>
       (Middle 50% masked with <motion_id_514>)

Output:
<motion_id_512><motion_id_23><motion_id_45><motion_id_67><motion_id_89><motion_id_513>
(Complete unmasked sequence)
```

### 5. Random Motion Generation

**200+ variations** for unconditional generation:

```
Input Examples:
- "Generate random motions"
- "Create movements that are spontaneous and freeform"
- "Produce movements that are free and intuitive"
- "Make the motions unpredictable"
- "Give me motions as you like"
- "Create movements that are unrestricted and free-flowing"

Output:
<motion_id_512><motion_id_X><motion_id_Y>...<motion_id_513> (random motion)
```

### 6. Motion Duration Prediction

**100+ variations** for frame estimation:

```
Input Examples:
- "Predict the frame count required for the motion corresponding to walking forward"
- "How many frames are needed to perform the motion described by jumping?"
- "Estimate the number of frames required for the motion that matches running"

Output:
"150" (predicted frame count)
```

### Special Placeholders

| Placeholder | Replaced With | Example |
|-------------|---------------|---------|
| `<Caption_Placeholder>` | Text description | "a person walks forward" |
| `<Motion_Placeholder>` | Full motion tokens | `<motion_id_512><motion_id_23>...<motion_id_513>` |
| `<Motion_Placeholder_s1>` | First 20% of motion | For prediction input |
| `<Motion_Placeholder_s2>` | Last 80% of motion | For prediction output |
| `<Motion_Placeholder_Masked>` | Masked motion | Middle 50% replaced with `<motion_id_514>` |
| `<Frame_Placeholder>` | Frame count | "120" |
| `<Second_Placeholder>` | Duration in seconds | "6.0" |

### Template Sampling Strategy

During training, templates are randomly sampled to increase diversity:

```python
# From mgpt_lm.py:465-484
def template_fulfill(tasks, lengths, motion_strings, texts):
    inputs = []
    outputs = []
    for i in range(len(lengths)):
        # Randomly pick a template variant
        input_template = random.choice(tasks[i]['input'])
        output_template = random.choice(tasks[i]['output'])

        # Fill in placeholders
        inputs.append(
            placeholder_fulfill(input_template, length, motion_strings[i], texts[i])
        )
        outputs.append(
            placeholder_fulfill(output_template, length, motion_strings[i], texts[i])
        )
    return inputs, outputs
```

### Instruction Template Statistics

| Task Type | # Templates | Purpose |
|-----------|-------------|---------|
| Text-to-Motion (basic) | 144 | Generate motion from text |
| Text-to-Motion (with length) | 48 | Generate motion with duration |
| Motion-to-Text | 80+ | Caption motion sequences |
| Motion Prediction | 1 | Forecast future motion |
| Motion In-between | 2 | Fill masked motion gaps |
| Random Motion | 200+ | Unconditional generation |
| Duration Prediction | 100+ | Estimate motion length |
| **TOTAL** | **~575+** | **Multi-task learning** |

---

## Technical Specifications

### Model Parameters

| Component | Size | Details |
|-----------|------|---------|
| VQ-VAE Encoder | ~5M | Conv1D + ResNet1D, width=512, depth=3 |
| VQ-VAE Decoder | ~5M | ResNet1D + Upsample, width=512, depth=3 |
| Codebook | 512 × 512 | 512 tokens, 512-dim embeddings |
| T5 Language Model | 770M | 12 encoder + 12 decoder layers |
| **Total Parameters** | **~780M** | |

### Dataset: HumanML3D

| Property | Value |
|----------|-------|
| Total Sequences | 14,616 motions |
| Feature Dimension | 263 (positions + velocities + foot contacts) |
| Frame Rate | 20 fps → 5 fps (after downsampling) |
| Min Length | 40 frames (2 seconds) |
| Max Length | 196 frames (9.8 seconds) |
| Unit Length | 4 frames |
| Train/Val/Test Split | ~80% / ~10% / ~10% |
| Joint Count | 22 joints |
| Text Descriptions | Multiple captions per motion |

### Training Hyperparameters

#### Stage 1: VQ-VAE
```yaml
Batch Size: 256
Learning Rate: 2e-4
Optimizer: AdamW (β1=0.9, β2=0.99)
Weight Decay: 0.0
Loss Weights:
  - Reconstruction (L1 Smooth): 1.0
  - Velocity: 0.5
  - Commitment: 0.02
Max Epochs: 999,999 (early stopping based on validation)
```

#### Stage 2: Pre-training
```yaml
Batch Size: 16
Learning Rate: 2e-4
Optimizer: AdamW (β1=0.9, β2=0.99)
Weight Decay: 0.0
Frozen: VQ-VAE (all parameters)
Trainable: T5 + Extended embeddings
Max Length: 256 tokens
```

#### Stage 3: Instruction Tuning
```yaml
Batch Size: 16
Learning Rate: 1e-4 (lower for fine-tuning)
Optimizer: AdamW (β1=0.9, β2=0.99)
Weight Decay: 0.0
Frozen: VQ-VAE
Trainable: T5
Max Length: 256 tokens
```

### Hardware Requirements

```yaml
Recommended Setup:
  - GPU: NVIDIA A100 (40GB) or V100 (32GB)
  - GPU Count: 1-8 (DDP supported)
  - RAM: 64GB+
  - Storage: 100GB+ (for dataset + checkpoints)

Training Time (approximate):
  - Stage 1 (VQ-VAE): ~2-3 days on 1×A100
  - Stage 2 (Pretrain): ~3-5 days on 4×A100
  - Stage 3 (Instruct): ~1-2 days on 4×A100
```

### Inference Speed

| Task | Speed | Hardware |
|------|-------|----------|
| Text-to-Motion | ~0.5s per motion | 1×A100 |
| Motion-to-Text | ~0.3s per caption | 1×A100 |
| Batch T2M (32) | ~5s per batch | 1×A100 |

---

## File Structure

### Core Files

```
MotionGPT/
│
├── train.py                          # Main training script
├── test.py                           # Evaluation script
├── demo.py                           # Batch inference demo
├── app.py                            # Web UI demo
│
├── configs/                          # Configuration files
│   ├── config_h3d_stage1.yaml       # Stage 1: VQ-VAE training
│   ├── config_h3d_stage2.yaml       # Stage 2: Pre-training
│   ├── config_h3d_stage3.yaml       # Stage 3: Instruction tuning
│   ├── default.yaml                  # Default parameters
│   └── assets.yaml                   # Asset paths
│
├── mGPT/                             # Main package
│   ├── models/
│   │   ├── mgpt.py                   # ⭐ Main MotionGPT model (3-stage routing)
│   │   ├── build_model.py            # Model instantiation
│   │   └── base.py                   # Base model class
│   │
│   ├── archs/
│   │   ├── mgpt_vq.py                # ⭐ VQ-VAE architecture
│   │   ├── mgpt_lm.py                # ⭐ Language model wrapper (T5/GPT2)
│   │   └── tools/
│   │       ├── quantize_cnn.py       # Quantization algorithms
│   │       └── resnet.py             # ResNet1D blocks
│   │
│   ├── data/
│   │   ├── HumanML3D.py              # ⭐ HumanML3D dataset loader
│   │   ├── build_data.py             # Data module builder
│   │   ├── humanml/                  # HumanML3D utilities
│   │   └── transforms/               # Data transformations
│   │
│   ├── losses/
│   │   └── mgpt.py                   # Loss functions
│   │
│   ├── metrics/                      # Evaluation metrics
│   │   ├── tm2t.py                   # Text-Motion metrics (FID, etc.)
│   │   └── m2t.py                    # Motion-Text metrics
│   │
│   ├── config.py                     # Config parsing
│   ├── callback.py                   # Training callbacks
│   └── utils/                        # Utilities
│       ├── logger.py
│       └── load_checkpoint.py
│
├── prepare/                          # Setup scripts
│   ├── instructions/
│   │   ├── template_instructions.json  # ⭐ Instruction templates (Stage 3)
│   │   └── template_pretrain.json      # ⭐ Pre-training templates (Stage 2)
│   ├── download_smpl_model.sh
│   ├── download_pretrained_models.sh
│   └── prepare_t5.sh
│
└── checkpoints/                      # Model checkpoints
    └── MotionGPT-base/
        └── motiongpt_s3_h3d.tar     # Final trained model
```

### Key File Descriptions

| File | Purpose | Key Content |
|------|---------|-------------|
| [train.py](train.py) | Main training entry | Orchestrates 3-stage training pipeline |
| [mGPT/models/mgpt.py](mGPT/models/mgpt.py) | Unified model | Routes between VAE/LM training, handles all tasks |
| [mGPT/archs/mgpt_vq.py](mGPT/archs/mgpt_vq.py) | Motion tokenizer | VQ-VAE encoder/decoder/quantizer |
| [mGPT/archs/mgpt_lm.py](mGPT/archs/mgpt_lm.py) | Language model | T5 wrapper with motion tokens |
| [mGPT/data/HumanML3D.py](mGPT/data/HumanML3D.py) | Dataset | Data loading, normalization, transforms |
| [prepare/instructions/template_instructions.json](prepare/instructions/template_instructions.json) | Instructions | 575+ instruction templates for Stage 3 |
| [prepare/instructions/template_pretrain.json](prepare/instructions/template_pretrain.json) | Pre-training | Basic templates for Stage 2 |

---

## Key Innovations

### 1. Motion as Language
- Converts continuous motion → discrete tokens (like words)
- Enables language models to process motion natively
- Shared vocabulary for text and motion

### 2. Unified Multi-Task Model
- Single model handles T2M, M2T, prediction, in-between
- No task-specific architectures needed
- Zero-shot transfer to new instructions

### 3. Progressive Training
- Stage 1: Learn motion vocabulary (VQ-VAE)
- Stage 2: Learn cross-modal alignment (pre-training)
- Stage 3: Learn instruction following (fine-tuning)

### 4. Instruction Diversity
- 575+ template variations
- Robust to different phrasings
- Strong zero-shot capabilities

---

## Training Commands

### Stage 1: Train Motion Tokenizer
```bash
python -m train --cfg configs/config_h3d_stage1.yaml --nodebug
```

### Stage 2: Pre-train MotionGPT
```bash
# First, extract motion tokens for efficiency
python -m scripts.get_motion_code --cfg configs/config_h3d_stage2.yaml

# Then train
python -m train --cfg configs/config_h3d_stage2.yaml --nodebug
```

### Stage 3: Instruction Tuning
```bash
python -m train --cfg configs/config_h3d_stage3.yaml --nodebug
```

### Evaluation
```bash
# Text-to-Motion
python -m test --cfg configs/config_h3d_stage3.yaml --task t2m

# Motion-to-Text
python -m test --cfg configs/config_h3d_stage3.yaml --task m2t

# Motion Prediction
python -m test --cfg configs/config_h3d_stage3.yaml --task pred

# Motion In-between
python -m test --cfg configs/config_h3d_stage3.yaml --task inbetween
```

---

## References

- **Paper**: [MotionGPT: Human Motion as a Foreign Language](https://arxiv.org/abs/2306.14795)
- **Project Page**: https://motion-gpt.github.io/
- **HuggingFace Demo**: https://huggingface.co/spaces/OpenMotionLab/MotionGPT
- **Dataset**: [HumanML3D](https://github.com/EricGuo5513/HumanML3D)

---

## Citation

```bibtex
@article{jiang2024motiongpt,
  title={Motiongpt: Human motion as a foreign language},
  author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

---

**Document Generated**: 2025-10-17
**Analysis Tools**: Claude Code + Manual Code Review
**Repository Version**: Main branch (post NeurIPS 2023 acceptance)
