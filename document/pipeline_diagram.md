# MotionGPT Pipeline Architecture

## Overview
MotionGPT is a two-stage framework for motion-language tasks with three training modes:
1. **Stage 1 (VAE)**: Motion Tokenization using VQ-VAE
2. **Stage 2a (LM_PRETRAIN)**: Motion-Language Pretraining
3. **Stage 2b (LM_INSTRUCT)**: Motion-Language Instruction Tuning

---

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MOTIONGPT ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                         STAGE 1: MOTION TOKENIZER (VAE)                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

Input Motion Sequence                                Training:
(B, T, 133)                                          - Reconstruction Loss
     │                                               - Commitment Loss
     ▼                                               - Perplexity Tracking
┌──────────────────┐
│  Preprocess      │  Permute (B, T, 133) → (B, 133, T)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   VQ Encoder     │  Conv1D blocks with ResNet1D
│                  │  - Input: 133 dims (motion features)
│  down_t=3        │  - Output: 512 dims (code_dim)
│  stride_t=2      │  - Downsampling: T → T/8
│  width=512       │
│  depth=3         │
└────────┬─────────┘
         │
         ▼
   Latent Features
   (B, 512, T/4)
         │
         ▼
┌──────────────────┐
│   Quantizer      │  QuantizeEMAReset (EMA + Reset)
│                  │  - Codebook size: 512
│  code_num=512    │  - Projects to discrete tokens
│  code_dim=512    │  - EMA update (mu=0.99)
│  mu=0.99         │
└────────┬─────────┘
         │
         ▼
  Quantized Latent  ────┐
  (B, 512, T/4)         │ Motion Tokens
         │              │ (B, T/4)
         ▼              │ [0-511] discrete IDs
┌──────────────────┐    │
│   VQ Decoder     │    │
│                  │    │
│  Conv1D blocks   │    │
│  Upsample×3      │    │
│  + ResNet1D      │    │
└────────┬─────────┘    │
         │              │
         ▼              │
┌──────────────────┐    │
│  Postprocess     │    │
└────────┬─────────┘    │
         │              │
         ▼              │
Reconstructed Motion    │
(B, T, 133)             │
                        │
                        │
╔═══════════════════════════════════════════════════════════════════════════╗
║              STAGE 2: MOTION-LANGUAGE MODEL (T5/GPT-2)                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
                        │
                        │ Motion Tokens Feed Into LM
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼

┌─────────────────────────────┐  ┌─────────────────────────────┐
│    Text-to-Motion (T2M)     │  │   Motion-to-Text (M2T)      │
└─────────────────────────────┘  └─────────────────────────────┘

TEXT INPUT                       MOTION INPUT
"A person walks forward"         Motion Tokens (from VAE encode)
     │                                │
     ▼                                ▼
┌──────────────────┐            ┌──────────────────┐
│ Motion Token     │            │ Encode Motion    │
│ String Format    │            │ to String        │
│                  │            │                  │
│ <motion_id_512>  │            │ <motion_id_512>  │
│ <motion_id_0>    │            │ <motion_id_45>   │
│ <motion_id_123>  │            │ <motion_id_234>  │
│ ...              │            │ ...              │
│ <motion_id_513>  │            │ <motion_id_513>  │
└────────┬─────────┘            └────────┬─────────┘
         │                                │
         ▼                                ▼
┌──────────────────────────────────────────────────────────┐
│         TEMPLATE FULFILLMENT ENGINE                      │
│                                                          │
│  Pretrain Templates:                                     │
│  - Supervised: Text → Motion / Motion → Text             │
│  - Unsupervised: Text → Text / Motion → Motion (T5 MLM)  │
│                                                          │
│  Instruct Templates (task-specific):                     │
│  - "Generate motion: <Caption_Placeholder>"              │
│  - "Generate text: <Motion_Placeholder>"                 │
│  - "Predict motion: <Motion_Placeholder_s1>"             │
│  - "Complete masked motion: <Motion_Placeholder_Masked>" │
│  - With length: "<Frame_Placeholder> frames: ..."        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│            T5/GPT-2 LANGUAGE MODEL                       │
│                                                          │
│  T5 (Encoder-Decoder):                                   │
│  ┌────────────┐         ┌────────────┐                  │
│  │  Encoder   │────────▶│  Decoder   │                  │
│  │ (Input)    │         │ (Output)   │                  │
│  └────────────┘         └────────────┘                  │
│                                                          │
│  GPT-2 (Decoder-only):                                   │
│  ┌──────────────────────────────────┐                   │
│  │   Decoder (Input + Output)       │                   │
│  └──────────────────────────────────┘                   │
│                                                          │
│  Token Embeddings:                                       │
│  - Original vocab (32000 for T5)                         │
│  - Added motion tokens (512 + 3 special tokens)          │
│    * <motion_id_512>: START token                        │
│    * <motion_id_513>: END token                          │
│    * <motion_id_514>: MASK token                         │
│                                                          │
│  Training:                                               │
│  - Cross-entropy loss on target tokens                   │
│  - Supervised + Unsupervised (MLM for T5)                │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
                  Generated Tokens
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   Motion Token IDs                Text Output
   [45, 234, 123, ...]             "A person walks"
         │
         ▼
┌──────────────────┐
│  VQ Decoder      │  Dequantize & Decode
│  (from VAE)      │
└────────┬─────────┘
         │
         ▼
  Generated Motion
  (B, T', 133)


╔═══════════════════════════════════════════════════════════════════════════╗
║                    DATA FLOW DURING TRAINING                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│ Dataset: How2Sign / CSL / Phoenix                                     │
│                                                                        │
│ Raw Data:                                                              │
│ - Motion: (T, 133) - Pose features (joints, hands, face)              │
│ - Text: Sign language captions/glosses                                │
│ - Length: Motion sequence length                                      │
│ - FPS: Frame rate                                                     │
└────────────────────┬───────────────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Normalization │  (x - mean) / std
            └────────┬───────┘
                     │
                     ▼
            ┌────────────────┐
            │  Length Check  │  Min: 40, Max: 196 frames
            │  & Resample    │  Adjust to unit_length=4
            └────────┬───────┘
                     │
                     ▼
              Batch (B, T, 133)
                     │
         ┌───────────┴──────────┐
         │                      │
         ▼                      ▼
  [STAGE 1: VAE]         [STAGE 2: LM]
  Train Tokenizer        (VAE frozen)
         │                      │
         │                      ▼
         │              ┌──────────────┐
         │              │ VAE Encode   │
         │              │ to Tokens    │
         │              └──────┬───────┘
         │                     │
         │                     ▼
         │              Motion Tokens (B, T/4)
         │                     │
         │                     ▼
         │              ┌──────────────────┐
         │              │ String Conversion│
         │              │ + Template       │
         │              └──────┬───────────┘
         │                     │
         │                     ▼
         │              ┌──────────────────┐
         │              │ LM Forward       │
         │              │ T5/GPT-2 Training│
         │              └──────────────────┘
         │
         ▼
   VAE Loss:
   - Reconstruction (L1 smooth)
   - Commitment (0.02)


╔═══════════════════════════════════════════════════════════════════════════╗
║                    INFERENCE TASKS                                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. TEXT-TO-MOTION (T2M)
   ───────────────────
   Input: "A person raises their hand"
          ↓
   Template: "Generate motion: A person raises their hand"
          ↓
   T5/GPT-2 Generate → Motion Token IDs [12, 45, 234, ...]
          ↓
   VAE Decode → Motion (T, 133)
          ↓
   Denormalize → Final Motion


2. MOTION-TO-TEXT (M2T)
   ────────────────────
   Input: Motion (T, 133)
          ↓
   VAE Encode → Motion Tokens [45, 123, ...]
          ↓
   String Format: "<motion_id_512><motion_id_45><motion_id_123>...<motion_id_513>"
          ↓
   Template: "Generate text: <Motion_Placeholder>"
          ↓
   T5/GPT-2 Generate → "A person raises their hand"


3. MOTION PREDICTION (PRED)
   ─────────────────────────
   Input: First 20% of motion tokens
          ↓
   Template: "Predict motion: <Motion_Placeholder_s1>"
          ↓
   Generate remaining 80% tokens
          ↓
   VAE Decode → Completed motion


4. MOTION INBETWEEN
   ─────────────────
   Input: Motion with middle 50% masked
          ↓
   Template: "Complete the masked motion: <Motion_Placeholder_Masked>"
          ↓
   Generate masked tokens
          ↓
   VAE Decode → Completed motion


╔═══════════════════════════════════════════════════════════════════════════╗
║                    KEY COMPONENTS SUMMARY                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│ Component          │ Details                                           │
├────────────────────┼───────────────────────────────────────────────────┤
│ Motion Features    │ 133 dims (body joints + hands + face)            │
│ Codebook Size      │ 512 discrete motion tokens                       │
│ Downsample Rate    │ 4× (T → T/4 tokens)                              │
│ Token Embedding    │ 512 dims per code                                │
│ LM Model           │ T5-base or GPT-2                                 │
│ Max Sequence       │ 256 tokens (T5/GPT-2)                            │
│ Training Tasks     │ T2M, M2T, Pred, Inbetween                        │
│ Batch Size         │ 8 (train), 16 (eval/test)                       │
│ Learning Rate      │ 2e-4 with Cosine Annealing                       │
└────────────────────┴───────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════╗
║                    EVALUATION METRICS                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

VAE Stage:
- Reconstruction Loss (L1 smooth)
- Motion Reconstruction metrics (MRMetrics)

LM Stage:
- Text-to-Motion: FID, R-Precision, Diversity, Multimodality (TM2TMetrics)
- Motion-to-Text: BLEU, ROUGE, CIDEr (M2TMetrics)
- Motion Reconstruction: If applicable (MRMetrics)

```

---

## File References

### Main Components
- **Model**: [mGPT/models/mgpt.py](../mGPT/models/mgpt.py) - Main MotionGPT model
- **VQ-VAE**: [mGPT/archs/mgpt_vq.py](../mGPT/archs/mgpt_vq.py) - Motion tokenizer
- **Language Model**: [mGPT/archs/mgpt_lm.py](../mGPT/archs/mgpt_lm.py) - T5/GPT-2 wrapper
- **Training**: [train.py](../train.py) - Training script
- **Config**: [configs/default.yaml](../configs/default.yaml) - Default configuration

### Key Methods
- **Training VAE**: `mgpt.py:304-320` - `train_vae_forward()`
- **Training LM**: `mgpt.py:123-135` - `train_lm_forward()`
- **T2M Inference**: `mgpt.py:138-210` - `val_t2m_forward()`
- **M2T Inference**: `mgpt.py:213-244` - `val_m2t_forward()`
- **Motion Tokenization**: `mgpt_vq.py:90-115` - `encode()` and `decode()`
- **LM Generation**: `mgpt_lm.py:279-384` - `generate_conditional()`

---

## Training Stages

### Stage 1: VAE Training
```bash
python train.py --cfg configs/vae_config.yaml
```
- Freezes: None
- Trains: VQ-VAE encoder + decoder + quantizer
- Loss: Reconstruction + Commitment

### Stage 2a: LM Pretraining
```bash
python train.py --cfg configs/pretrain_config.yaml --pretrained_vae path/to/vae.ckpt
```
- Freezes: VQ-VAE
- Trains: T5/GPT-2 language model
- Tasks: Supervised (T2M, M2T) + Unsupervised (MLM)

### Stage 2b: LM Instruction Tuning
```bash
python train.py --cfg configs/instruct_config.yaml --pretrained path/to/pretrained_lm.ckpt
```
- Freezes: VQ-VAE
- Trains: T5/GPT-2 with task-specific templates
- Tasks: T2M, M2T, Prediction, Inbetween

---

## Notes
- Motion tokens are treated as special tokens in the language model vocabulary
- Each motion token represents 4 frames of motion (downsampling factor)
- The system supports multiple datasets: How2Sign, CSL, Phoenix
- Template-based instruction following enables flexible task specification
