# Hierarchical RVQ-GPT Architecture
## High-Level Overview

---

## 🎯 Core Concept

```
Traditional Approach:           Hierarchical Approach (Ours):
━━━━━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━━━━━━━━━━━━━━

Text → All codes                Text → Coarse codes (Q0)
       at once                         ↓
                                    Medium codes (Q1) ← uses Q0
                                       ↓
                                    Fine codes (Q2) ← uses Q0+Q1

❌ No structure                  ✅ Natural hierarchy
❌ Codes independent             ✅ Coarse-to-fine refinement
```

**Key Innovation**: Generate motion codes in a coarse-to-fine hierarchy, where each refinement level depends on the previous level.

---

## 📊 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    HIERARCHICAL RVQ-GPT                         │
│                                                                 │
│  Input: Text or Speech                                         │
│  Output: Motion (sign language / human movement)               │
│                                                                 │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐             │
│  │  Coarse  │  →   │  Medium  │  →   │   Fine   │             │
│  │   (Q0)   │      │   (Q1)   │      │   (Q2)   │             │
│  └──────────┘      └──────────┘      └──────────┘             │
│       ↑                 ↑                  ↑                    │
│       └─────────────────┴──────────────────┘                   │
│                Text/Speech Input                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture Pipeline

### Stage 1: Input Encoding
```
┌─────────────┐
│   Input     │
│             │
│  "Hello"    │  or  🔊 Audio Waveform
│  (Text)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│      Text/Speech Encoder    │
│                             │
│  CLIP, BERT, HuBERT, etc.  │
└──────────┬──────────────────┘
           │
           ▼
    Feature Vector
    (1024 dimensions)
```

### Stage 2: Hierarchical Generation
```
┌────────────────────────────────────────────────────────┐
│                    LEVEL 1: COARSE                     │
│                                                        │
│  Input:  Text/Speech features                         │
│  Output: Q0 codes → Basic motion structure            │
│                                                        │
│  Example: "Person moves right arm"                    │
└─────────────────────┬──────────────────────────────────┘
                      │ Q0 codes
                      ▼
┌────────────────────────────────────────────────────────┐
│                   LEVEL 2: MEDIUM                      │
│                                                        │
│  Input:  Text/Speech features + Q0 codes ⭐           │
│  Output: Q1 codes → Add detail to structure           │
│                                                        │
│  Example: "Arm moves in arc, hand open"               │
└─────────────────────┬──────────────────────────────────┘
                      │ Q0 + Q1 codes
                      ▼
┌────────────────────────────────────────────────────────┐
│                    LEVEL 3: FINE                       │
│                                                        │
│  Input:  Text/Speech features + Q0 + Q1 codes ⭐      │
│  Output: Q2 codes → Add fine details                  │
│                                                        │
│  Example: "Fingers slightly bent, wrist rotates"      │
└─────────────────────┬──────────────────────────────────┘
                      │ Q0 + Q1 + Q2 codes
                      ▼
```

### Stage 3: Motion Decoding
```
┌─────────────────┐
│  Motion Codes   │
│  Q0, Q1, Q2     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RVQ-VAE       │
│   Decoder       │
│   (Frozen)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Motion   │
│  (3D poses)     │
└─────────────────┘
```

---

## 🔄 How Generation Works (Autoregressive)

For each time step, generate codes sequentially:

```
Time t=0:
  1. Generate Q0[0] (coarse)         based on: text
  2. Generate Q1[0] (medium)         based on: text + Q0[0]
  3. Generate Q2[0] (fine)           based on: text + Q0[0] + Q1[0]

Time t=1:
  1. Generate Q0[1] (coarse)         based on: text + Q0[0]
  2. Generate Q1[1] (medium)         based on: text + Q0[0:1] + Q1[0]
  3. Generate Q2[1] (fine)           based on: text + Q0[0:1] + Q1[0:1] + Q2[0]

...continue until EOS (end of sequence)
```

**Visual Flow**:
```
t=0           t=1           t=2
─────────────────────────────────
Q0:  ●    →    ●    →    ●    →  ...
     ↓         ↓         ↓
Q1:  ●    →    ●    →    ●    →  ...
     ↓         ↓         ↓
Q2:  ●    →    ●    →    ●    →  ...
```

---

## 🧩 Three Decoder Components

### Decoder Q0 (Coarse Motion)
```
┌─────────────────────────────┐
│  Purpose:                   │
│  • Capture global motion    │
│  • Basic structure          │
│  • Main movements           │
│                             │
│  Conditioning:              │
│  • Text/Speech only         │
│                             │
│  Size: ~82M parameters      │
└─────────────────────────────┘
```

### Decoder Q1 (Medium Refinement)
```
┌─────────────────────────────┐
│  Purpose:                   │
│  • Add detail to motion     │
│  • Refine Q0 structure      │
│  • Intermediate details     │
│                             │
│  Conditioning:              │
│  • Text/Speech              │
│  • Q0 codes ⭐              │
│                             │
│  Size: ~88M parameters      │
└─────────────────────────────┘
```

### Decoder Q2 (Fine Details)
```
┌─────────────────────────────┐
│  Purpose:                   │
│  • Final fine-tuning        │
│  • Subtle movements         │
│  • High-frequency details   │
│                             │
│  Conditioning:              │
│  • Text/Speech              │
│  • Q0 + Q1 codes ⭐         │
│                             │
│  Size: ~88M parameters      │
└─────────────────────────────┘
```

**Total Model Size**: ~258M parameters

---

## 🎨 Analogy: Painting a Picture

Think of it like an artist creating a painting:

```
┌────────────────────────────────────────────────────┐
│                                                    │
│  Q0 (Coarse):    Sketch the outline               │
│                  ▪▪▪▪▪▪▪                           │
│                  ▪    ▪                            │
│                  ▪▪▪▪▪▪▪                           │
│                  "Basic shape of person"           │
│                                                    │
│  Q1 (Medium):    Add main colors & shapes         │
│                  ███████                           │
│                  █  ◐  █                           │
│                  ███████                           │
│                  "Person with arms, head"          │
│                                                    │
│  Q2 (Fine):      Add details & textures           │
│                  ███████                           │
│                  █ ◐◡◐ █                           │
│                  ███◢◣██                           │
│                  "Facial features, fingers"        │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 📈 Why Hierarchical?

### Problem with Traditional Approach
```
❌ Traditional (Parallel Generation):

Text → [Q0, Q1, Q2, Q3, Q4, Q5] all at once

Problems:
  • Q2 doesn't know what Q0 decided
  • Codes might conflict with each other
  • No coordination between levels
  • Ignores natural coarse-to-fine structure
```

### Our Solution (Hierarchical)
```
✅ Hierarchical (Sequential Generation):

Text → Q0 → Q1 → Q2
       ↓    ↓    ↓
     coarse → medium → fine

Benefits:
  • Q1 refines Q0 explicitly
  • Q2 refines Q0+Q1 explicitly
  • Natural coordination
  • Matches how RVQ works internally
```

---

## 🔬 Technical Details (Simplified)

### What's Inside Each Decoder?

```
┌─────────────────────────────────────────┐
│         Single Decoder Block            │
│                                         │
│  1️⃣  Look at previous tokens           │
│      (Self-Attention)                   │
│         ↓                               │
│  2️⃣  Look at text/speech input         │
│      (Cross-Attention)                  │
│         ↓                               │
│  3️⃣  Look at previous quantizer codes  │
│      (Cross-Attention) ⭐ NEW!         │
│         ↓                               │
│  4️⃣  Process information                │
│      (Feed-Forward Network)             │
│         ↓                               │
│      Output                             │
│                                         │
│  × Repeat 9 times (9 layers)           │
└─────────────────────────────────────────┘
```

**Note**: Q0 doesn't have step 3 (no previous quantizers)

---

## 🎛️ Configuration Options

### Input Modality
```
Text Mode:
  • CLIP (512D)
  • BERT (768D)
  • BERT-Large (1024D)

Speech Mode:
  • HuBERT-Large (1024D)
  • Whisper-Large (1280D)
  • WavLM-Large (1024D)
```

### Architecture Variants
```
3-Layer (Default):
  • Uses Q0, Q1, Q2
  • ~258M parameters
  • 3× slower than baseline

6-Layer (Full):
  • Uses Q0, Q1, Q2, Q3, Q4, Q5
  • ~516M parameters
  • 6× slower than baseline
  • Maximum quality

T2M-Style:
  • Special Q0 architecture
  • Lighter Q1, Q2 (3 layers each)
  • ~180M parameters
  • Balanced speed/quality
```

---

## 📊 Performance Comparison

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Metric          Baseline    Hierarchical (Ours)    │
│  ─────────────────────────────────────────────────   │
│  Speed           Fast ⚡      Slower 🐢             │
│                  1× step      3× steps               │
│                                                      │
│  Quality         Good 👍      Better 🌟             │
│                               More coherent          │
│                                                      │
│  Coordination    None ❌      Explicit ✅           │
│                               Q0→Q1→Q2               │
│                                                      │
│  Parameters      ~180M        ~258M                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Use Cases

### Progressive Generation
```
Low Quality (Fast):
  Generate Q0 only → Decode → Motion
  Speed: 1× | Quality: Coarse

Medium Quality (Balanced):
  Generate Q0 + Q1 → Decode → Motion
  Speed: 2× | Quality: Medium

High Quality (Best):
  Generate Q0 + Q1 + Q2 → Decode → Motion
  Speed: 3× | Quality: Fine
```

### Multi-Modal Generation
```
Text → Motion:
  "A person waves hello"
  → Encoder → Hierarchical GPT → Motion

Speech → Motion:
  🔊 Audio recording
  → Encoder → Hierarchical GPT → Motion
```

---

## 🎓 Key Takeaways

1. **Hierarchical Structure**: Generate motion codes from coarse to fine
   - Q0: Basic structure
   - Q1: Medium details (uses Q0)
   - Q2: Fine details (uses Q0+Q1)

2. **Explicit Conditioning**: Each level explicitly depends on previous levels
   - Better coordination between quantizers
   - More coherent motion

3. **Flexible**: Can generate at different quality levels
   - Trade-off between speed and quality

4. **Multi-Modal**: Supports both text and speech input
   - CLIP/BERT for text
   - HuBERT/Whisper for speech

5. **Scalable**: Can extend to more levels (Q3, Q4, Q5...)
   - More layers = higher quality

---

## 📁 Quick Reference

```
Main Architecture:
  mGPT/archs/mgpt_rvq_hierarchical.py

Configurations:
  configs/deto_h2s_rvq_hierarchical_3layer.yaml    (3-layer)
  configs/deto_h2s_rvq_hierarchical_6layer.yaml    (6-layer)
  configs/deto_h2s_rvq_hierarchical_t2m.yaml       (T2M-style)

Documentation:
  document/HIERARCHICAL_RVQ_IMPLEMENTATION.md      (Technical)
  hierarchical_gpt_architecture.md                  (Detailed)
  hierarchical_gpt_architecture_highlevel.md        (This file)
```

---

## 🎯 Summary in One Sentence

**"Instead of generating all motion codes at once, we generate them hierarchically (coarse→medium→fine), where each refinement level explicitly conditions on the previous level, leading to more coherent and structured motion generation."**

---

*Generated for MotionGPT - Hierarchical RVQ-GPT Implementation*
