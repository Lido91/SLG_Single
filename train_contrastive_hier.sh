#!/bin/bash
# Train contrastive learning + hierarchical RVQ pipeline
# Step 1: Train contrastive model (speech-motion alignment)
# Step 2: Train hierarchical RVQ GPT with contrastive features

set -e

GPU=${1:-0}

# # Step 1: Contrastive pre-training
# python -m contrastive.train \
#     --config contrastive/configs/contrastive_h2s.yaml \
#     --use_gpus "$GPU"

# Step 2: Hierarchical RVQ GPT with HuBERT + contrastive features
python -m train \
    --cfg configs/deto_h2s_rvq_hierarchical_3layer_hubert_contrastive.yaml \
    --use_gpus "$GPU" \
    --nodebug
