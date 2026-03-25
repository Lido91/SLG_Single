#!/bin/bash

# =============================================================================
# Complete Training Pipeline for MotionGPT
# =============================================================================
# Workflow:
#   1. Train VAE -> Update VAE config -> Test VAE
#   2. Get motion codes using trained VAE
#   3. Train T2M (Text-to-Motion) -> Update T2M config -> Test T2M
# =============================================================================

set -e  # Exit on error

# GPU Configuration
export CUDA_VISIBLE_DEVICES=1,2,3
GPUS="1,2,3"

# Config files
VAE_CONFIG="configs/deto_h2s_rvq_3_youtube.yaml"
T2M_CONFIG="configs/deto_h2s_rvq_hierarchical_3layer_ytb.yaml"

echo "=============================================="
echo "Starting Complete Training Pipeline"
echo "=============================================="
echo "VAE Config: $VAE_CONFIG"
echo "T2M Config: $T2M_CONFIG"
echo "GPUs: $GPUS"
echo "=============================================="

# =============================================================================
# Stage 1: VAE Training
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 1: Training VAE"
echo "=============================================="

python -m train --cfg $VAE_CONFIG --use_gpus $GPUS --nodebug

echo ""
echo "Updating VAE config with best checkpoint..."
python update_config_checkpoints.py -c $VAE_CONFIG

echo ""
echo "=============================================="
echo "Stage 1: Testing VAE"
echo "=============================================="

python -m test --cfg $VAE_CONFIG --use_gpus $GPUS

# =============================================================================
# Stage 2: Generate Motion Codes
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 2: Generating Motion Codes"
echo "=============================================="

python -m get_motion_code --cfg $VAE_CONFIG --nodebug --use_gpus $GPUS

# =============================================================================
# Stage 3: T2M (Text-to-Motion) Training
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 3: Copying PRETRAINED_VAE to T2M config"
echo "=============================================="

python copy_pretrained_vae.py -f $VAE_CONFIG -t $T2M_CONFIG

echo ""
echo "=============================================="
echo "Stage 3: Training T2M"
echo "=============================================="

python -m train --cfg $T2M_CONFIG --use_gpus $GPUS --nodebug

echo ""
echo "Updating T2M config with best checkpoint..."
python update_config_checkpoints.py -c $T2M_CONFIG

echo ""
echo "=============================================="
echo "Stage 3: Testing T2M"
echo "=============================================="

python -m test --cfg $T2M_CONFIG --use_gpus $GPUS --nodebug

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "VAE checkpoint: See $VAE_CONFIG"
echo "T2M checkpoint: See $T2M_CONFIG"
echo "=============================================="
