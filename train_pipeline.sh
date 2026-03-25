#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
# Train MotionGPT model
echo "Starting MotionGPT training..."
cd /home/student/hwu/Workplace/MotionGPT
python -m train --cfg configs/deto_h2s_rvq_hierarchical_3layer_ytb.yaml --use_gpus 1,2 --nodebug

# Train SOKE model
echo "Starting SOKE training..."
cd /home/student/hwu/Workplace/SOKE
python -m train --cfg configs/deto_youtube.yaml --use_gpus 1,2 --nodebug

echo "Both training jobs completed."
