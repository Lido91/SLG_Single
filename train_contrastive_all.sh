#!/bin/bash

# 1. Speech + Text + Motion (triplet)
python -m contrastive.train --config contrastive/configs/contrastive_h2s.yaml --use_gpus 0,1,2 --nodebug

# 2. Speech + Motion
python -m contrastive.train --config contrastive/configs/contrastive_speech_motion.yaml --use_gpus 0,1,2 --nodebug

# 3. Text + Motion
python -m contrastive.train --config contrastive/configs/contrastive_text_motion.yaml --use_gpus 0,1,2 --nodebug
