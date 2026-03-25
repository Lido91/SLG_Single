#!/bin/bash
# Train all codex contrastive configs (triplet, text-motion, speech-motion) for both H2S and YTB

set -e

# ---- How2Sign ----
# echo "=== [1/6] H2S Triplet (text + speech + motion) ==="
# python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_h2s.yaml

echo "=== [2/6] H2S Text-Motion ==="
python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_h2s_text_motion.yaml

echo "=== [3/6] H2S Speech-Motion ==="
python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_h2s_speech_motion.yaml

# # ---- Youtube3D ----
# echo "=== [4/6] YTB Triplet (text + speech + motion) ==="
# python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_ytb.yaml

echo "=== [5/6] YTB Text-Motion ==="
python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_ytb_text_motion.yaml

echo "=== [6/6] YTB Speech-Motion ==="
python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex_ytb_speech_motion.yaml

echo "=== All done ==="
