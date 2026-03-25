#!/bin/bash
# Evaluate all contrastive checkpoints and record results
# Usage: bash eval_contrastive_all.sh [GPU_ID]

GPU=${1:-0}
SPLIT="test"
BATCH_SIZE=32
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
name="new_contrastive_codex_h2s_speech_motion"
CONFIG="contrastive/configs/${name}.yaml"
CKPT_DIR="experiments/contrastive_codex/${name}/checkpoints"
RESULTS_DIR="experiments/contrastive_codex/${name}/eval_results"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary_${SPLIT}_${TIMESTAMP}.txt"

echo "============================================================" | tee "$SUMMARY_FILE"
echo " Contrastive Evaluation Summary — split=${SPLIT}" | tee -a "$SUMMARY_FILE"
echo " $(date)" | tee -a "$SUMMARY_FILE"
echo " GPU: ${GPU}  Batch: ${BATCH_SIZE}" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"

# Collect all checkpoint files, sorted
CKPTS=($(ls "$CKPT_DIR"/*.ckpt | sort))

echo "" | tee -a "$SUMMARY_FILE"
echo "Found ${#CKPTS[@]} checkpoints:" | tee -a "$SUMMARY_FILE"
for c in "${CKPTS[@]}"; do
    echo "  $(basename $c)" | tee -a "$SUMMARY_FILE"
done
echo "" | tee -a "$SUMMARY_FILE"

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT" .ckpt)
    LOG_FILE="$RESULTS_DIR/${CKPT_NAME}_${SPLIT}.log"

    echo "============================================================" | tee -a "$SUMMARY_FILE"
    echo " Evaluating: $CKPT_NAME" | tee -a "$SUMMARY_FILE"
    echo "============================================================" | tee -a "$SUMMARY_FILE"

    python -m contrastive.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --gpu "$GPU" \
        2>&1 | tee "$LOG_FILE"

    # Extract Global Retrieval section into summary
    echo "" >> "$SUMMARY_FILE"
    echo ">>> $CKPT_NAME — Global Retrieval:" >> "$SUMMARY_FILE"
    sed -n '/Global Retrieval/,$ p' "$LOG_FILE" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

echo "============================================================" | tee -a "$SUMMARY_FILE"
echo " All evaluations complete!" | tee -a "$SUMMARY_FILE"
echo " Full logs:  $RESULTS_DIR/*_${SPLIT}.log" | tee -a "$SUMMARY_FILE"
echo " Summary:    $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"

echo ""
echo "=== SUMMARY ==="
cat "$SUMMARY_FILE"
