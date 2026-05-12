#!/bin/bash
# ============================================================
# M9: VLM-SFP + Prior-Action Co-training
#
# Usage:
#   bash scripts/run_m9_prior_action_cotrain.sh
#
# Optional:
#   NPROC=4 bash scripts/run_m9_prior_action_cotrain.sh
#   PYTHON_CMD="python" bash scripts/run_m9_prior_action_cotrain.sh
#   OUTPUT_DIR=outputs/runs/m9_test bash scripts/run_m9_prior_action_cotrain.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/vlm_paligemma_m9_prior_action_cotrain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/m9_prior_action_cotrain_$(date +%Y%m%d)}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
NPROC="${NPROC:-1}"

if [ "$NPROC" -gt 1 ]; then
    PYTHON_CMD="${PYTHON_CMD:-torchrun --nproc_per_node=$NPROC}"
else
    PYTHON_CMD="${PYTHON_CMD:-python}"
fi

echo "============================================================"
echo " M9: VLM-SFP + Prior-Action Co-training"
echo " config    : $CONFIG"
echo " output_dir: $OUTPUT_DIR"
echo " run_name  : $RUN_NAME"
echo " command   : $PYTHON_CMD"
echo "============================================================"

$PYTHON_CMD scripts/train_vlm.py \
    --config "$CONFIG" \
    --override training.output_dir="$OUTPUT_DIR" \
               logging.run_name="$RUN_NAME"

echo ""
echo "============================================================"
echo " M9 complete"
echo " output_dir: $OUTPUT_DIR"
echo "============================================================"
