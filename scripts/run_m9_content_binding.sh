#!/bin/bash
# ============================================================
# M9: Posterior z content separation + content CF binding
#
# Usage:
#   bash scripts/run_m9_content_binding.sh
#
# Optional:
#   NPROC=4 bash scripts/run_m9_content_binding.sh
#   PYTHON_CMD="python" bash scripts/run_m9_content_binding.sh
#   OUTPUT_DIR=outputs/runs/m9_test bash scripts/run_m9_content_binding.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/vlm_paligemma_m9_content_binding.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/m9_content_binding_$(date +%Y%m%d)}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
NPROC="${NPROC:-1}"

if [ "$NPROC" -gt 1 ]; then
    PYTHON_CMD="${PYTHON_CMD:-torchrun --nproc_per_node=$NPROC}"
else
    PYTHON_CMD="${PYTHON_CMD:-python}"
fi

echo "============================================================"
echo " M9: Posterior z content separation + content CF binding"
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
