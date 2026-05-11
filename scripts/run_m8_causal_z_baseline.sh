#!/bin/bash
# ============================================================
# M8 causal-z baseline
#
# Goal:
#   Re-evaluate the M8 checkpoint with separated prior/posterior
#   z interventions before starting M9.
#
# Usage:
#   bash scripts/run_m8_causal_z_baseline.sh
#
# Optional overrides:
#   PYTHON_CMD="python" bash scripts/run_m8_causal_z_baseline.sh
#   MAX_BATCHES=30 bash scripts/run_m8_causal_z_baseline.sh
#   DEVICE=cpu bash scripts/run_m8_causal_z_baseline.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_CMD="${PYTHON_CMD:-conda run -n vla python}"
CHECKPOINT="${CHECKPOINT:-outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt}"
MAX_BATCHES="${MAX_BATCHES:-}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-outputs/analyses/m8_causal_z_baseline}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[run_m8_causal_z_baseline] checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

echo "============================================================"
echo " M8 causal-z baseline"
echo " checkpoint : $CHECKPOINT"
echo " python     : $PYTHON_CMD"
echo " device     : $DEVICE"
echo " max_batches: ${MAX_BATCHES:-all}"
echo " out_root   : $OUT_ROOT"
echo "============================================================"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --mode both
    --device "$DEVICE"
)

if [ -n "$MAX_BATCHES" ]; then
    COMMON_ARGS+=(--max_batches "$MAX_BATCHES")
fi

for INTERVENTION in shuffle null random task_negative motion_negative; do
    OUT_DIR="$OUT_ROOT/$INTERVENTION"
    echo ""
    echo ">>> intervention=$INTERVENTION"
    $PYTHON_CMD scripts/eval_causal_z.py \
        "${COMMON_ARGS[@]}" \
        --intervention "$INTERVENTION" \
        --out_dir "$OUT_DIR"
done

echo ""
echo "============================================================"
echo " M8 causal-z baseline complete"
echo " reports: $OUT_ROOT/*/causal_z_report.md"
echo "============================================================"
