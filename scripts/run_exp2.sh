#!/bin/bash
# ============================================================
# Exp 2: VLM-z 3가지 form 학습 (Aha 2 + Aha 3 핵심 실험)
# 사용법: bash scripts/run_exp2.sh
# ============================================================

set -e

CONFIG="configs/vlm_paligemma.yaml"

echo "=============================="
echo " Exp 2: VLM z_form Ablation"
echo "=============================="

for Z_FORM in last pool plan; do
    echo ""
    echo ">>> z_form = $Z_FORM 학습 시작"
    python scripts/train_vlm.py \
        --config $CONFIG \
        --override system2.z_form=$Z_FORM \
                   training.output_dir=outputs/runs/vlm_sfp_$Z_FORM \
                   logging.run_name=vlm_sfp_$Z_FORM
    echo ">>> z_form = $Z_FORM 완료"
done

echo ""
echo "=============================="
echo " Exp 2 전체 완료"
echo " 결과 위치: outputs/runs/vlm_sfp_{last,pool,plan}/"
echo "=============================="
