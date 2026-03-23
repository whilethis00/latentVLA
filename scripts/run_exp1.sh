#!/bin/bash
# ============================================================
# Exp 1: z_shuffle_gap 측정 (Aha 1 — z 품질 ↔ 성능 상관관계)
# 기존 M1~M4 + VLM-z 모델들의 오프라인 지표 측정 후
# scatter plot 생성
#
# 사용법: bash scripts/run_exp1.sh
# 전제: 기존 M1~M4 학습 완료, VLM Exp2 학습 완료
# ============================================================

set -e

RESULTS_DIR="outputs/exp1"
mkdir -p $RESULTS_DIR

echo "=============================="
echo " Exp 1: z 품질 지표 수집"
echo "=============================="

# ── M1~M4 베이스라인 ────────────────────────────────────────
for MODEL in flat_flow det_latent stoch_vae stoch_flow_prior; do
    RUN_DIR="outputs/runs/$MODEL"
    if [ ! -d "$RUN_DIR" ]; then
        echo "[SKIP] $RUN_DIR 없음 — 먼저 run_ablations.sh 실행 필요"
        continue
    fi
    echo ">>> $MODEL 평가 중..."
    python scripts/evaluate_offline.py \
        --run_dir $RUN_DIR \
        --best_of_k 1 3 5 10 \
        --temperature_sweep 1.0 0.7 0.5 0.3 \
        --save_path $RESULTS_DIR/${MODEL}_metrics.json
    echo ">>> $MODEL 완료"
done

# ── VLM 모델 ────────────────────────────────────────────────
for Z_FORM in last pool plan; do
    RUN_DIR="outputs/runs/vlm_sfp_$Z_FORM"
    if [ ! -d "$RUN_DIR" ]; then
        echo "[SKIP] $RUN_DIR 없음 — 먼저 run_exp2.sh 실행 필요"
        continue
    fi
    echo ">>> vlm_sfp_$Z_FORM 평가 중..."
    python scripts/evaluate_offline.py \
        --run_dir $RUN_DIR \
        --model_type vlm \
        --best_of_k 1 3 5 10 \
        --save_path $RESULTS_DIR/vlm_sfp_${Z_FORM}_metrics.json
    echo ">>> vlm_sfp_$Z_FORM 완료"
done

# ── Scatter plot 생성 ────────────────────────────────────────
echo ""
echo ">>> Exp 1 scatter plot 생성 중..."
python scripts/plot_exp1.py --results_dir $RESULTS_DIR

echo ""
echo "=============================="
echo " Exp 1 완료"
echo " 결과: $RESULTS_DIR/"
echo " 그래프: $RESULTS_DIR/scatter_z_quality.png"
echo "=============================="
