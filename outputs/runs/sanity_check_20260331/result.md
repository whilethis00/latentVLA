# Sanity Check 실험 결과

- **날짜**: 2026-03-31 ~ 2026-04-01
- **목적**: LatentVLA (StochFlowPrior, M4) 전체 파이프라인이 LIBERO-Object 데이터에서 정상 학습되는지 확인
- **설정**: Stage 1 only (PaliGemma frozen), 10 epoch, LIBERO-Object subset

---

## 무엇을 검증했나

1. `train_vlm.py` + `trainer_vlm.py` 전체 학습 루프가 에러 없이 완주되는지
2. loss가 epoch마다 단조 감소하는지 (학습이 실제로 이루어지고 있는지)
3. posterior가 prior보다 낮은 MSE를 달성하는지 (z가 유의미한 정보를 담는지)
4. best-of-K 샘플링이 K 증가에 따라 개선되는지 (분포의 다양성이 존재하는지)
5. 체크포인트 저장 (`ckpt_5.pt`, `ckpt_10.pt`, `ckpt_final.pt`) 정상 작동 여부

---

## 학습 손실 곡선

| Epoch | total_loss | action_flow | prior_flow | semantic_future |
|-------|-----------|-------------|------------|-----------------|
| 1     | 2.1219    | 1.0150      | 1.1010     | 0.0592          |
| 2     | 1.1046    | 0.7025      | 0.3988     | 0.0333          |
| 3     | 1.0095    | 0.6794      | 0.3272     | 0.0286          |
| 4     | 0.8124    | 0.6033      | 0.2070     | 0.0210          |
| 5     | 0.7139    | 0.5594      | 0.1529     | 0.0152          |
| 6     | 0.6411    | 0.5200      | 0.1199     | 0.0124          |
| 7     | 0.5942    | 0.4896      | 0.1035     | 0.0112          |
| 8     | 0.5621    | 0.4705      | 0.0906     | 0.0105          |
| 9     | 0.5400    | 0.4568      | 0.0822     | 0.0102          |
| 10    | 0.5273    | 0.4489      | 0.0774     | 0.0101          |

---

## 검증 지표 (Val)

| 지표 | Epoch 5 | Epoch 10 |
|------|---------|---------|
| action_mse_prior | 1.2833 | 1.0542 |
| action_mse_posterior | 0.7463 | 0.6451 |
| prior_posterior_gap | 0.5370 | 0.4091 |
| best_of_1 | 1.2782 | 1.0413 |
| best_of_5 | 0.5185 | 0.4408 |
| future_cosine_sim | 0.9864 | 0.9897 |
| z_shuffle_gap | -0.0038 | 0.0350 |
| sampling_diversity | 0.6714 | 0.5677 |

---

## 결과 해석 및 인사이트

### 긍정적
- **loss 단조 감소** — 10 epoch 내내 total_loss 2.12 → 0.53, 정상 수렴
- **posterior < prior** — Epoch 10에서 0.6451 < 1.0542, z가 실제로 유용한 정보를 인코딩하고 있음
- **best_of_5 << best_of_1** — 1.04 → 0.44 (57% 감소), 분포에 의미 있는 다양성 존재
- **future_cosine_sim ≈ 0.99** — 미래 의미 표현 예측이 매우 정확, semantic_future_loss 잘 학습됨

### 주의
- **z_shuffle_gap이 작음 (0.035)** — z를 셔플해도 성능 저하가 거의 없음. z가 아직 action-critical 정보를 충분히 인코딩하지 못하고 있을 가능성. Epoch 5에서는 심지어 음수(-0.004)였다가 10에서 양수로 전환된 것은 고무적이나, 더 긴 학습이 필요함
- **prior_posterior_gap 감소 추세** — Ep5: 0.537 → Ep10: 0.409. z 활용이 줄어드는 신호일 수 있음. 100 epoch 학습에서 추이 관찰 필요

---

## 다음 스텝

1. **Stage 1 full run** — 100 epoch, LIBERO-Object 전체 데이터로 본 학습 시작. `z_shuffle_gap`이 계속 증가하는지 확인
2. **M1~M3 동일 조건 평가** — `evaluate_offline_vlm.py`로 FlatFlow, DetLatent, StochVAE 각각 실행해 `eval_results.json` 생성 → 비교표 완성
3. **prior_posterior_gap 추이 모니터링** — 100 epoch 학습 중 gap이 수렴/발산하는지 확인, 필요 시 KL weight 또는 prior flow step 수 조정 검토
4. **Stage 2 (LoRA fine-tune) 설정** — Stage 1 full 완료 후 PaliGemma unfreezing 범위 결정

---

## 저장 파일

| 파일 | 내용 |
|------|------|
| `ckpt_5.pt` | Epoch 5 체크포인트 |
| `ckpt_10.pt` | Epoch 10 체크포인트 |
| `ckpt_final.pt` | 최종 체크포인트 (ckpt_10과 동일) |
| `train_log.jsonl` | Epoch별 전체 loss / val 지표 |
