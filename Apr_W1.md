# Apr W1 (2026-03-31 ~ 2026-04-06)

> 목표: 코드 완성 → smoke test 통과 → 오프라인 평가 파이프라인 VLM 지원 완성
> paper.md 4월 목표 기준: "Smoke test + LIBERO 빠른 학습 확인"

---

## 현재 상태 (2026-03-31 기준)

| 파일 | 상태 |
|------|------|
| `models/system2_vlm.py` | ✅ 완료 + transformers 5.x 버그 3개 수정 |
| `models/latent_vla.py` | ✅ 완료 + load graceful 처리 |
| `models/encoders.py` | ✅ `encode_image_only()` API 수정 |
| `training/trainer_vlm.py` | ✅ 완료 + `max_batches` 옵션 |
| `training/builder.py` | ✅ `build_vlm_model(proprio_dim=)` 추가 |
| `configs/vlm_paligemma.yaml` | ✅ 존재 |
| `scripts/smoke_test_vlm.py` | ✅ 3/3 통과 |
| `scripts/train_vlm.py` | ✅ 존재 |
| `scripts/evaluate_offline_vlm.py` | ✅ 신규 작성, LIBERO end-to-end 통과 |
| `evaluation/metrics.py` | ✅ M1~M4용 (기존 유지) |

---

## 이번 주 태스크

### Day 1-2 (3/31~4/1): Smoke Test 통과 ✅

- [x] `smoke_test_vlm.py` 실행해서 PaliGemma 로드 + forward pass 확인
- [x] real 모드 3/3 통과 확인 (last/pool/plan 전부)
- [x] LatentVLA forward pass shape 체크 → `(2,8,7)`, `predict×3 (2,3,8,7)` ✓

**수정된 버그:**
- `system2_vlm.py` `max_length=128→512` (이미지 토큰 256개 truncation 방지)
- `system2_vlm.py` `vlm_feat.float()` + `token_type_ids` 자동생성 (transformers 5.x)
- `encoders.py` `encode_image_only()` → `vision_model.pooler_output + F.normalize()` (API 변경 대응)

### Day 3-4 (4/2~4/3): 오프라인 평가 파이프라인 VLM 지원 ✅

- [x] `scripts/evaluate_offline_vlm.py` 작성 및 동작 확인
  - 옵션: `--checkpoint`, `--config`, `--data_path`, `--z_form`, `--sweep_k`, `--max_batches`
  - 결과 JSON 자동 저장
- [x] LIBERO 데이터로 end-to-end 평가 통과 (5 batches)

**출력 지표 확인:**
```
action_mse_prior    : 2.9994
action_mse_posterior: 2.6400
prior_posterior_gap : 0.3593   ← z가 정보를 담고 있음
z_shuffle_gap       : 0.0766
best_of_1           : 2.7889
best_of_5           : 2.2992   ← K 증가할수록 개선됨
sampling_diversity  : 0.9554
future_cosine_sim   : 0.1333
```

**수정 파일:**
- `builder.py` — `build_vlm_model(proprio_dim=)` 파라미터 추가
- `latent_vla.py` — `load_state_dict_from_save` shape 불일치 graceful 처리
- `trainer_vlm.py` — `evaluate(max_batches=)` 옵션 추가

### Day 5 (4/4): 빠른 학습 실험 (Sanity Check)

- [x] LIBERO-Object, 10 epoch, Stage 1만 (PaliGemma frozen)
  ```bash
  conda run -n vla torchrun --nproc_per_node=GPU scripts/train_vlm.py \
    --config configs/vlm_paligemma.yaml \
    --override training.max_epochs=10
  ```
- [x] `curve_ep010.png` 저장되는지, loss 감소 확인
- [x] `latest.pt` 체크포인트 저장 확인

**결과 (2026-04-01):**
```
total_loss=0.5273  action_flow_loss=0.4489  prior_flow_loss=0.0774  semantic_future_loss=0.0101
action_mse_prior=1.0542  action_mse_posterior=0.6451  prior_posterior_gap=0.4091
best_of_1=1.0413  best_of_5=0.4408  future_cosine_sim=0.9897  z_shuffle_gap=0.0350
```
체크포인트: `outputs/runs/sanity_check/ckpt_10.pt`, `ckpt_final.pt`

### Day 6-7 (4/5~4/6): 버퍼 + 점검

- [ ] smoke test 또는 학습 중 발견된 버그 수정
- [ ] `experiments.md` 업데이트 (sanity check 결과 기록)
- [ ] Apr_W2 계획 초안 작성
- [ ] OfflineEvaluator 전체 테스트 — M1~M3 동일 조건 평가 실행 (`eval_results.json` 생성)
- [ ] M1~M4 비교표 초안 작성 (`experiment_easy.md` 섹션 10 채우기)

---

## 완료 기준 (Definition of Done)

1. ✅ `smoke_test_vlm.py` 에러 없이 통과
2. ✅ `evaluate_offline_vlm.py`가 LatentVLA 체크포인트를 받아 z_shuffle_gap 출력 가능
3. ✅ LIBERO-Object 10 epoch 학습 완주 + loss curve 저장 (Day 5)

---

## 다음 주 예정 (Apr W2)

- Stage 1 full (100 epoch) 학습 런 시작
- Exp 1: M1~M4 z_shuffle_gap vs 성공률 scatter plot
- Stage 2 (LoRA fine-tune) 실험 설정

---

## 참고

- `evaluation/metrics.py` L26~51: OfflineEvaluator 클래스 진입점
- `scripts/evaluate_offline.py` L32~47: 기존 모델 로드 방식 (enc + policy 분리)
- `models/latent_vla.py`: `state_dict_for_save()` → LoRA 가중치만 저장
- GPU walltime 72h 제한 → latest.pt에서 자동 resume
