# M6: VLM SFP Plan + z-InfoNCE — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-14 |
| **베이스** | M5 (VLM SFP Plan, 100ep) |
| **목적** | z 공간에 InfoNCE loss를 추가해 task-discriminative 클러스터 강화 |
| **설정** | λ=0.1, temperature=0.07, 20ep 학습 |
| **현황** | 학습 완료 (20ep), 오프라인 eval 완료 (2026-04-16) |

---

## 2. 무엇을 검증하나

M5의 z_shuffle_gap이 0.016으로 낮아 task 정보가 z에 약하게 인코딩됨.
같은 task의 z는 가깝게, 다른 task의 z는 멀어지도록 InfoNCE loss를 추가했을 때:

- `z_shuffle_gap > 0.043` (M4 수준 초과) 달성 여부
- `action_mse_prior ≤ 0.608` (M5 수준) 유지 여부
- 학습 안정성 (후반부 collapse 없는지)

---

## 3. 학습 손실 곡선

| Epoch | total_loss | action_flow | prior_flow | sem_future | infonce |
|------:|:----------:|:-----------:|:----------:|:----------:|:-------:|
| 1 | 2.8612 | 1.2393 | 1.3439 | 0.1105 | 2.6704 |
| 2 | 2.1922 | 0.9189 | 1.0452 | 0.0358 | 2.2458 |
| 3 | 1.8249 | 0.8894 | 0.7059 | 0.0359 | 2.2607 |
| 4 | 1.4536 | 0.8380 | 0.3874 | 0.0356 | 2.2464 |
| 5 | 1.2602 | 0.7922 | 0.2391 | 0.0352 | 2.2534 |
| 10 | 1.6074 | 0.7148 | 0.7403 | 0.0344 | 1.4890 |
| 15 | 1.4934 | 0.6192 | 0.8121 | 0.0248 | 0.5966 |
| 17 | 1.6382 | 0.6101 | 0.9684 | 0.0235 | 0.5739 |
| 18 | 1.3512 | 0.6116 | 0.6858 | 0.0222 | 0.5160 |
| 19 | 1.3976 | 0.6063 | 0.7370 | 0.0222 | 0.5208 |
| 20 | 1.5271 | 0.5864 | 0.8830 | 0.0227 | 0.5547 |
| 21 (train only) | 1.6344 | 0.6051 | 0.9693 | 0.0231 | 0.5764 |

**관찰:**
- infonce_loss가 ep1~5에서 2.67 → 2.25로 완만하게 감소하다가, ep10~20 구간에서 1.49 → 0.55로 급감
- infonce loss가 낮아질수록 오히려 z_shuffle_gap도 붕괴 → InfoNCE가 "쉬운 해법"(클러스터 collapse)으로 수렴한 것으로 추정
- prior_flow_loss는 ep5 이후 진동 (0.24 → 0.88) → prior 학습 불안정

---

## 4. 검증 지표

| Epoch | action_mse_prior ↓ | action_mse_posterior ↓ | best_of_1 ↓ | best_of_5 ↓ | sampling_diversity ↑ | future_cosine_sim ↓ | z_shuffle_gap ↑ | prior_posterior_gap ↑ |
|------:|:------------------:|:----------------------:|:-----------:|:-----------:|:--------------------:|:-------------------:|:---------------:|:---------------------:|
| **5** | 1.7412 | 1.4570 | 1.7260 | 0.8355 | 0.7855 | **0.9647** | **0.1370** | **0.2842** |
| **10** | 1.2917 | 1.1940 | 1.3135 | 0.6379 | 0.6752 | 0.9678 | 0.0851 | 0.0976 |
| **15** | 0.9239 | 0.8649 | 0.9232 | 0.4390 | 0.5415 | 0.9758 | -0.0110 | 0.0590 |
| **20** | 0.8492 | 0.8208 | 0.8570 | 0.4330 | 0.4845 | 0.9766 | 0.0041 | 0.0284 |

### M5 대비 비교 (val 로그 + 오프라인 eval)

| 지표 | M5 | M6 val best (ep5) | M6 val last (ep20) | M6 offline ckpt_10 | M6 offline ckpt_20 | 목표 달성? |
|------|:--:|:-----------------:|:------------------:|:------------------:|:------------------:|:----------:|
| action_mse_prior ↓ | **0.6084** | 1.7412 | 0.8492 | 1.2946 | 0.8588 | ❌ |
| action_mse_posterior ↓ | **0.5318** | 1.4570 | 0.8208 | 1.1883 | 0.8168 | ❌ |
| z_shuffle_gap ↑ | 0.0163 | **0.1370** | 0.0041 | **0.0537** | 0.0268 | ✅ (ckpt_10) |
| prior_posterior_gap ↑ | 0.0766 | **0.2842** | 0.0284 | 0.1063 | 0.0420 | ✅ (ckpt_10) |
| future_cosine_sim ↓ | 0.9945 | **0.9647** | 0.9766 | **0.9678** | 0.9766 | ✅ (ckpt_10) |
| best_of_10 ↓ | — | — | — | 0.5033 | **0.3517** | — |
| sampling_diversity ↑ | — | — | — | **0.7027** | 0.5076 | — |

---

## 5. 결과 해석 및 인사이트

### 긍정적
- **ep5 기준 z_shuffle_gap=0.137** — M4(0.043) 목표를 3배 이상 초과. InfoNCE가 방향성은 맞음
- ep5에서 future_cosine_sim=0.965, prior_posterior_gap=0.284 — M5 대비 z 구조 뚜렷이 개선됨
- infonce_loss 자체는 ep1부터 하강 → loss 구현 및 계산 정상 동작 확인

### 주의
- **ep5 이후 z_shuffle_gap 급격 붕괴** (0.137 → 0.085 → -0.011 → 0.004): InfoNCE 효과가 지속되지 않음
- infonce_loss가 ep10~20에서 급감 (2.25 → 0.55) → z가 collapse 방향으로 수렴해 loss가 trivially 낮아진 것으로 추정
- action_mse_prior는 ep20에서 0.849 — M5 목표(0.608) 미달. flow prior가 InfoNCE와 경합해 학습 효율 저하
- sampling_diversity도 ep5(0.79) → ep20(0.48)으로 감소 → z 다양성 상실 진행 중

### ckpt_10 vs ckpt_20 트레이드오프

- **ckpt_10**: z 구조 살아있음 (z_shuffle_gap=0.054, 목표 달성). prior 비효율 — K 많이 뽑아야 좋은 action (best_of_10=0.503).
- **ckpt_20**: prior MSE 낮고 best_of_10=0.352로 더 좋음. 근데 z_shuffle_gap=0.027로 절반 감소.
- 학습이 진행될수록 **"prior 효율↑, z 구조↓"** 트레이드오프 발생 — z가 prior한테 끌려가서 구조 붕괴.

### 근본 원인 가설
1. **batch 내 positive pair 부족**: batch_size=16, 10 task → 평균 1.6개/task → ep 후반 동일 task 쌍이 충분치 않아 InfoNCE가 "모두 모으기" 대신 "하나로 collapse"로 빠짐
2. **λ=0.1이 prior flow loss와 경합**: prior_flow_loss 진동 구간과 z_shuffle_gap 붕괴 구간 일치

---

## 6. 다음 스텝

### 오프라인 eval
- [x] `ckpt_10.pt` eval 완료 (2026-04-16)
- [x] `ckpt_20.pt` eval 완료 (2026-04-16)

### M6 ablation
- [ ] λ=0.01로 낮춰서 재실행 — flow loss와 경합 완화
- [ ] batch_size 증가 또는 task-balanced sampler 도입 — positive pair 밀도 확보
- [ ] early stopping 기준: z_shuffle_gap < 0 시 조기 종료

### M7 방향 후보
- InfoNCE 대신 triplet loss (hard negative mining) — pair 부족 문제 완화 가능
- z에 직접 regularization보다 prior 네트워크에 task conditioning 추가

---

## 7. 저장 파일 목록

```
outputs/runs/vlm_sfp_infonce/
├── ckpt_10.pt                  ← z 구조 최선 (z_shuffle_gap=0.054 offline)
├── ckpt_20.pt                  ← prior MSE 최선 (best_of_10=0.352 offline)
├── train.log
├── train_log.jsonl
├── eval_ckpt_10_plan.json      ← 오프라인 eval (2026-04-16)
├── eval_ckpt_20_plan.json      ← 오프라인 eval (2026-04-16)
└── result.md                   ← 이 파일
```
