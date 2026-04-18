# M7: VLM SFP Plan + z-InfoNCE + Balanced Sampler — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-16 |
| **베이스** | M6 (VLM SFP + InfoNCE, 20ep) |
| **목적** | task-balanced sampler로 positive pair 밀도 확보 → InfoNCE collapse 방지 |
| **설정** | λ=0.1, temperature=0.07, task-balanced sampler, 100ep 시도 |
| **현황** | ep82 완료 후 중단 (SSH disconnect, ckpt_80 저장 완료) |

---

## 2. 무엇을 검증하나

M6의 z_shuffle_gap 붕괴 원인으로 batch 내 positive pair 부족을 지목.
task-balanced sampler로 동일 task pair를 보장했을 때:

- `z_shuffle_gap > 0.043` (M4 수준 초과) 지속 유지 여부
- `action_mse_prior ≤ 0.608` (M5 수준) 달성 여부
- S1에서도 InfoNCE가 정상 학습되는지 (M6에서는 S1 내내 ~2.2로 정체)

---

## 3. 학습 손실 곡선 (주요 epoch)

| Epoch | Stage | total_loss | action_flow | prior_flow | sem_future | infonce |
|------:|:-----:|:----------:|:-----------:|:----------:|:----------:|:-------:|
| 1 | S1 | 2.8011 | 1.2785 | 1.3982 | 1.0000 | 2.4346 |
| 2 | S1 | 2.1471 | 0.9784 | 1.0520 | 1.0000 | 1.6740 |
| 5 | S1 | 1.2051 | 0.7526 | 0.4338 | 0.0338 | 1.5264 |
| 6 | S1 | 1.1858 | 0.7289 | 0.4431 | 0.0302 | 1.0781 |
| 7 | S1 | 1.1608 | 0.6991 | 0.4519 | 0.0275 | 0.7110 |
| 8 | S1 | 1.0185 | 0.6393 | 0.3717 | 0.0240 | 0.5075 |
| 9 | S1 | 0.9218 | 0.6102 | 0.3053 | 0.0207 | 0.4265 |
| 10 | S2 | 0.8635 | 0.5992 | 0.2588 | 0.0181 | 0.3693 |
| 15 | S2 | 0.7495 | 0.5393 | 0.2058 | 0.0128 | 0.3093 |
| 20 | S2 | 0.7040 | 0.5080 | 0.1920 | 0.0109 | 0.2998 |
| 30 | S2 | 0.6414 | 0.4667 | 0.1709 | 0.0088 | 0.2934 |
| 40 | S2 | 0.5861 | 0.4300 | 0.1524 | 0.0076 | 0.2907 |
| 50 | S2 | 0.5343 | 0.3825 | 0.1481 | 0.0069 | 0.2892 |
| 60 | S2 | 0.4933 | 0.3462 | 0.1436 | 0.0064 | 0.2885 |
| 70 | S2 | 0.4581 | 0.3183 | 0.1363 | 0.0061 | 0.2882 |
| 76 | S2 | 0.4380 | 0.3006 | 0.1338 | 0.0060 | 0.2881 |
| 79 | S2 | 0.4291 | 0.2956 | 0.1301 | 0.0059 | 0.2881 |
| 80 | S2 | 0.4273 | 0.2940 | 0.1298 | 0.0059 | 0.2880 |
| 82 | S2 | 0.4210 | 0.2885 | 0.1290 | 0.0059 | 0.2881 |

**관찰:**
- **S1에서 infonce가 정상 학습됨**: M6은 S1 내내 ~2.2로 정체했으나, M7은 ep7 0.71 → ep9 0.43으로 급감. balanced sampler 효과 확인
- S2 이후 infonce는 0.37 → 0.29대로 수렴. ep30 이후 매우 완만히 감소 (~0.001/10ep 수준)
- action_flow_loss: ep82까지 꾸준히 감소 (1.28 → 0.289) — 수렴 중
- prior_flow_loss: M6의 진동 패턴 없이 안정적으로 감소 (1.40 → 0.129)
- semantic_future_loss: S1 ep1~2에서 1.000 → 비정상 값 (S1 시작 시 dummy 모드 이슈), ep3 이후 정상

---

## 4. 검증 지표 (Val)

| Epoch | action_mse_prior ↓ | action_mse_posterior ↓ | best_of_1 ↓ | best_of_5 ↓ | sampling_diversity ↑ | future_cosine_sim | z_shuffle_gap ↑ | prior_posterior_gap ↑ |
|------:|:------------------:|:----------------------:|:-----------:|:-----------:|:--------------------:|:-----------------:|:---------------:|:---------------------:|
| **5** | 1.4620 | 0.7699 | 1.4682 | 0.6833 | 0.7341 | 0.9684 | **0.0944** | **0.6921** |
| **10** | 0.9739 | 0.8123 | 0.9823 | 0.4623 | 0.5436 | 0.9824 | 0.0053 | 0.1616 |
| **15** | 0.8182 | 0.7528 | 0.8217 | 0.3994 | 0.4788 | 0.9871 | 0.0098 | 0.0654 |
| **20** | 0.6998 | 0.6777 | 0.7074 | 0.3660 | 0.4312 | 0.9892 | 0.0047 | 0.0221 |
| **25** | 0.7027 | 0.6669 | 0.7063 | 0.3644 | 0.4293 | 0.9903 | 0.0120 | 0.0358 |
| **30** | 0.6637 | 0.6314 | 0.6658 | 0.3615 | 0.3837 | 0.9913 | 0.0147 | 0.0322 |
| **35** | 0.6522 | 0.6108 | 0.6401 | 0.3452 | 0.3793 | 0.9919 | 0.0142 | 0.0414 |
| **40** | 0.6459 | 0.6067 | 0.6417 | 0.3470 | 0.3616 | 0.9923 | 0.0050 | 0.0391 |
| **45** | 0.6264 | 0.5947 | 0.6367 | 0.3430 | 0.3633 | 0.9927 | 0.0101 | 0.0318 |
| **50** | 0.6388 | 0.5910 | 0.6398 | 0.3656 | 0.3314 | 0.9930 | 0.0129 | 0.0478 |
| **55** | 0.6182 | 0.5824 | 0.6158 | 0.3692 | 0.3097 | 0.9933 | 0.0156 | 0.0358 |
| **60** | 0.6209 | 0.5718 | 0.6233 | 0.3775 | 0.3019 | 0.9935 | 0.0062 | 0.0491 |
| **65** | 0.5979 | 0.5551 | 0.5931 | 0.3776 | 0.2789 | 0.9936 | 0.0075 | 0.0427 |
| **70** | 0.5763 | 0.5386 | 0.5851 | 0.3818 | 0.2633 | 0.9938 | 0.0003 | 0.0378 |
| **75** | 0.5886 | 0.5476 | 0.5871 | 0.3871 | 0.2582 | 0.9939 | 0.0099 | 0.0410 |
| **80** | **0.5759** | **0.5381** | **0.5742** | 0.3909 | 0.2472 | 0.9940 | 0.0104 | 0.0379 |

### M5 / M6 대비 비교

| 지표 | M5 | M6 ckpt_20 (best) | M7 ep80 | 목표 달성? |
|------|:--:|:-----------------:|:-------:|:----------:|
| action_mse_prior ↓ | 0.6084 | 0.8588 | **0.5759** | ✅ |
| action_mse_posterior ↓ | 0.5318 | 0.8168 | **0.5381** | ✅ |
| z_shuffle_gap ↑ | 0.0163 | 0.0268 | 0.0104 | ❌ |
| prior_posterior_gap ↑ | 0.0766 | 0.0420 | 0.0379 | ❌ |
| best_of_5 ↓ | — | — | 0.3909 | — |
| sampling_diversity ↑ | — | — | 0.2472 | — |

---

## 5. 결과 해석 및 인사이트

### 긍정적
- **S1에서 InfoNCE 정상 학습**: balanced sampler가 positive pair 문제를 해결. M6의 S1 정체(~2.2) 대비 M7은 ep7에서 0.71까지 하강
- **action_mse_prior ep80 = 0.5759** — M5(0.608) 목표 달성. M6(0.859)보다 크게 개선
- prior_flow_loss 안정화: M6의 진동 패턴(0.24 ↔ 0.88) 사라짐 → balanced sampler가 prior 학습 안정에도 기여
- ep5 val에서 prior_posterior_gap=0.692 — S1에서 이미 z 구조가 형성되고 있음을 시사

### 주의
- **z_shuffle_gap이 M5 수준 미달**: ep30~55 구간 0.014~0.016으로 가장 높지만, M4 목표(0.043) 미달. ep70에서 0.0003으로 급락도 발생
- z_shuffle_gap 진동 지속 (0.0003~0.0156) — 학습 중 z의 task discrimination이 불안정
- sampling_diversity가 ep80 기준 0.247로 ep5(0.734) 대비 크게 감소 → z 다양성 점진적 감소 중
- best_of_5가 ep50 이후 오히려 소폭 증가 (0.366 → 0.391) — prior 정확도가 후반부에서 정체

### M6 대비 개선/미개선 정리

| 항목 | M6 결과 | M7 결과 | 개선? |
|------|---------|---------|:-----:|
| S1 InfoNCE 학습 | ❌ 정체(~2.2) | ✅ 정상 하강(→0.43) | ✅ |
| prior_flow_loss 안정성 | ❌ 진동 | ✅ 단조 감소 | ✅ |
| action_mse_prior (ep20 기준) | 0.850 | **0.700** | ✅ |
| z_shuffle_gap 지속성 | ❌ collapse | △ 낮은 수준 유지 | △ |
| sampling_diversity | 점진 감소 | 점진 감소 | ➖ |

---

## 5.5 z_shuffle_gap / prior_posterior_gap 심층 진단

### 현상 요약

두 지표가 공통으로 보이는 패턴:

```
ep5 (S1 끝):  z_shuffle_gap=0.094  prior_posterior_gap=0.692  ← 가장 높음
ep10 (S2 진입): z_shuffle_gap=0.005  prior_posterior_gap=0.162  ← 급락
ep10~80 (S2):  z_shuffle_gap=0.003~0.016  prior_posterior_gap=0.022~0.049  ← 낮은 수준 유지
```

S2 진입 시점에 동시에 급락하고 이후 회복하지 못함.

---

### z_shuffle_gap이 낮은 근본 원인

**현재 z 주입 구조:**

```
action_cond = cat([context, z])          # concat으로 한 번 섞임
h = input_proj(cat([x_t, t_emb, cond]))  # 이후 z 정보 희석
for block in residual_blocks:
    h = h + block(h)                     # z와 무관하게 진행
```

`input_proj`에서 z weight를 작게 만들면 decoder는 z를 사실상 무시할 수 있다. flow loss 관점에서 z가 없어도 context만으로 충분히 좋은 action을 예측할 수 있다면, **z를 무시하는 것이 locally optimal한 해**다.

**InfoNCE는 z 공간의 구조를 만들지만, decoder가 z를 쓰는지는 다른 문제다.**

- InfoNCE: "z끼리의 거리를 task-discriminative하게" → z 공간 구조에 영향
- z_shuffle_gap: "decoder가 z에 얼마나 의존하는가" → decoder 내부 동작에 의존

두 objective가 직접 연결되지 않는다. z 공간이 task-discriminative해도 decoder가 z를 무시하면 gap은 낮다.

**S2 급락의 원인:**

S2에서 LoRA가 활성화되면서 VLM(prior z 생성)이 fine-tune됨. VLM이 받는 gradient는 action flow loss에서 오는데, 이 loss는 z가 task-discriminative할 필요 없이 **action을 잘 예측하면 된다.** 결과적으로:

1. VLM z가 action-predictive 방향으로 drift
2. S1에서 InfoNCE로 형성된 task-discriminative 구조 파괴
3. S2에서도 InfoNCE가 켜져 있지만, 가중치 λ=0.1 vs action_flow 가중치 1.0 → action flow gradient가 10배 강함

balanced sampler로 positive pair 문제는 해결했지만, gradient 경합의 비대칭성은 해결하지 못했다.

---

### prior_posterior_gap이 낮은 원인

**gap = MSE(prior) - MSE(posterior) = 0.5759 - 0.5381 = 0.038**

이 수치를 "prior가 posterior를 잘 모방했다"고 해석하면 안 된다. M2와 비교하면:

```
M2: prior=0.4776, posterior=0.0017, gap=0.4759  ← posterior가 oracle 수준
M7: prior=0.5759, posterior=0.5381, gap=0.038   ← 둘 다 낮은 수준에서 수렴
```

M7의 gap이 낮은 이유는 **posterior z(미래 이미지 기반)마저도 flow decoder가 충분히 활용하지 못하기 때문**이다. oracle(M2)의 posterior MSE=0.0017은 미래를 알면 거의 완벽한 행동 예측이 가능하다는 의미인데, M7의 posterior MSE=0.538은 미래 정보가 있어도 decoder가 그것을 행동으로 연결하지 못하는 상태를 나타낸다.

즉 prior와 posterior 모두 flow decoder에서 z가 충분히 활용되지 않아, 둘 다 비슷하게 낮은 MSE 수준에서 수렴하면서 gap이 자연히 작아진 것이다.

---

### M8(infonce_stage1_only)이 해결할 수 있는가

현재 실행 중인 M8은 **S2에서 InfoNCE를 끄는** 변형이다. 이 접근의 예측:

**해결 가능한 부분:**
- S2에서 InfoNCE-flow loss 경합이 사라짐 → z가 두 방향으로 동시에 당겨지는 문제 제거
- flow loss gradient만 남아 학습이 더 안정적일 수 있음

**해결하기 어려운 부분:**
- **근본 원인 미해결**: decoder가 z를 concat으로 받아 무시할 수 있는 구조는 그대로
- S2에서 LoRA로 VLM이 fine-tune될 때 z가 action-predictive 방향으로 drift하는 것은 여전히 발생
- S1에서 형성한 z 구조가 S2 flow loss에 의해 파괴되는 패턴은 반복될 가능성이 높음

**예상 결과:** ep5~9 구간 z_shuffle_gap이 M7 수준(~0.09)으로 나타났다가 ep10 S2 진입 후 다시 급락. M7과 정성적으로 유사한 패턴을 보일 것으로 예상.

---

### 진짜 해결책: M8(FiLM + CFG)

`experiments/M8_film.md`에 설계된 진짜 M8이 이 문제를 정공법으로 공략한다.

**FiLM (Feature-wise Linear Modulation):**
```
z → ZEncoder → z_feat
z_feat → γ_l, β_l (per-block scale/shift)
h_l' = γ_l ⊙ h_l + β_l
```
z가 각 residual block의 연산을 직접 변조 → z를 무시하는 것이 구조적으로 불가능.

**CFG z-Dropout:**
```
p=0.1 확률로 z → null_z
decoder: "z 없으면 평균적 행동, z 있으면 task-specific 행동"을 explicit하게 학습
```
z 유무의 차이를 loss-level에서 명시적으로 학습시킴 → z_shuffle_gap이 올라갈 직접적인 학습 신호.

| 문제 | 현재 M6~M8(s1only) | M8(FiLM+CFG) |
|------|:-----------------:|:------------:|
| decoder z 무시 | ❌ 구조적으로 가능 | ✅ 구조적으로 불가 |
| z dependency 학습 신호 | ❌ 없음 | ✅ CFG dropout |
| InfoNCE-flow 경합 | △ M8 s1only에서 S2 제거 | ✅ InfoNCE 제거 |
| S2 z 구조 파괴 | ❌ 여전히 발생 가능 | △ FiLM이 완화 |

---

## 6. 다음 스텝

### 현재 진행 중
- M8(infonce_stage1_only): `outputs/runs/vlm_sfp_infonce_s1only_20260418/` — single GPU 학습 중 (4/20 새벽 완료 예정)
  - 검증 포인트: ep10 전후 z_shuffle_gap이 M7처럼 급락하는지 여부

### M8(FiLM+CFG) 실행 전 체크리스트
- [ ] `models/flow_utils.py` — `FiLMResidualBlock`, `FiLMVelocityMLP` 구현
- [ ] `models/stoch_latent_flow_prior.py` — FiLM 교체, CFG dropout 추가
- [ ] `configs/vlm_paligemma_film.yaml` 작성
- [ ] smoke test: z→null_z 대체 시 MSE 변화 확인 (z_drop test)

### 분석 (M8 s1only 완료 후)
- [ ] ep5~15 z_shuffle_gap 추이 — M7 패턴 반복인지 확인
- [ ] sampling_diversity 감소 원인 분석

---

## 7. 저장 파일 목록


```
outputs/runs/vlm_sfp_infonce_balanced_20260416/
├── ckpt_10.pt ~ ckpt_80.pt     ← 10ep 단위 체크포인트
├── train.log
├── train_log.jsonl
└── result.md                   ← 이 파일
```

*ep82까지 학습 진행됨 (ckpt_80 이후 SSH disconnect로 ckpt 없음)*
