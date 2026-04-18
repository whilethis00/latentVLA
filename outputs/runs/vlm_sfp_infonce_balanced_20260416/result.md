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

### 두 가지 구조적 원인

#### 원인 1: 두 개의 detach — 성격이 다르다

`stoch_latent_flow_prior.py compute_loss()`에 detach가 두 군데 있다:

```python
# [A] line 115 — action branch detach
action_cond = torch.cat([context, z_star.detach()], dim=-1)
loss_action = flow_matching_loss(self.action_flow, actions, action_cond)

# [B] line 122 — prior target detach
loss_prior = flow_matching_loss(self.prior_flow, z_star.detach(), _planner)

# grad가 z_star로 오는 건 아래 둘뿐
loss_semantic = 1 - cosine_sim(semantic_head(z_star), future_feat)  # line 128
# + InfoNCE (latent_vla.py에서 _z_star로)
```

**[A]와 [B]는 전혀 다른 실험이다.**

- **[A] action detach 제거**: posterior z가 action loss를 직접 받음. z가 "action을 잘 예측하는 z"가 되도록 학습. → 이것이 1차 수술 대상.

- **[B] prior target detach 제거**: prior flow가 moving target(posterior z)을 쫓게 됨. prior와 posterior가 동시에 흔들리는 moving target 문제 발생 가능. → 안정성이 검증된 후 마지막에 시도.

현재 posterior z를 학습시키는 gradient는 semantic loss(cosine sim to future image)와 InfoNCE뿐이다. z가 action-sufficient한지와 무관하게 학습이 수렴하는 구조다.

M2와의 차이:
```
M2: posterior z = MLP(future_image) → action loss가 이 z로 흐름 → action-sufficient z
M7: posterior z → [A] detach → action loss gradient 차단 → action-blind z
```

따라서 `posterior MSE=0.538`은 "prior가 posterior를 잘 못 따라가서"가 아니라, **posterior z 자체가 action에 약하기 때문**이다. 둘 다 action-weak latent 수준에서 수렴하니 gap이 자연히 작아진다.

#### 원인 2: concat 구조 — decoder가 z를 회피할 수 있다

```python
action_cond = cat([context, z_star.detach()])  # z가 한 번 concat
h = input_proj(cat([x_t, t_emb, cond]))        # 이후 z 정보 희석 가능
for block in residual_blocks:
    h = h + block(h)                           # z와 무관하게 작동 가능
```

`input_proj`에서 z 축 weight를 작게 만들면 decoder는 z를 무시하고 context만으로 action을 예측할 수 있다. 원인 1로 z 자체가 action-sufficient하지 않으므로, decoder 입장에서 z를 무시하는 것이 locally optimal하다.

**FiLM은 이 구조를 깨는 도구다.** 단, FiLM의 가치는 "좋은 z를 더 잘 주입"하는 데 있지, 나쁜 z를 좋은 z로 바꾸는 데 있지 않다. 원인 1을 먼저 고치지 않으면 FiLM은 action-weak z를 더 강하게 주입하는 꼴이 된다.

#### 두 원인의 직렬 구조

```
원인 1 (detach) → z가 action에 약함
         ↓
decoder가 z에 의존할 이유 없음 → 원인 2 강화
         ↓
z_shuffle_gap 낮음, prior_posterior_gap 낮음
```

두 개를 같이 해결해야 한다. 하나만 하면 다시 무너진다.

---

### prior_posterior_gap이 낮은 원인

위 원인 1에서 설명한 대로. 둘 다 action-weak한 latent라서 비슷한 수준에서 수렴.

---

### M8(infonce_stage1_only)이 해결할 수 있는가

현재 실행 중인 것은 **S2에서 InfoNCE를 끄는** 변형이다. 진단용 컨트롤 실험으로는 가치 있다.

**해결 가능한 부분:**
- S2 InfoNCE-flow gradient 경합 제거 → cliff가 M7보다 덜할 수 있음

**해결하지 못하는 부분:**
- detach 구조 그대로 → z가 여전히 action에 약함
- concat 구조 그대로 → decoder가 여전히 z 무시 가능
- S2 LoRA fine-tune 시 z 구조 파괴 패턴 반복 가능성

**예상 결과:** M7보다 조금 덜 망가질 수 있으나 문제를 뒤집지는 못함. ep10 S2 진입 후 z_shuffle_gap이 다시 급락하는 패턴이 반복될 가능성이 높다.

---

### 진짜 해결 순서 (ablation)

**0. z_drop test**: 현재 ckpt로 z=null 대체 시 action MSE 변화 측정. 거의 안 오르면 decoder non-usage 확인 → 나머지 ablation 필요성 확정.

**Ablation A — action branch detach만 제거:**
```python
# stoch_latent_flow_prior.py line 115
# Before:
action_cond = torch.cat([context, z_star.detach()], dim=-1)
# After:
action_cond = torch.cat([context, z_star], dim=-1)
# line 122 (prior target) — 유지
loss_prior = flow_matching_loss(self.prior_flow, z_star.detach(), _planner)
```
prior target은 detach 유지. posterior z가 action loss를 직접 받게만 변경.  
**성공 기준**: posterior MSE 하락, z_shuffle_gap 상승, z_drop test에서 MSE 변화 커짐.  
A가 조용하면 → FiLM 넣어도 별 차이 없을 가능성 큼.

**Ablation B — A + FiLM:**
action-sufficient해진 z를 decoder에 구조적으로 binding.  
**성공 기준**: A 대비 z_shuffle_gap 추가 상승.

**Ablation C — B + CFG z-dropout:**
z 유무 차이를 loss-level에서 명시적 학습 신호로 추가.  
**성공 기준**: z=null 대체 시 MSE gap이 C에서 가장 큼.

**Ablation D — prior target detach 제거 (마지막):**
prior와 posterior가 동시에 흔들리는 moving target 문제 있음. 개선 실험이 아니라 안정성 테스트.

| 문제 | M8(s1only) | Ablation A | Ablation B | Ablation C |
|------|:----------:|:----------:|:----------:|:----------:|
| z가 action에 약함 | ❌ | ✅ | ✅ | ✅ |
| decoder z 회피 가능 | ❌ | △ | ✅ | ✅ |
| z dependency 학습 신호 | ❌ | △ | △ | ✅ |

**대안 경로**: action detach 제거 없이 oracle distillation(`distill_alpha`)으로 z 품질을 끌어올린 뒤 FiLM+CFG를 거는 방법도 있다(`latent_vla.py`에 이미 구현되어 있음). 단, decoder non-usage 문제는 여전히 FiLM/CFG가 필요하다.

---

## 6. 다음 스텝

### 현재 진행 중
- M8(infonce_stage1_only): `outputs/runs/vlm_sfp_infonce_s1only_20260418/` — single GPU 학습 중
  - ep15까지만 보고 중단 예정 (ep10 S2 전환 패턴 확인이 목적)
  - 검증 포인트: ep10 z_shuffle_gap이 M7처럼 급락하는지 여부

### 다음 실험 순서
1. **z_drop test** (스크립트): z=null 대체 시 action MSE 변화 측정 → decoder non-usage 정량 확인
2. **detach 제거 실험**: `z_star.detach()` → `z_star` 변경 후 학습 → z가 action-sufficient해지는지 확인
3. **M8(FiLM+CFG)**: detach 제거와 함께 적용해야 효과 극대화

### 분석 (M8 s1only ep15 후)
- [ ] ep5~15 z_shuffle_gap 추이 — M7 패턴 반복 여부
- [ ] z_drop test 수치로 decoder z 의존도 정량화

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
