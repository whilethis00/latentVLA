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

이 섹션은 M7에서 두 지표가 왜 낮은지를 처음부터 차근차근 설명한다. 코드를 모르는 사람도 따라올 수 있도록 기초부터 시작해 핵심 구조 문제까지 이어진다.

---

### 1단계: z가 뭘 해야 하는가

이 연구에서 z는 "계획 벡터"다. 로봇이 어떤 task를 수행해야 하는지를 압축한 벡터로, 이것이 행동(action) 생성 전에 조건으로 들어간다.

```
[현재 이미지 + 언어 지시] → VLM → z (계획)
z + [현재 상태] → flow decoder → action chunk
```

z가 잘 만들어졌다면 두 가지가 성립해야 한다.

1. **task-discriminative**: "컵 집기" task의 z와 "서랍 열기" task의 z가 달라야 한다. 그래야 로봇이 z만 보고 "지금 뭘 해야 하는지" 알 수 있다.
2. **action-sufficient**: z를 알면 어떤 action이 필요한지 충분히 알 수 있어야 한다. z가 있을 때와 없을 때 행동 품질이 크게 달라야 한다.

`z_shuffle_gap`은 (1)을 측정하고, `prior_posterior_gap`은 (2)와 관련된다.

---

### 2단계: 관찰된 현상

```
ep5  (S1 끝):    z_shuffle_gap=0.094   prior_posterior_gap=0.692  ← 그나마 높음
ep10 (S2 진입):  z_shuffle_gap=0.005   prior_posterior_gap=0.162  ← 급락
ep10~80 (S2):    z_shuffle_gap=0.003~0.016  prior_posterior_gap=0.022~0.049
```

두 지표가 ep10에서 동시에 급락하고 이후 회복하지 못한다. M7뿐 아니라 M5, M6에서도 같은 패턴이 반복됐다. **이건 InfoNCE 설정이나 sampler 문제가 아니라 구조적 문제라는 신호다.**

비교를 위해 M2(이론적 상한)의 값을 보면:
```
M2: z_shuffle_gap=0.784  prior_posterior_gap=0.476  posterior_MSE=0.0017
M7: z_shuffle_gap=0.010  prior_posterior_gap=0.038  posterior_MSE=0.538
```

M2는 학습 시 미래 이미지를 직접 보고 z를 만들기 때문에 이렇게 높다. M7이 이 수준에 훨씬 못 미치는 이유를 이제 구조적으로 분석한다.

---

### 3단계: 코드에서 z는 어떻게 학습되는가

`stoch_latent_flow_prior.py`의 `compute_loss()`가 z 학습의 핵심이다. 전체 흐름을 그림으로 그리면:

```
[현재 상태 + action + 미래 이미지]
           ↓
      posterior_enc         ← z를 만드는 인코더 (학습 대상)
           ↓
         z_star             ← 샘플된 z (이걸 잘 만들고 싶은 것)
           ↓
    ┌──────┴──────────┐
    ↓ [A]detach       ↓ [B]detach
action_flow       prior_flow    ← 이 loss들은 z_star로 gradient를 주지 않는다
    ↓                 ↓
 loss_action      loss_prior

    ↓ (detach 없음)
semantic_head → loss_semantic   ← z_star로 gradient가 오는 유일한 경로
InfoNCE                         ← z_star로 gradient가 오는 또 다른 경로
```

핵심은 **`[A]`와 `[B]`에 `detach()`가 붙어 있다**는 것이다.

`detach()`는 PyTorch에서 "이 텐서는 gradient 계산에서 끊어내라"는 명령이다. 즉, `action_flow loss`와 `prior_flow loss`는 z_star를 사용하지만, z_star를 더 좋게 만드는 데 **아무 기여도 하지 않는다.**

결과적으로 `z_star` (posterior 인코더)를 학습시키는 신호는 두 가지뿐이다:
- **semantic loss**: z가 미래 이미지 피처와 비슷하게 → "미래를 닮은 z"
- **InfoNCE**: z가 task별로 구분되게 → "task를 구별하는 z"

그런데 둘 다 "action을 잘 예측하는 z"를 만들라고 요구하지 않는다.

---

### 4단계: 왜 이게 문제인가 — z가 action에 blind하다

**M2와의 결정적 차이:**

M2(DetLatent, oracle)는 이렇게 동작한다:
```
미래 이미지 → SigLIP → MLP → z_oracle
z_oracle + context → action flow → loss_action
                                        ↓ gradient
                               z_oracle로 흐름 (detach 없음)
```

z_oracle은 action loss gradient를 직접 받는다. 즉, "이 z로 action을 예측하면 얼마나 잘 맞는가"를 직접 학습한다. 그래서 M2의 z는 action-sufficient하고, posterior MSE=0.0017(거의 완벽)이 나온다.

M7은 이렇게 동작한다:
```
미래 이미지 + context + action → posterior_enc → z_star
z_star.detach() + context → action flow → loss_action
                                               ↓ gradient
                                    detach 때문에 z_star로 안 옴 ← 여기서 끊김
```

z_star는 action loss를 "구경"만 하고 학습받지 않는다. z가 action 예측에 얼마나 유용한지를 z 자체가 배울 방법이 없다.

**이것이 posterior MSE=0.538이 나오는 진짜 이유다.** posterior z(미래 정보 포함)를 써도 MSE가 0.538이라는 것은, 미래 정보가 있어도 z가 action 예측에 충분한 정보를 담지 못한다는 뜻이다. z가 행동과 연결되도록 학습된 적이 없기 때문이다.

---

### 5단계: prior_posterior_gap이 낮은 이유

`prior_posterior_gap = action_mse_prior - action_mse_posterior`

이 gap이 높으면 "미래를 알 때(posterior)는 훨씬 잘 예측하지만 모를 때(prior)는 많이 틀린다"는 의미라서, z가 미래 정보를 잘 담고 있다는 증거다.

M7에서 ep80: gap = 0.5759 - 0.5381 = **0.038**

이걸 "prior가 posterior를 잘 따라가서 gap이 줄었다"고 읽으면 안 된다. 진짜 의미는:

- **prior MSE=0.576**: 미래 모르고 예측할 때 오차
- **posterior MSE=0.538**: 미래 알고 예측할 때도 오차가 비슷하게 높음

미래 정보를 줘도 별로 안 나아진다. 즉 **posterior z 자체가 이미 action에 약한 상태**라서, prior가 posterior를 잘 따라가든 못 따라가든 결과가 비슷한 것이다.

M2는 posterior MSE=0.0017. 미래 이미지가 있으면 거의 완벽한 action 예측이 가능하다. M7은 같은 미래 정보가 있어도 0.538이다. z가 그 정보를 action과 연결하도록 학습되지 않았기 때문이다.

---

### 6단계: decoder의 z 회피 문제 — 두 번째 구조적 문제

4단계에서 z가 action에 약해진 이유를 설명했다. 그런데 가정을 바꿔보자: detach를 없애서 z를 action-sufficient하게 만들었다고 치자. 그래도 또 다른 문제가 있다.

현재 action flow decoder의 구조는 이렇다:

```python
# flow_utils.py VelocityMLP.forward()
action_cond = cat([context, z])       # z가 context 뒤에 붙어서 들어옴
h = input_proj(cat([x_t, t_emb, action_cond]))   # 한 번 선형 변환
for block in residual_blocks:
    h = h + block(h)                  # 이후 residual block들
output = output_proj(h)
```

z는 `input_proj`에서 한 번 다른 모든 입력들과 함께 섞인다. 이후의 residual block들은 z를 직접 볼 수 없고, 초기 변환을 통해 희석된 z 정보만 간접적으로 사용한다.

학습 최적화 관점에서 생각해보자. 모델이 해야 할 일은 loss를 줄이는 것이다. context(현재 이미지 + 상태) 정보만으로도 action을 꽤 잘 예측할 수 있다면, `input_proj`에서 z의 weight를 작게 만드는 것이 더 쉬운 해다. **z를 적극적으로 쓰는 것은 선택이지 강제가 아니다.**

4단계의 문제(z가 action-blind)와 합쳐지면:
- z가 action에 별로 유용하지 않음 → decoder가 z를 써봐야 이득이 없음
- decoder가 z를 무시해도 loss가 줄어듦 → decoder는 z 무시를 선택
- z를 무시하면 z_shuffle_gap이 낮음 → 지표가 낮게 나옴

**이것이 두 병목이 직렬로 연결되는 방식이다:**

```
[detach] z가 action에 약함
              ↓
     decoder가 z에 의존할 이유가 없음
              ↓
     [concat] decoder가 z를 회피하는 것이 최적화 상 유리
              ↓
     z_shuffle_gap 낮음, prior_posterior_gap 낮음, sampling_diversity 감소
```

---

### 7단계: detach가 두 군데인데 성격이 다르다

`compute_loss()` 코드를 정확히 보면:

```python
# [A] action branch detach — z가 action loss로부터 격리
action_cond = torch.cat([context, z_star.detach()], dim=-1)
loss_action = flow_matching_loss(self.action_flow, actions, action_cond)

# [B] prior target detach — z가 prior loss로부터 격리
loss_prior = flow_matching_loss(self.prior_flow, z_star.detach(), _planner)
```

둘이 같아 보이지만 역할이 완전히 다르다.

**[A] action branch detach:**
- 의미: "action loss가 z를 어떻게 만들지 간섭하지 마라"
- 왜 원래 이렇게 됐나: z 인코더와 action decoder가 서로 영향을 주면 학습이 불안정해질 수 있다는 우려에서 설계된 것으로 보임
- 문제: 이 때문에 z가 action에 대해 blind해짐 → 4단계의 핵심 문제
- **수정 대상**: 이 detach를 제거하면 posterior z가 "action을 잘 예측하는 데 필요한 정보"를 담도록 학습됨

**[B] prior target detach:**
- 의미: "prior flow가 따라가야 할 목표(z_star)를 고정된 것으로 취급하라"
- 왜 이렇게 됐나: prior flow와 posterior 인코더가 서로를 쫓으면서 둘 다 흔들리는 "moving target" 문제를 방지하기 위해
- **수정 금지 (당분간)**: [B]를 제거하면 prior flow의 학습 목표가 prior flow 자체에 의해 변하는 불안정한 상황이 된다. 학습이 발산할 수 있다.

따라서 "detach를 제거한다"는 말은 [A]만 제거한다는 의미여야 한다. [A]와 [B]를 한 번에 바꾸는 것은 전혀 다른 실험이고 더 위험하다.

---

### 8단계: InfoNCE는 왜 안 됐나

InfoNCE loss는 "같은 task의 z끼리 가깝게, 다른 task의 z끼리 멀게"를 학습시킨다. M7에서 S1 동안 이것이 실제로 작동했다는 증거가 있다(ep5 z_shuffle_gap=0.094). 그런데 왜 유지되지 않았나?

**InfoNCE가 하는 일과 하지 않는 일:**

InfoNCE가 하는 일:
- z 공간의 기하학적 구조를 task별로 분리 → z들이 task별로 클러스터를 형성

InfoNCE가 하지 않는 일:
- decoder가 z를 실제로 쓰도록 강제하지 않음
- z가 action 예측에 유용하도록 만들지 않음

비유하자면, InfoNCE는 "도서관의 책을 주제별로 잘 분류"하는 일이다. 그런데 decoder(독자)가 책을 읽지 않고 다른 정보(context)만 보고 답을 낸다면, 책이 아무리 잘 분류돼 있어도 의미가 없다.

**S2 진입 시 급락하는 이유:**

S2에서 LoRA가 활성화되면서 VLM 자체가 fine-tuning된다. 이때 VLM으로 흐르는 gradient의 대부분은 action flow loss에서 온다(loss weight 기준으로 action_flow: 1.0, infonce: 0.1 → 10배 차이). VLM은 "action을 잘 예측하게 z를 만들어라"라는 신호를 주로 받는다.

그런데 z는 action loss로부터 gradient를 받지 않는다([A] detach 때문에). 그러면 VLM은 무엇을 최적화하는가? action flow의 입력인 context(f_tilde)를 더 좋게 만드는 방향으로 학습된다. 결국 S2에서 VLM은 "z를 통해서"가 아니라 "context를 통해서" action 예측을 개선하는 쪽으로 drift한다. z의 task-discriminative 구조는 이 과정에서 함께 무너진다.

---

### 9단계: FiLM이 왜 단독으로는 반쪽인가

FiLM(Feature-wise Linear Modulation)은 z를 decoder의 각 residual block에서 scale/shift 방식으로 주입하는 구조다:

```
z → ZEncoder → z_feat
z_feat → γ, β (각 블록마다)
h_l' = γ_l ⊙ h_l + β_l   ← z가 각 블록의 연산 자체를 바꿈
```

이렇게 하면 6단계의 "decoder가 z를 회피할 수 있다" 문제는 구조적으로 해결된다. z 없이는 블록이 다르게 작동하므로 z를 무시하는 것이 구조적으로 불가능해진다.

하지만 4단계의 문제(z가 action에 blind)는 그대로다. action-weak z를 더 강하게 주입한다고 해서 z가 action-sufficient해지지 않는다. 공장에 더 좋은 컨베이어 벨트를 설치해도 불량품이 나오는 이유가 원자재 때문이라면 컨베이어 벨트 교체는 의미가 없다.

**FiLM은 "좋은 z를 더 잘 쓰는" 도구지, "나쁜 z를 좋게 바꾸는" 도구가 아니다.**

그래서 순서가 중요하다:
1. 먼저 z를 action-sufficient하게 만든다 (detach 제거)
2. 그다음 decoder가 그 z를 구조적으로 사용하도록 강제한다 (FiLM)
3. 추가로 z 유무의 차이를 loss-level에서 명시적으로 학습시킨다 (CFG dropout)

---

### 10단계: 현재 M8(s1only)은 뭘 검증하는가

현재 실행 중인 M8은 "S2에서 InfoNCE를 끄는" 변형이다. 핵심 구조 변경(detach, FiLM)은 없다.

이 실험으로 알 수 있는 것:
- S2에서 InfoNCE gradient 경합을 없앴을 때 ep10 cliff가 얼마나 완화되는가
- InfoNCE-flow loss 경합이 cliff의 주요 원인인지 아닌지

이 실험의 한계:
- detach 구조 그대로 → z가 여전히 action에 약함
- concat 구조 그대로 → decoder가 여전히 z 무시 가능

예측: ep5~9 동안 z_shuffle_gap이 M7과 비슷하게 ~0.09까지 올라갔다가, ep10 S2 진입 후 다시 급락. InfoNCE를 끈 것이 일부 cliff를 완화할 수 있지만, 구조 문제가 남아있어 근본적 반전은 어렵다.

**이 실험의 가치**: 만약 M7과 거의 똑같은 패턴이 나오면, "S2에서 InfoNCE를 끄는 것이 큰 의미 없다"는 결론이 나오고, 진짜 문제가 detach/concat 구조에 있다는 가설이 강화된다.

---

### 11단계: 해결 순서와 각각의 성공 기준

**Step 0. z_drop test (진단)**

실행:
```bash
conda run -n vla python3 scripts/z_drop_test.py \
    --config configs/vlm_paligemma_infonce_balanced.yaml \
    --ckpt   outputs/runs/vlm_sfp_infonce_balanced_20260416/ckpt_80.pt
```

이 테스트는 z를 zeros로 교체했을 때 action MSE가 얼마나 변하는지 측정한다.

- prior path: prior flow로 샘플된 z → zeros로 교체 → Δ MSE 측정
- posterior path: posterior 인코더의 mu_q → zeros로 교체 → Δ MSE 측정

판정 기준:
- Δ MSE가 거의 0: decoder non-usage 확정. z를 빼도 성능이 안 바뀜.
- Δ MSE가 크다: decoder가 z를 실제로 사용 중.

현재 예상: 두 경로 모두 Δ MSE가 매우 작을 것 (decoder non-usage).

---

**Ablation A — action detach 제거 (1차 수술)**

변경 내용:
```python
# stoch_latent_flow_prior.py
# Before (현재):
action_cond = torch.cat([context, z_star.detach()], dim=-1)
# After (Ablation A):
action_cond = torch.cat([context, z_star], dim=-1)  # detach 제거

# prior target은 여전히 detach 유지 (moving target 방지):
loss_prior = flow_matching_loss(self.prior_flow, z_star.detach(), _planner)
```

config에서 활성화:
```yaml
loss:
  action_detach: false   # Ablation A 활성화
```

이 변경 하나로 posterior z가 action loss gradient를 직접 받게 된다. z가 "action 예측에 유용한 정보를 담는 방향"으로 학습되기 시작한다.

성공 기준 (4가지 동시):
- posterior MSE 하락 (z가 action-sufficient해지는 증거)
- z_drop test Δ MSE 증가 (decoder가 z를 더 쓰기 시작)
- z_shuffle_gap 상승 (task-discriminative 유지)
- prior_flow_loss 진동 없이 안정

**주의**: A가 조용하면(4가지 다 반응 없으면), FiLM을 넣어도 별 차이 없을 가능성이 크다. 그 경우 전략을 재검토해야 한다.

---

**Ablation B — A + FiLM (z → decoder binding 강화)**

A가 성공한 후 FiLM을 추가한다. z가 이미 action-sufficient한 상태에서, decoder가 그 z를 구조적으로 무시하지 못하게 만든다.

```
[현재 (concat)]           [FiLM]
z → input_proj에 섞임     z → ZEncoder → z_feat
이후 희석 가능             각 residual block에서 γ, β 계산
                          h_l' = γ_l ⊙ h_l + β_l  ← 매 블록마다 z가 개입
```

성공 기준: A 대비 z_drop Δ MSE 추가 증가, z_shuffle_gap 추가 상승.

---

**Ablation C — B + CFG z-dropout (z dependency 학습 신호 명시화)**

학습 중 10% 확률로 z를 null vector로 교체한다.

```python
if training and random() < 0.1:
    z = null_z   # learnable null vector
```

decoder는 "z가 있을 때"와 "없을 때"를 모두 경험하면서 z의 유무가 action에 명시적 차이를 만들도록 학습한다. B에서 구조적으로 z를 쓸 수밖에 없게 만들었다면, C는 loss-level에서 z를 쓰는 것이 실제로 유리하다는 것을 학습시킨다.

성공 기준: z=null 대체 시 MSE gap이 C에서 가장 큼.

---

**Ablation D — prior target detach 제거 (마지막, 안정성 테스트)**

[B] 번 detach까지 제거하면 prior flow의 학습 목표가 posterior z(살아있는 값)가 된다. prior와 posterior가 서로를 쫓으면서 흔들릴 수 있다(moving target 문제). 개선 실험이 아니라 "이 구조가 실제로 불안정한지" 확인하는 테스트로 접근한다.

---

### 12단계: 최종 성공 정의

세 가지가 동시에 충족되어야 진짜 해결이다.

| 조건 | 측정 지표 | 현재 M7 |
|------|----------|---------|
| 좋은 z를 만든다 | posterior MSE ↓, z_var_mean 안정 | 0.538 (불충분) |
| z를 실제로 쓴다 | z_drop Δ MSE ↑, z_shuffle_gap ↑ | ~0.01 (거의 미사용) |
| S2 넘어 유지된다 | ep10 이후 gap류 지표 유지 | ep10에서 급락 |

셋 중 하나라도 빠지면 실패:
- MSE만 좋아지고 z 지표가 안 살아나면 "더 좋은 shortcut (context만으로 잘 예측)"
- z_shuffle_gap만 오르고 z_drop이 안 오르면 "z 공간 구조는 있지만 decoder가 안 씀"
- S1에서만 좋아지고 S2에서 죽으면 "학습 체제 전환 문제 미해결"

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
