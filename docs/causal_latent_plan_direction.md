# Causal Latent Plan 방향 전환 종합 정리

## 1. 결론부터

현재 실험을 그대로 "VLM-z가 MLP-z보다 좋다"로 밀면 위험하다. 지금 데이터가 가장 강하게 말하는 사실은 다음이다.

> VLM을 붙인다고 latent plan이 저절로 생기지 않는다.
> 행동 MSE는 좋아질 수 있지만, inference-time prior `z`가 action generation에 causal하게 쓰이지 않을 수 있다.

이 실패는 버릴 결과가 아니라 논문의 핵심 문제다.

**연구 질문**

> Can a Vision-Language-Action policy learn a deployable latent plan channel that is causally necessary for action generation?

한국어로:

> VLA가 현재 관찰과 언어로부터 앞으로 어떻게 행동할지를 나타내는 latent plan을 만들고, 그 plan이 실제 action generation에 인과적으로 개입하게 만들 수 있는가?

---

## 2. 지금까지의 결과에서 살릴 것

### 2.1 전체 실험 요약

| Run | Epoch | prior MSE | posterior MSE | pp gap | best-of-5 | future cosine | z gap | 핵심 해석 |
|-----|------:|----------:|--------------:|-------:|----------:|--------------:|------:|-----------|
| `flat_flow_100ep_20260404` | 100 | 0.5530 | - | - | - | - | - | direct flow baseline. latent channel 없음 |
| `det_latent_100ep_20260405` | 100 | 0.4776 | 0.0017 | 0.4759 | - | - | 0.7837 | posterior/oracle 성격이 너무 강해 deployable prior와 괴리 큼 |
| `stoch_vae_100ep_20260405` | 100 | 0.6321 | 0.6205 | 0.0116 | 0.3180 | 0.9650 | -0.0039 | VAE식 stochastic latent는 collapse/non-usage 위험 |
| `sfp_100ep_20260401` | 100 | 0.6540 | 0.5395 | 0.1144 | 0.3638 | 0.9965 | 0.0428 | latent flow prior는 작동하지만 causal gap은 충분히 크지 않음 |
| `vlm_sfp_plan_100ep_20260405` | 100 | 0.6084 | 0.5318 | 0.0766 | 0.3645 | 0.9945 | 0.0163 | naive VLM plan token은 action-controlling plan이 아님 |
| `vlm_sfp_infonce_20260414` | 20 | 0.8492 | 0.8208 | 0.0284 | 0.4330 | 0.9766 | 0.0041 | InfoNCE 효과가 유지되지 않고 collapse |
| `vlm_sfp_infonce_balanced_20260416` | 100 | 0.5694 | 0.5252 | 0.0442 | 0.3946 | 0.9941 | 0.0120 | balanced sampler도 causal usage를 충분히 만들지 못함 |
| `vlm_sfp_infonce_s1only_20260418` | 90 | 0.5120 | 0.4606 | 0.0514 | 0.4257 | 0.9971 | 0.0086 | MSE는 좋지만 prior `z` intervention 민감도 낮음 |
| `vlm_sfp_infonce_s2also_20260421` | 100 | 0.6540 | 0.6538 | 0.0002 | 0.3360 | 0.9874 | 0.0046 | prior/posterior gap이 작아도 latent가 causal하다는 뜻은 아님 |

### 2.2 가장 중요한 관찰

1. VLM plan token을 쓰면 action MSE가 개선될 수 있다.
2. 하지만 `future_cosine_sim`이 0.99 근처여도 `z_shuffle_gap`은 낮을 수 있다.
3. InfoNCE는 초반에 효과가 있어 보여도 후반 collapse를 막지 못했다.
4. M8 진단 관점에서는 posterior path는 살아 있을 수 있지만, inference-time prior path가 decoder에 제대로 묶이지 않는 문제가 핵심이다.

따라서 다음 문장이 현재 결과를 가장 잘 설명한다.

> Modern VLA policies can achieve reasonable action prediction while bypassing the latent planning channel.

---

## 3. 기존 연구와의 차별점

우리는 "future-aware latent branch"를 처음 제안한다고 주장하면 안 된다. latent plan/action, future-aware posterior, VLM 기반 latent reasoning, anti-collapse regularization은 이미 가까운 흐름이 있다.

차별점은 다음으로 잡는다.

> 기존 연구는 latent action/latent reasoning을 학습하지만, 그 latent가 action generation에 실제로 causal하게 쓰이는지를 중심 진단 지표와 학습 목표로 삼지 않는다. 우리는 VLA의 latent plan channel을 diagnose -> intervene -> bind하는 방법을 제안한다.

관련 흐름:

| 흐름 | 예시 | 부족한 점 |
|------|------|-----------|
| Large-scale direct VLA | OpenVLA, Octo, pi0 | observation/language -> action 성능 중심. latent intervention 중심 아님 |
| Generative action policy | Diffusion Policy, ACT | multimodal action modeling 중심. deployable plan channel 검증은 별도 문제 |
| Latent plan/action | Play-LMP, LAPA, LatBot, Being-H0.7 | latent 학습 자체는 가깝지만 causal usage를 중심 metric/objective로 삼는 포지션은 약함 |

---

## 4. Research Questions

**RQ1. Do naive VLA latent plans become causally useful action plans?**

현재 답은 No에 가깝다. PaliGemma plan token은 MSE를 낮출 수 있지만 `z_shuffle_gap`이 낮다.

**RQ2. Why does the latent plan channel fail?**

단순히 decoder가 모든 `z`를 무시하는 문제가 아니다. posterior path와 prior path를 분리해야 한다. posterior `z`는 decoder가 쓰지만, prior `z`가 posterior space와 mismatch되어 inference-time에 약해질 수 있다.

**RQ3. Can we train a latent plan channel that is interventionally bound to action generation?**

핵심은 counterfactual `z` intervention을 학습 objective 안에 넣는 것이다.

**RQ4. Does causal latent-plan quality predict and improve rollout success?**

ICRA급 주장에는 LIBERO rollout success와 causal z metric의 상관/개선이 필요하다.

---

## 5. 제안 방법: PlanBind-VLA

### 5.1 Architecture

```text
c = VLM(o_t, language, proprio)

z_q ~ q_phi(z | c, action_chunk, future_obs)   # training-only posterior
z_p ~ p_psi(z | c)                             # deployable prior

a ~ pi_theta(a | c, z)
```

현재 `System2VLM + StochLatentFlowPrior` 구조를 유지하되, 학습 목표를 바꾼다.

### 5.2 Loss

```text
L =
  L_post_action
+ lambda_prior L_prior
+ lambda_mix   L_mix_action
+ lambda_cf    L_counterfactual_binding
+ lambda_var   L_variance_spread
+ lambda_spec  L_spectral_diversity
+ lambda_fut   L_futureNCE
```

**Posterior action loss**

```text
L_post_action = FlowLoss(a | c, z_q)
```

**Prior flow matching**

```text
L_prior = FlowMatch(z_q | c)
```

**Prior-action co-training**

```text
z_mix = z_q with probability 1 - rho
      = z_p with probability rho

L_mix_action = FlowLoss(a | c, z_mix)
```

권장 schedule:

```text
epoch 0~10:  rho = 0.0
epoch 10~30: rho = 0.25
epoch 30~:   rho = 0.5
```

**Counterfactual binding loss**

```text
z_neg = shuffle(z_q) or hard_negative_z
L_cf = max(0, margin + Loss(a | c, z_q) - Loss(a | c, z_neg))
```

해석:

> 같은 context `c`에 잘못된 plan `z`를 넣으면 행동 예측이 나빠져야 한다.

**Anti-collapse spread**

```text
L_var = max(0, gamma - Var_B(mu_q))
```

필요하면 spectral diversity도 추가한다. 단, 이것은 novelty가 아니라 안정화 장치로 위치시킨다.

**FutureNCE**

```text
L_futureNCE = InfoNCE(pred_future(z), true_future_embedding, batch_negatives)
```

단순 cosine future prediction은 너무 쉬운 보조 과제일 수 있다.

---

## 6. 지표 개편

기존 `z_shuffle_gap` 하나로는 부족하다. prior와 posterior 경로를 분리한다.

| 지표 | 목적 |
|------|------|
| `prior_z_shuffle_gap` | inference-time prior `z`가 action decoder에 쓰이는지 |
| `posterior_z_shuffle_gap` | training-time posterior `z` 경로가 살아 있는지 |
| `delta_null_prior` | prior `z` 제거 민감도 |
| `delta_null_posterior` | posterior `z` 제거 민감도 |
| `z_mu_norm` | posterior mean norm |
| `z_mu_var` | posterior mean spread |
| `z_var_mean` | posterior variance 평균 |
| `z_sample_var` | sampled posterior spread |
| `probe_ratio` | future/task separation |
| `prior_posterior_mmd` | prior/posterior distribution mismatch |

이번 리팩토링에서 VLM evaluator는 다음 metric을 새로 로깅하도록 수정했다.

```text
prior_z_shuffle_gap
posterior_z_shuffle_gap
delta_null_prior
delta_null_posterior
z_mu_norm
z_mu_var
z_var_mean
z_sample_var
probe_ratio
```

VLM trainer는 `training.causal_eval_every`를 기준으로 causal metric을 평가한다. 기본값은 `1`이라 다음 run부터 매 epoch 기록된다. GPU 시간이 부담되면 명시적으로 `training.causal_eval_every=5`처럼 낮출 수 있지만, 논문용 run은 매 epoch 로그를 권장한다.

별도 평가 스크립트는 다음 인터페이스를 사용한다.

```bash
python scripts/eval_causal_z.py \
  --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
  --mode both \
  --intervention shuffle \
  --max_batches 30
```

지원 intervention:

```text
shuffle
null
random
task_negative
motion_negative
```

---

## 7. 다음 실험 순서

| 단계 | 이름 | 넣을 것 | 확인할 failure mode |
|------|------|---------|---------------------|
| M9 | VLM-SFP + Prior-Action Co-training | `z_mix` schedule | prior `z` mismatch |
| M10 | M9 + Counterfactual Binding | shuffle/hard-negative `z` margin | decoder bypass |
| M11 | M10 + Anti-Collapse Spread | variance/spectral regularizer | posterior collapse |
| M12 | M11 + FutureNCE | batch future contrastive | cosine auxiliary가 너무 쉬운 문제 |

주의: 한 번에 다 넣지 않는다. 논문에서는 어떤 loss가 어떤 failure mode를 해결했는지가 보여야 한다.

---

## 8. 논문 contribution 초안

**Contribution 1 — Diagnosis**

Naive VLM-derived latent plans are not necessarily causally used by the action generator.

**Contribution 2 — Method**

We propose a counterfactually bound latent plan channel for VLA policies.

**Contribution 3 — Evidence**

Causal latent-plan metrics predict and improve rollout success.

---

## 9. 논문 한 문장

영어:

> We study why latent plans in Vision-Language-Action policies fail to become causally effective action plans, and propose a counterfactually bound latent plan channel that makes inference-time plans both predictable from current context and necessary for action generation.

한국어:

> 우리는 VLA에서 latent plan이 왜 action을 실제로 통제하지 못하는지 진단하고, inference-time prior plan이 action generation에 인과적으로 묶이도록 학습하는 방법을 제안한다.
