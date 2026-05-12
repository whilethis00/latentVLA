# PlanBind-VLA: 인과적으로 묶인 Latent Plan을 위한 VLA 연구

이 저장소는 Vision-Language-Action 정책에서 latent plan `z`가 실제 행동 생성을 통제하는지 진단하고, inference-time prior plan이 action decoder에 인과적으로 쓰이도록 학습하는 방향의 연구 코드입니다.

현재 결론은 단순히 "VLM-z가 MLP-z보다 좋다"가 아닙니다. 오히려 지금까지의 실험은 더 중요한 실패 모드를 보여줍니다.

> VLM을 붙인다고 deployable latent plan이 저절로 생기지 않는다.
> 행동 MSE는 좋아질 수 있지만, prior `z`가 action generation에 causal하게 쓰이지 않을 수 있다.

따라서 핵심 질문은 다음입니다.

**Can a Vision-Language-Action policy learn a deployable latent plan channel that is causally necessary for action generation?**

한국어로는:

**VLA가 현재 관찰과 언어로부터 앞으로의 행동 방식을 나타내는 latent plan을 만들고, 그 plan이 실제 action generation에 인과적으로 개입하게 만들 수 있는가?**

---

## 연구 방향

기존 VLA와 generative action policy는 이미지/언어/상태에서 행동 chunk를 잘 예측하는 데 집중합니다. 그러나 manipulation은 같은 장면과 지시에서도 접근 방향, grasp pose, subgoal ordering이 달라질 수 있는 다봉 문제입니다. 이때 latent plan은 자연스러운 중간 표현이지만, 좋은 latent처럼 보이는 것과 action decoder가 그 latent를 실제로 쓰는 것은 다릅니다.

이 저장소의 새 방향은 `diagnose -> intervene -> bind`입니다.

1. **Diagnose**: `z`가 task/future 정보를 담는지, prior/posterior 경로에서 decoder가 `z`에 민감한지 분리해서 측정합니다.
2. **Intervene**: shuffle/null/random/task-negative intervention으로 `z`를 바꿨을 때 action이 바뀌는지 확인합니다.
3. **Bind**: prior-action co-training, counterfactual binding loss, anti-collapse regularization으로 inference-time `z`를 action generation에 묶습니다.

---

## 현재까지의 핵심 발견

| 실험 | 핵심 결과 | 해석 |
|------|----------|------|
| M4 StochFlowPrior | epoch 100 `action_mse_prior=0.6540`, `z_shuffle_gap=0.0428` | MLP 기반 latent flow는 어느 정도 작동하지만, gap이 초기 기대만큼 크지는 않음 |
| M5 VLM plan token | `action_mse_prior=0.6084`, `future_cosine_sim=0.9945`, `z_shuffle_gap=0.0163` | VLM feature는 MSE를 낮출 수 있지만 action-controlling plan은 아님 |
| M6 InfoNCE | early gap은 올라갔으나 epoch 20 `z_shuffle_gap=0.0041` | contrastive만으로 causal usage collapse를 막지 못함 |
| M8/S1-only 진단 | epoch 90 `action_mse_prior=0.5120`, `z_shuffle_gap=0.0086` | MSE는 좋아져도 inference-time prior `z` 경로는 약함 |
| S2-also InfoNCE | epoch 100 `prior_posterior_gap=0.0002`, `z_shuffle_gap=0.0046` | prior/posterior가 비슷해 보여도 `z`가 행동에 필요하다는 뜻은 아님 |

가장 중요한 논문 문장은 다음입니다.

> Modern VLA policies can achieve reasonable action prediction while bypassing the latent planning channel.

---

## 제안 방법: PlanBind-VLA

현재 구조를 버리지 않고 확장합니다.

```text
Current context:
  c = VLM(o_t, language, proprio)

Training-only posterior:
  z_q ~ q_phi(z | c, action_chunk, future_obs)

Deployable prior:
  z_p ~ p_psi(z | c)

Action decoder:
  a ~ pi_theta(a | c, z)
```

핵심은 "VLM에서 plan token 하나 뽑기"가 아니라, `z`가 action decoder에 causal하게 묶이도록 학습시키는 것입니다.

### Loss 구성

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

각 loss는 특정 failure mode를 막기 위한 장치입니다.

| Failure mode | 관측 지표 | 대응 |
|--------------|----------|------|
| posterior `z` collapse | `z_mu_var`, `z_sample_var` 낮음 | variance/spectral spread |
| posterior `z`가 future/task를 못 나눔 | `probe_ratio` 약함 | futureNCE, hard negatives |
| prior `z`를 decoder가 못 씀 | `prior_z_shuffle_gap` 낮음 | prior-action co-training |
| decoder가 `z` 없이 shortcut 사용 | MSE는 낮지만 causal gap 낮음 | counterfactual binding |
| naive VLM plan token 실패 | 높은 cosine, 낮은 gap | learned causal plan channel |

---

## 실험 로드맵

한 번에 모든 loss를 넣지 않고 failure mode별로 검증합니다.

| 단계 | 이름 | 목적 |
|------|------|------|
| M9 | VLM-SFP + Prior-Action Co-training | decoder를 prior `z`에도 노출 |
| M10 | M9 + Counterfactual Binding Loss | 잘못된 `z`를 넣으면 action loss가 올라가게 학습 |
| M11 | M10 + Anti-Collapse Spread | posterior `z` collapse 방지 |
| M12 | M11 + FutureNCE | cosine future prediction의 쉬운 해법 제거 |

최종 평가는 offline MSE가 아니라 rollout success가 중심입니다.

```text
x-axis: causal z metric
y-axis: LIBERO rollout success rate

FlatFlow, VAE, SFP, Naive VLA, InfoNCE, PlanBind-VLA
```

---

## 프로젝트 구조

```text
VLA/
├── configs/                    # 실험 config
├── data/                       # Robomimic / LIBERO dataset loader
├── models/
│   ├── encoders.py             # SigLIP/state context encoder
│   ├── flat_flow.py            # M1 direct action flow
│   ├── det_latent.py           # M2 deterministic latent
│   ├── stoch_latent_vae.py     # M3 VAE latent
│   ├── stoch_latent_flow_prior.py
│   ├── system2_vlm.py          # PaliGemma 기반 System 2
│   └── latent_vla.py           # System 2 + System 1 통합
├── training/
│   ├── builder.py              # dataset/model factory
│   ├── config_utils.py         # config/override/seed helper
│   ├── trainer.py              # non-VLM trainer
│   └── trainer_vlm.py          # VLM trainer + causal z eval metrics
├── evaluation/
│   └── metrics.py              # offline evaluator
├── scripts/
│   ├── train.py
│   ├── train_vlm.py
│   ├── evaluate_offline.py
│   ├── evaluate_offline_vlm.py
│   ├── eval_z_diag.py          # posterior z spread/drop/probe 진단
│   └── smoke_test*.py
├── outputs/runs/               # 보존된 실험 로그와 결과
└── docs/
    └── causal_latent_plan_direction.md
```

---

## 주요 지표

기존 `z_shuffle_gap` 하나로는 부족합니다. prior와 posterior 경로를 반드시 분리합니다.

| 지표 | 의미 |
|------|------|
| `action_mse_prior` | deployable prior `z_p`로 생성한 행동 MSE |
| `action_mse_posterior` | training-only posterior `z_q`로 생성한 행동 MSE |
| `prior_posterior_gap` | prior/posterior 성능 차이 |
| `prior_z_shuffle_gap` | prior `z_p`를 섞었을 때 MSE 증가량 |
| `posterior_z_shuffle_gap` | posterior `z_q`를 섞었을 때 MSE 증가량 |
| `delta_null_prior` | prior `z_p`를 0으로 제거했을 때 MSE 증가량 |
| `delta_null_posterior` | posterior `z_q`를 0으로 제거했을 때 MSE 증가량 |
| `z_mu_var` | posterior mean의 batch variance |
| `z_sample_var` | posterior sample의 batch variance |
| `probe_ratio` | same-task future pair와 random pair의 posterior 거리 비율 |
| `best_of_K` | stochastic action sampling의 다양성 효용 |

---

## 빠른 시작

```bash
pip install -r requirements.txt
```

기본 smoke test:

```bash
python scripts/smoke_test.py
python scripts/smoke_test_vlm.py
```

일반 모델 학습:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --override model.type=stoch_flow_prior \
             data.dataset_type=robomimic \
             data.dataset_path=/path/to/dataset.hdf5
```

VLM 모델 학습:

```bash
python scripts/train_vlm.py \
  --config configs/vlm_paligemma_infonce.yaml \
  --override training.output_dir=outputs/runs/m9_prior_action_cotrain
```

M9 content binding:

```bash
bash scripts/run_m9_content_binding.sh
```

새 방향의 VLM trainer는 causal-z metric을 기본적으로 매 epoch 평가합니다. 비용을 줄이고 싶으면 config override로 조절합니다.

```bash
python scripts/train_vlm.py \
  --config configs/vlm_paligemma_infonce.yaml \
  --override training.causal_eval_every=5
```

z-space 진단:

```bash
python scripts/eval_z_diag.py \
  --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
  --max_batches 30
```

prior/posterior causal intervention 분리 평가:

```bash
python scripts/eval_causal_z.py \
  --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
  --mode both \
  --intervention shuffle \
  --max_batches 30
```

---

## 논문 포지셔닝

우리는 future-aware posterior branch 자체를 처음 제안한다고 주장하지 않습니다. latent plan/action, future-aware posterior, anti-collapse regularization은 이미 가까운 연구 흐름이 있습니다.

차별점은 다음입니다.

> 기존 연구는 latent action/reasoning을 학습하지만, 그 latent가 action generation에 실제로 causal하게 쓰이는지를 중심 진단 지표와 학습 목표로 삼지 않는다. 우리는 VLA의 latent plan channel을 diagnose, intervene, bind하는 방법을 제안한다.

한 문장 정의:

> 우리는 VLA에서 latent plan이 왜 action을 실제로 통제하지 못하는지 진단하고, inference-time prior plan이 action generation에 인과적으로 묶이도록 학습하는 방법을 제안한다.

---

## 현재 환경 메모

이 저장소의 결과 로그는 `outputs/runs/**/train_log.jsonl` 및 `result.md`로 보존합니다. 체크포인트 `.pt` 파일은 용량이 커서 git 추적 대상이 아닙니다.
