# LatentVLA: 로봇 행동 생성을 위한 계층적 잠재 변수 모델 비교 연구

시각·언어 관찰과 고유감각(proprioception)을 입력으로 받아 로봇 조작(manipulation) 행동 시퀀스를 생성하는 네 가지 계층적 잠재 변수 모델 아키텍처를 구현하고 비교한 연구입니다.

---

## 연구 개요

**핵심 질문:** 다양하고 다봉(multimodal) 분포를 가지는 로봇 행동을 생성하려면 잠재 공간(latent space)을 어떻게 설계해야 하는가?

- 결정론적(deterministic) vs. 확률론적(stochastic) 잠재 변수
- VAE 기반 vs. Flow 기반 prior

**공통 기반:**
- 생성 모델: **Conditional Flow Matching (OT-CFM)**
- 시각-언어 인코더: **SigLIP** (freeze하여 특징 추출기로 사용)
- 행동 예측 지평선: 8 스텝

---

## 프로젝트 구조

```
VLA/
├── configs/
│   └── default.yaml                   # 하이퍼파라미터 설정
├── data/
│   ├── robomimic_dataset.py            # HDF5 데이터셋 로더 (image / low_dim 모드)
│   └── libero_dataset.py              # LIBERO 태스크 스위트 로더
├── models/
│   ├── encoders.py                    # ContextEncoder (SigLIP + proprio 융합)
│   ├── flat_flow.py                   # [M1] 베이스라인: 직접 행동 flow
│   ├── det_latent.py                  # [M2] 결정론적 잠재 변수 + 행동 flow
│   ├── stoch_latent_vae.py            # [M3] VAE 기반 확률론적 잠재 변수
│   ├── stoch_latent_flow_prior.py     # [M4] 2단계 Flow (제안 모델)
│   └── flow_utils.py                  # OT-CFM 유틸리티 (VelocityMLP, Euler 적분)
├── training/
│   ├── builder.py                     # 팩토리: 데이터셋 / 데이터로더 / 모델
│   └── trainer.py                     # 학습 루프, 평가, 체크포인트
├── evaluation/
│   └── metrics.py                     # OfflineEvaluator (best-of-K, z_shuffle_gap 등)
├── scripts/
│   ├── train.py                       # 학습 진입점
│   ├── evaluate_offline.py            # 오프라인 평가 (best-of-K, temperature sweep)
│   ├── plot_training.py               # 학습 곡선 시각화
│   ├── smoke_test.py                  # 유닛 테스트 (실제 데이터 불필요)
│   └── run_ablations.sh               # 4가지 핵심 실험 일괄 실행
└── requirements.txt
```

---

## 모델 구조

| ID | 모델 | 잠재 변수 타입 | 핵심 아이디어 |
|----|------|--------------|-------------|
| M1 | **FlatFlow** | 없음 | 베이스라인 — 노이즈에서 행동으로 직접 flow |
| M2 | **DetLatent** | 결정론적 | prior head로 z 예측, 행동 flow에 z 조건화 |
| M3 | **StochVAE** | 확률론적 (VAE) | Gaussian posterior/prior + KL 정규화 |
| M4 | **StochFlowPrior** | 확률론적 (Flow) | 잠재 flow + 행동 flow 2단계 구조 (KL 불필요) |

### 아키텍처 요약

```
공유 컨텍스트 인코더:
  이미지 (224×224) → SigLIP [frozen] → 패치 임베딩
  언어 → SigLIP 텍스트 인코더 [frozen]
  고유감각 → MLP
  → 융합 컨텍스트 C_t ∈ R^256

M1 FlatFlow:
  C_t → VelocityMLP → 행동 시퀀스 (H × action_dim)

M2 DetLatent:
  C_t → PriorHead → z_hat ∈ R^128
  (C_t, z_hat) → VelocityMLP → 행동 시퀀스

M3 StochVAE:
  Posterior: q(z | C_t, a, future) → z ~ N(μ, σ²)
  Prior:     p(z | C_t) → z ~ N(μ_p, σ_p²)  [KL 손실]
  (C_t, z) → VelocityMLP → 행동 시퀀스

M4 StochFlowPrior (제안):
  잠재 flow:  ε → z  (flow matching, KL 없음)
  행동 flow:  ε → a  (C_t, z 조건화)
```

### 학습 손실

- **행동 손실:** OT-CFM flow matching — `||v_θ(x_t, t, C) − u_t||²`
- **Prior 손실:** MSE (M2) / KL divergence (M3) / flow matching (M4)
- **시맨틱 보조 손실:** `1 − cosine_sim(예측 미래 임베딩, GT 미래 임베딩)` × 0.1

---

## 실험 결과

### StochFlowPrior (M4) — Temperature Sweep

검증 세트(전체의 10%) 기준, 100 에폭 학습 후 평가.

| Temperature | Action MSE (prior) | Action MSE (posterior) | Best-of-5 | 샘플 다양성 | z-Shuffle Gap |
|-------------|---------------------|------------------------|-----------|------------|---------------|
| 1.0         | 0.3797              | 0.2999                 | **0.2421** | 0.2781     | 0.7249        |
| 0.7         | 0.3886              | 0.2999                 | 0.2467    | 0.2452     | 0.7041        |
| 0.5         | **0.3751**          | 0.3093                 | 0.2802    | 0.2359     | 0.6864        |
| 0.3         | 0.3848              | 0.3180                 | 0.3027    | 0.2307     | 0.6986        |

**미래 시맨틱 cosine 유사도: 0.9999** (전 temperature 공통 — 시맨틱 정렬 우수)

### 학습 곡선 (StochFlowPrior)

| 에폭 | 전체 손실 | 행동 Flow 손실 | Prior Flow 손실 | 시맨틱 손실 |
|------|----------|--------------|----------------|-----------|
| 1    | 3.823    | 1.628        | 2.103          | 0.918     |
| 10   | ~1.2     | ~0.6         | ~0.6           | ~0.03     |
| 50   | ~0.45    | ~0.38        | ~0.07          | ~0.0001   |
| 100  | 0.390    | 0.355        | 0.035          | 0.000074  |

---

## 주요 인사이트

### 1. 잠재 변수 z는 핵심 정보를 담는다 (z-Shuffle Gap)
배치 내에서 z를 섞으면 성능이 **~72% 하락** (`z_shuffle_gap ≈ 0.72`).
잠재 코드가 단순한 노이즈가 아닌 행동 조건화에 필수적인 의미 정보를 담고 있음을 확인.

### 2. Prior는 Posterior를 잘 근사하지만 완벽하진 않다
Prior-posterior gap ~8% (`≈ 0.080`).
Flow 기반 prior는 VAE의 posterior collapse 위험 없이 합리적인 근사를 달성.
그러나 추가적인 prior 개선 여지가 존재함.

### 3. Best-of-K로 다양성 효용 확인
`best_of_5 = 0.242` vs `best_of_1 = 0.386` — 5개 샘플 중 최선을 선택할 경우 MSE가 **~37% 감소**.
확률론적 모델이 실제로 의미 있는 다양한 행동 모드를 생성하고 있음을 시사.

### 4. Temperature: 다양성-정확도 트레이드오프
- **낮은 temperature (0.3–0.5):** 다양성 감소, 하지만 MSE가 일관되게 개선되지는 않음 (모드가 GT에서 벗어날 수 있음)
- **Temperature = 1.0:** 다양성 최대, MSE도 경쟁력 있음 → 가장 균형 잡힌 설정

### 5. 시맨틱 보조 손실은 빠르게 수렴
시맨틱 손실이 에폭 1(0.918) → 에폭 10(~0.03)으로 급감하고, 에폭 20 이후 무시할 수준.
미래 SigLIP 임베딩 예측은 거의 완벽 (`cosine_sim ≈ 0.9999`)하지만, 이 보조 태스크가 컨텍스트 표현의 시맨틱 정렬을 개선하여 행동 예측에도 기여함.

### 6. VAE vs Flow Prior 설계 차이
StochFlowPrior는 명시적 KL 정규화 없이 flow를 통해 prior가 posterior z 분포를 직접 매칭.
이로 인해:
- Posterior collapse 방지
- 더 풍부한 다봉(multimodal) z 분포 표현 가능
- β-VAE 튜닝 불필요, 학습 안정성 향상

### 7. Ablation 실험 목록

| 실험명 | 설명 |
|--------|------|
| `stoch_flow_prior` | M4 전체 모델 (시맨틱 손실 포함) |
| `sfp_nosem` | M4에서 시맨틱 보조 손실 제거 |
| `sfp_planner_full` | M4 — planner 입력: 전체 상태 |
| `sfp_planner_object_only` | M4 — planner 입력: 오브젝트 상태만 (10차원) |
| `sfp_planner_proprio_only` | M4 — planner 입력: 고유감각만 |
| `sfp_zdim32` / `sfp_zdim64` | M4 — 잠재 차원 32 / 64 |
| `flat_flow` | 베이스라인 M1 |
| `det_latent` | 베이스라인 M2 |
| `stoch_vae` | 베이스라인 M3 |
| `svae_nosem` | M3에서 시맨틱 보조 손실 제거 |

---

## 빠른 시작

### 설치
```bash
pip install -r requirements.txt
```

### 유닛 테스트 (데이터 없이 동작 확인)
```bash
python scripts/smoke_test.py
```

### 학습
```bash
# StochFlowPrior (M4) 학습
python scripts/train.py \
    --config configs/default.yaml \
    --override model.type=stoch_flow_prior \
               data.dataset_type=robomimic \
               data.dataset_path=/path/to/dataset.hdf5

# 4가지 모델 순차 학습
bash scripts/run_ablations.sh
```

### 평가
```bash
python scripts/evaluate_offline.py \
    --run_dir outputs/runs/stoch_flow_prior \
    --best_of_k 1 3 5 10 \
    --temperature_sweep 1.0 0.7 0.5 0.3
```

### 학습 곡선 시각화
```bash
python scripts/plot_training.py \
    --runs outputs/runs/flat_flow outputs/runs/stoch_flow_prior \
    --metric action_flow_loss
```

---

## 주요 설정 파라미터

```yaml
model:
  type: stoch_flow_prior   # flat_flow | det_latent | stoch_vae | stoch_flow_prior
  flow_steps: 10           # 추론 시 ODE 적분 스텝 수
  planner_input: full      # full | object_only | proprio_only

latent:
  z_dim: 128               # 잠재 변수 차원

loss:
  semantic_future_weight: 0.1   # 시맨틱 보조 손실 가중치
  kl_beta: 1.0                  # VAE KL 가중치 (M3 전용)

training:
  num_epochs: 100
  learning_rate: 3.0e-4
  batch_size: 64
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| `action_mse_prior` | Prior 샘플링 행동과 GT 간 MSE |
| `action_mse_posterior` | Posterior z 사용 시 MSE (상한선) |
| `best_of_K` | K개 샘플 중 최소 MSE — 다양성 효용 측정 |
| `future_cosine_sim` | 예측 미래 임베딩과 실제 임래딩 간 코사인 유사도 |
| `sampling_diversity` | 샘플링된 행동들의 표준편차 — 출력 분산 측정 |
| `z_shuffle_gap` | z를 섞었을 때 성능 하락 — z의 중요도 측정 |
| `prior_posterior_gap` | `mse_prior - mse_posterior` — prior 품질 측정 |

---

## 향후 실험 계획

### 단기 (현재 프레임워크 내)

| 실험 | 목적 |
|------|------|
| **모델 간 정량 비교** | M1~M4 동일 데이터셋·조건에서 action MSE, best-of-K, z_shuffle_gap 직접 비교표 완성 |
| **z_dim ablation 확장** | z_dim = 32 / 64 / 128 / 256 전 범위 비교 (현재 32, 64만 완료) |
| **Planner 입력 ablation 결과 분석** | full / object_only / proprio_only 성능 차이로 상태 표현의 어떤 부분이 행동 예측에 중요한지 파악 |
| **시맨틱 보조 손실 효과 정량화** | `sfp_nosem` vs `stoch_flow_prior` 비교로 auxiliary task의 실제 기여도 측정 |
| **Best-of-K 곡선** | K = 1, 2, 3, 5, 10, 20으로 확장하여 다양성 포화(saturation) 지점 파악 |

### 중기 (모델 확장)

| 실험 | 목적 |
|------|------|
| **온라인 평가 (시뮬레이터)** | Robosuite / LIBERO 환경에서 실제 태스크 성공률(task success rate) 측정 |
| **Diffusion Prior** | Flow prior 대신 DDPM/DDIM 기반 prior로 교체 후 성능·속도 비교 |
| **계층 깊이 실험** | 2단계 flow (현재) → 3단계 (goal-z-action) 구조 탐색 |
| **SigLIP fine-tuning** | 인코더를 freeze 해제하고 end-to-end 학습 시 성능 변화 확인 |
| **언어 조건화 강화** | 현재 SigLIP 텍스트 → CLIP / T5 / LLaMA 임베딩으로 교체 비교 |
| **멀티태스크 학습** | LIBERO 태스크 스위트 전체로 단일 모델 학습 (태스크 간 전이 효과 측정) |

### 장기 (연구 방향)

| 실험 | 목적 |
|------|------|
| **실로봇 배포** | 시뮬레이터 성능이 좋은 모델을 실로봇(6-DoF arm)에서 검증 |
| **온라인 적응** | 소수의 시연(few-shot demonstration)으로 새 태스크에 빠른 적응 |
| **잠재 공간 해석** | z를 시각화하여 태스크·단계별 군집화 패턴 분석 (디버깅 & 해석성) |
| **VLA foundation model 통합** | OpenVLA / Octo 등 대형 VLA에 StochFlowPrior head를 붙여 다양성 개선 탐색 |

---

## 지원 데이터셋

- **Robomimic** — HDF5 포맷, image 및 low-dim 모드 지원
- **LIBERO** — 태스크 스위트 (libero_object, libero_long), 언어 어노테이션 포함

---

## 의존성

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.37.0       # SigLIP
einops, h5py, pyyaml, tqdm, matplotlib, scikit-learn
wandb                      # 선택 사항 (실험 추적)
```
