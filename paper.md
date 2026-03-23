# LatentVLA: 연구 설계 문서

> ICRA 2027 제출 목표
> 작성일: 2026-03-23
> 상태: 연구 설계 단계

---

## 0. 논문 핵심 서사 — 3개의 Aha Moment

리뷰어를 설득하려면 딱 3개의 "aha moment"가 논리적으로 연결되어야 한다.

```
Aha 1: z 품질 → 성능         (왜 z가 중요한가?)
    ↓
Aha 2: VLM-z > MLP-z         (어떻게 z를 개선하는가?)
    ↓
Aha 3: z form이 품질을 결정함 (어떻게 VLM에서 z를 뽑는가?)
```

**Aha 1 — "z 품질이 성능을 결정한다"**
기존 4개 모델(M1~M4)로 증명. z_shuffle_gap vs LIBERO 성공률 scatter plot 하나로
논문의 전제를 확립. 이것이 없으면 "왜 VLM으로 z를 개선해야 하냐"는 질문에 답할 수 없다.
→ **Exp 1이 논문 전체를 정당화한다.**

**Aha 2 — "VLM이 만든 z는 질이 다르다"**
MLP prior(M4) vs PaliGemma prior. z_shuffle_gap, best-of-K, 태스크 성공률 전부에서
VLM-z가 이겨야 논문이 성립한다.
→ **Exp 2가 핵심 contribution이다.**

**Aha 3 — "z를 어떻게 뽑느냐도 중요하다 — 우리가 최선을 찾았다"**
last token / mean pool / plan token 비교 + t-SNE 시각화. "VLM을 그냥 갖다 쓴 게
아니라 z를 설계했다"는 것을 보여주는 파트. 방법론의 깊이를 만든다.
→ **Exp 3이 논문의 완성도를 만든다.**

**6월 전 목표: 이 3개를 끝낸다. 시뮬레이션 스케일업은 그 이후.**

---

## 0b. 한 줄 포지션

> **"LDM이 이미지 생성의 패러다임을 바꿨듯, 우리는 로봇 행동 생성에서 latent space의 역할을 처음으로 체계적으로 정의하고, VLM이 만든 semantic latent z가 그 핵심임을 증명한다."**

---

## 1. 동기 (Motivation)

### LDM 비유

Latent Diffusion Model(LDM, Rombach et al. 2022)의 핵심 통찰은 단순했다:

```
기존: 픽셀 공간에서 Diffusion  →  고차원, 비효율, 의미 없는 노이즈 제거
LDM:  VAE로 latent 압축 → latent 공간에서 Diffusion  →  Stable Diffusion 탄생
```

로봇 행동 생성도 동일한 문제를 안고 있다:

```
기존 VLA(π0, OpenVLA): 행동 공간에서 직접 Flow/Diffusion
  → 7DoF × 8 step = 56차원의 raw action space
  → 언어/시각 이해와 모터 제어가 같은 공간에 뒤섞임
  → "무엇을 할지"와 "어떻게 움직일지"가 분리되지 않음

우리의 제안: VLM → semantic latent z → action space에서 Flow
  → "무엇을 할지"는 VLM이 z로 압축
  → "어떻게 움직일지"는 z에 conditioned된 flow가 처리
```

### 기존 연구의 공백

| 연구 | z 존재 | z 설계 방식 | 문제 |
|------|--------|------------|------|
| π0 | ❌ | VLM 토큰 그냥 공유 | z 개념 없음, 해석 불가 |
| π0.5 | △ | 이산 subtask 텍스트 | 학습 안 되는 z, 연속 X |
| Fast-in-Slow | △ | VLM hidden state 그대로 | z 설계 없음, 품질 측정 없음 |
| RationalVLA | △ | `<ACT>` 토큰 단 하나 | 너무 단순, 표현력 한계 |
| **우리** | ✅ | VLM → 설계된 z → Two-level Flow | **z를 처음으로 제대로 정의** |

**아무도 묻지 않은 질문**: z가 좋아야 행동이 좋아지는가? 어떤 z가 좋은 z인가?

---

## 2. 핵심 연구 질문 (Research Questions)

### RQ1 (핵심): z의 품질이 행동 생성 품질을 결정하는가?

> "Latent z의 품질 지표(z_shuffle_gap, prior_posterior_gap)가 실제 태스크 성공률과 얼마나 강하게 상관관계를 가지는가?"

이 질문이 논문 전체의 근거다. z 품질 → 행동 품질이 성립해야, VLM으로 z를 개선하는 것이 의미 있다.

---

### RQ2: VLM이 만든 z는 MLP가 만든 z보다 품질이 높은가?

> "PaliGemma-3B가 생성한 z는 단순 MLP prior가 생성한 z보다 z_shuffle_gap이 높고, best-of-K MSE가 낮고, 온라인 성공률이 높은가?"

---

### RQ3: z의 형태(form)가 품질에 영향을 미치는가?

> "VLM에서 z를 추출하는 방식(last token / mean pool / plan token)에 따라 품질 지표가 어떻게 달라지는가? 어떤 형태가 가장 'LDM의 latent'에 가까운가?"

---

### RQ4: Two-level Flow가 Single-level Flow보다 VLM z와 시너지가 있는가?

> "VLM-z를 conditioning으로 쓸 때, StochFlowPrior(Two-level Flow)가 DetLatent나 FlatFlow보다 더 큰 성능 향상을 보이는가? VLM z가 클수록 two-level 구조의 이점이 커지는가?"

---

### RQ5: VLM-z는 무엇을 인코딩하는가? (해석 가능성)

> "VLM이 만든 z를 시각화하면 어떤 구조가 보이는가? 같은 태스크의 z는 클러스터링 되는가? z 공간에서 보간(interpolation)하면 자연스러운 행동 전환이 일어나는가?"

---

## 3. 가설 (Hypotheses)

### H1 (z 품질-성능 상관관계)
> z_shuffle_gap이 높을수록 온라인 태스크 성공률이 높다.
> **근거**: 사전 실험에서 StochFlowPrior의 z_shuffle_gap = 0.72 (z가 의미 있음을 확인).

### H2 (VLM-z 우월성)
> VLM-z의 z_shuffle_gap > MLP-z의 z_shuffle_gap.
> VLM은 언어와 시각을 통합해 "태스크의 의도"를 이해하므로, 단순 MLP보다 더 의미 있는 z를 생성할 것이다.

### H3 (z form 가설)
> plan_token z > mean_pool z > last_token z (z_shuffle_gap 기준).
> 명시적으로 계획을 요약하도록 설계된 `[PLAN]` 토큰이 가장 압축된 의미를 담을 것이다.

### H4 (Two-level Flow 시너지)
> VLM-StochFlowPrior > VLM-DetLatent > VLM-FlatFlow (성공률 기준).
> z의 표현력이 높아질수록 Two-level Flow의 이점(다양성, best-of-K)이 더 두드러질 것이다.

### H5 (LDM 스케일링 법칙)
> System 2 VLM 파라미터가 클수록 z 품질이 높아지고, 성능이 향상된다.
> (PaliGemma-3B로 시작, 이후 Qwen2-VL-7B로 검증 가능)

---

## 4. 방법론 (Method)

### 4.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  System 2 (Slow, ~1-5 Hz)                               │
│                                                         │
│  image (224×224) ──┐                                    │
│  language inst.  ──┼──► PaliGemma-3B ──► feature f      │
│  (current state) ──┘    (LoRA fine-tune)  (2048-dim)    │
│                                  │                      │
│                          z-extraction head               │
│                       (last / pool / plan)               │
│                                  │                      │
│                          semantic latent f̃               │
└──────────────────────────────────┼──────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────┐
│  System 1 (Fast, ~10-30 Hz)      │                      │
│                                  ▼                      │
│  ε ~ N(0,I) ──► prior_flow(ε, f̃) ──► z (128-dim)       │
│                                  │                      │
│  ε ~ N(0,I) ──► action_flow(ε, [f̃, z]) ──► actions     │
│                                  │                      │
│                    (H=8 steps × action_dim=7)            │
└─────────────────────────────────────────────────────────┘
```

**현재 코드(StochFlowPrior)와의 차이**: prior_flow의 conditioning이
`SigLIP C_t (256-dim)` → `PaliGemma f̃ (projected, 256-dim)`으로 교체됨.
나머지 구조 동일.

### 4.2 z 추출 방식 (3가지 ablation)

```python
# z_last: 마지막 토큰의 hidden state
f = paligemma.last_hidden_state[:, -1, :]        # (B, 2048)

# z_pool: 마지막 K=8 토큰 평균
f = paligemma.last_hidden_state[:, -8:, :].mean(1)  # (B, 2048)

# z_plan: 입력에 [PLAN] 특수 토큰 추가, 그 위치의 hidden state
#   input: [image tokens] [language tokens] [PLAN]
f = paligemma.last_hidden_state[:, plan_token_idx, :]  # (B, 2048)

# 공통: Linear projection
f̃ = nn.Linear(2048, 256)(f)  # (B, 256) → prior_flow conditioning
```

### 4.3 학습 전략

**Stage 1 (Warm-up, 10 epoch)**: PaliGemma frozen, projection head + prior/action flow만 학습
**Stage 2 (Joint, 90 epoch)**: PaliGemma LoRA(rank=16) + 전체 joint 학습

**손실 함수 (기존 StochFlowPrior와 동일)**:
```
L = L_action_flow + λ_prior * L_prior_flow + λ_sem * L_semantic
  = FM(a | f̃, z*) + FM(z* | f̃) + (1 - cos_sim(pred_future, gt_future))
```

---

## 5. 실험 설계 (Experiments)

### Exp 1. z 품질 지표 검증 (RQ1)

**목적**: z_shuffle_gap이 실제 성능의 proxy metric임을 증명
**방법**:
- 현재 4개 모델(M1~M4)의 z_shuffle_gap과 LIBERO 성공률을 측정
- Pearson/Spearman 상관계수 계산
- scatter plot: x=z_shuffle_gap, y=task success rate

**예상 결과**: 강한 양의 상관관계 (r > 0.85)
**의의**: z_shuffle_gap을 온라인 평가 없이도 쓸 수 있는 proxy metric으로 제시 → 논문 기여 1

**구체적 설정**:
- 데이터셋: LIBERO-Object (10 tasks, 50 demos/task)
- 평가: 각 task 20회 롤아웃, 성공률 측정
- 모델: M1(FlatFlow), M2(DetLatent), M3(StochVAE), M4(StochFlowPrior)

---

### Exp 2. VLM-z vs MLP-z 비교 (RQ2, 핵심 실험)

**목적**: VLM이 만든 z가 MLP z보다 실제로 좋음을 증명
**방법**: System 2만 교체, System 1(StochFlowPrior) 고정

| 모델명 | System 2 | System 1 |
|--------|---------|---------|
| `mlp_sfp` | MLP prior (기존 M4) | StochFlowPrior |
| `vlm_sfp_last` | PaliGemma (last token) | StochFlowPrior |
| `vlm_sfp_pool` | PaliGemma (mean pool) | StochFlowPrior |
| `vlm_sfp_plan` | PaliGemma (plan token) | StochFlowPrior |

**평가 지표**:
- z_shuffle_gap (오프라인, 빠름) → z 품질 측정
- action_mse_prior (오프라인) → 행동 예측 정확도
- best_of_5 (오프라인) → 다양성 효용
- task_success_rate (온라인, LIBERO) → 실제 성능

**기대 결과**:
```
z_shuffle_gap: vlm_sfp_plan > vlm_sfp_pool > vlm_sfp_last > mlp_sfp
success_rate:  vlm_sfp_plan > vlm_sfp_pool > vlm_sfp_last > mlp_sfp
```

**구체적 설정**:
- 데이터셋: LIBERO-Object + LIBERO-Spatial
- Epoch: 100, LR: 3e-4 (flow), 3e-5 (LoRA)
- LoRA rank: 16, alpha: 32, target: q_proj, v_proj
- Seed: 42, 43, 44 (3회 반복 평균)

---

### Exp 3. z Form Ablation (RQ3)

**목적**: z 추출 방식 중 어떤 것이 가장 좋은 z를 만드는지 체계적 분석
**방법**: Exp 2에서 3가지 VLM-z 변형 결과 비교 (동일 데이터)

추가 분석:
- **z 시각화**: t-SNE로 각 방식의 z 공간 구조 비교
  - 같은 task 종류의 z가 클러스터링 되는가?
  - plan_token이 더 명확한 클러스터를 형성하는가?
- **z 보간 실험**: 두 task의 z를 선형 보간 → 생성된 행동이 자연스럽게 전환되는가?

---

### Exp 4. Two-level Flow vs Single-level, VLM 조건에서 (RQ4)

**목적**: VLM z가 있을 때 Two-level 구조의 이점 확인
**방법**: System 2 고정(vlm_sfp_plan), System 1 변경

| 모델명 | System 1 구조 |
|--------|-------------|
| `vlm_flat` | FlatFlow (z 없이 f̃ → action) |
| `vlm_det` | DetLatent (f̃ → MLP → z → action flow) |
| `vlm_vae` | StochVAE (f̃ → Gaussian z → action flow) |
| `vlm_sfp` | StochFlowPrior (f̃ → flow → z → action flow) |

**기대 결과**:
- VLM z가 강할수록 Two-level 구조가 더 큰 이점을 냄
- best-of-K 지표에서 vlm_sfp가 압도적

---

### Exp 5. 스케일링 및 일반화 (RQ2 확장, ICRA appeal)

**목적**: VLM 크기가 z 품질에 미치는 영향 + 미학습 태스크 일반화

**5a. VLM 크기 스케일링**:
| 모델 | System 2 파라미터 | z_shuffle_gap | 성공률 |
|------|--------------|--------------|--------|
| `mlp_sfp` | ~1M | ? | ? |
| `vlm_sfp_paligemma3b` | 3B | ? | ? |
| `vlm_sfp_qwen7b` | 7B | ? | ? |

**5b. Cross-task 일반화**:
- LIBERO-Object로 학습 → LIBERO-Long으로 테스트 (미학습 태스크)
- VLM-z vs MLP-z의 일반화 격차 측정
- **가설**: VLM은 언어 이해 능력으로 미학습 태스크도 z를 잘 만든다

---

### Exp 6. z 해석 가능성 분석 (RQ5, ICRA 심사위원 흥미 유발)

**목적**: VLM-z가 의미 있는 구조를 갖는지 시각적으로 증명

**6a. 태스크별 z 클러스터링**
- 10가지 LIBERO task의 z를 t-SNE/UMAP으로 2D 투영
- 같은 태스크끼리 묶이면 "z가 태스크를 이해함" 증명

**6b. z 보간 실험**
- Task A: "pick up ketchup" / Task B: "pick up milk"
- z_A와 z_B를 0~1로 보간 → 각 z에서 행동 생성
- 중간 z에서는 두 물체 모두 시도하는 행동이 나오는가?

**6c. z 차원 기여도 분석**
- 각 z 차원을 하나씩 끄고(zero out) 성능 변화 측정
- 어떤 차원이 "물체 위치", "접근 방향", "그리퍼 스타일"을 인코딩하는지 추론

---

## 6. 평가 지표 전체 목록

### 오프라인 지표 (빠름, 학습 중 모니터링)

| 지표 | 의미 | 방향 |
|------|------|------|
| `action_mse_prior` | prior z로 예측한 행동 오차 | ↓ |
| `action_mse_posterior` | 정답 z로 예측한 행동 오차 (상한) | ↓ |
| `best_of_5` | 5샘플 중 최선 MSE | ↓ |
| `z_shuffle_gap` | z 품질 proxy | ↑ |
| `prior_posterior_gap` | prior가 posterior를 얼마나 잘 근사 | ↓ |
| `future_cosine_sim` | z의 미래 예측 능력 | ↑ |
| `sampling_diversity` | 행동 다양성 | 적정 수준 |

### 온라인 지표 (느림, 최종 평가)

| 지표 | 의미 |
|------|------|
| `task_success_rate` | 시뮬레이터 태스크 성공률 (LIBERO) |
| `cross_task_generalization` | 학습 안 한 태스크 성공률 |
| `inference_latency` | 행동 생성 시간 (실시간성) |

---

## 7. 베이스라인 정리

| 베이스라인 | 출처 | 비교 목적 |
|-----------|------|---------|
| FlatFlow (M1) | 우리 (기존) | latent 없는 경우 |
| DetLatent (M2) | 우리 (기존) | 결정론적 MLP z |
| StochVAE (M3) | 우리 (기존) | Gaussian z |
| StochFlowPrior (M4) | 우리 (기존) | Flow z, 최강 baseline |
| π0-style flat | π0 논문 재현 | 현재 SOTA VLA 방식 |
| OpenVLA (if feasible) | 공개 모델 | 대형 VLA baseline |

---

## 8. 데이터셋 계획

| 데이터셋 | 현재 상태 | 용도 |
|---------|---------|------|
| LIBERO-Object | ✅ 있음 (25GB) | 핵심 실험 (Exp 1-4) |
| LIBERO-Spatial | ✅ 있음 (25GB 내 포함 추정) | Cross-task 일반화 |
| LIBERO-Long | ✅ 있음 (25GB 내 포함 추정) | 어려운 일반화 테스트 |
| BridgeData V2 | ❌ 없음 → 다운로드 필요 | 스케일 실험 (Exp 5) |

---

## 9. 논문 스토리 구조 (ICRA 기준 8페이지)

```
1. Introduction (1.5p)
   - LDM 비유로 motivation
   - 기존 VLA의 "latent 없는 문제" 지적
   - RQ와 기여 명시

2. Related Work (0.5p)
   - Flow matching for robotics (π0, Diffusion Policy)
   - System 1/2 VLA (Fast-in-Slow, RationalVLA)
   - Latent variable models (LDM, VAE)

3. Method (2p)
   - System 1/2 아키텍처 전체
   - z 추출 방식 3가지
   - 학습 전략 (2-stage)

4. Experiments (3p)
   - Exp 1: z 품질 지표 검증 (0.5p)
   - Exp 2: VLM-z vs MLP-z (1p, 핵심)
   - Exp 3: z Form Ablation (0.5p)
   - Exp 4: System 1 구조 비교 (0.5p)
   - Exp 5/6: 스케일링 + 해석 (0.5p)

5. Conclusion (0.5p)
```

---

## 10. 차별화 요약 (심사위원 설득 포인트)

1. **새로운 관점**: "로봇 행동 생성 = latent space 문제"를 처음으로 LDM 프레임으로 정의
2. **측정 프레임워크**: z 품질 측정 지표(z_shuffle_gap)를 제시하고, 이것이 실제 성능의 proxy임을 실험으로 증명
3. **VLM z의 우월성**: 단순 주장이 아니라 3가지 z form ablation + 온/오프라인 지표로 체계적 증명
4. **해석 가능성**: z 시각화로 VLM-z가 실제로 의미 있는 구조를 학습함을 보임
5. **재현 가능성**: 모든 코드, 데이터(LIBERO), 설정을 공개

---

## 11. 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| VLM-z가 MLP-z보다 별로 안 좋음 | 낮음 | z form 중 하나는 확실히 좋을 것, Qwen2-VL-7B로 확인 |
| LIBERO가 너무 쉬워서 차이가 없음 | 중간 | LIBERO-Long + BridgeData V2 추가 |
| 학습이 불안정 (LoRA + Flow 동시) | 중간 | 2-stage 학습, warm-up으로 완화 |
| 추론 속도가 너무 느림 | 낮음 | PaliGemma async 실행, action chunking으로 완화 |
| 9월 마감까지 실험 부족 | 중간 | Exp 1-4를 핵심으로, Exp 5-6은 선택 |

---

## 12. 타임라인

```
4월 2026: 코드 구현
  - System2VLM (PaliGemma wrapper)
  - LatentVLA (통합 모델)
  - 학습 파이프라인 업데이트
  - Smoke test + LIBERO 빠른 학습 확인

5월 2026: 핵심 실험
  - Exp 1 (z 품질 검증): 2주
  - Exp 2 (VLM-z vs MLP-z): 2주
  - 중간 결과로 방향 점검

6월 2026: 확장 실험
  - Exp 3 (z form ablation)
  - Exp 4 (System 1 비교)
  - BridgeData V2 다운로드 + 실험

7월 2026: 분석 + 보강
  - Exp 5 (스케일링)
  - Exp 6 (z 해석 가능성)
  - 약한 결과 보강 실험

8월 2026: 논문 작성
  - 초안 작성
  - 피규어 제작
  - 내부 리뷰

9월 2026: 제출
  - ICRA 2027 제출
  - arXiv 동시 공개
```

---

*본 문서는 연구 진행에 따라 지속 업데이트됩니다.*
