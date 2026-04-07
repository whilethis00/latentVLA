# Apr W2 (2026-04-07 ~ 2026-04-13)

> 목표: M5 실패 원인 진단 → M6 z-Distillation 설계·구현·실험
> paper.md 기준: Aha Moment 2 ("VLM z > MLP z") 증명 경로 확보

---

## 이번 주 핵심 문제

M5 (vlm_sfp_plan_100ep) 결과:

| 지표 | M4 StochFlowPrior | M5 VLM SFP Plan | 판정 |
|------|:-----------------:|:---------------:|------|
| MSE prior ↓ | 0.6540 | **0.6084** | M5 승 |
| z_shuffle_gap ↑ | 0.0428 | **0.0163** | M5 패 |
| cosine_sim | — | 0.9945 | 비정상적으로 높음 |

**진단:** VLM plan token은 task를 구분하지 않는다.
- action flow loss만으로는 z가 discriminative해질 동기가 없음
- cosine_sim 0.9945 = 모든 task에서 거의 같은 z를 냄
- MSE prior가 M4보다 나은 건 VLM feature의 일반적인 표현력 덕분, z 품질 덕분이 아님

**핵심 질문:** 어떻게 VLM z가 oracle z(M2, z_shuffle_gap 0.784)에 가까워지게 할 수 있나?

---

## 해결책: M6 z-Distillation

### 아이디어

M2의 oracle z encoder는 미래 이미지를 보기 때문에 task-discriminative z를 만든다.
이 z를 teacher로 삼아, VLM z가 oracle z를 모방하도록 학습시킨다.

```
학습 시:
  z_oracle = MLP(SigLIP(future_image))   ← M2 encoder, frozen
  z_vlm    = VLM_plan_token → projection ← 학습 대상

  L = L_action_flow(z_vlm)
    + L_prior_flow(z_vlm)
    + λ_sem · L_semantic
    + α · ||z_vlm - z_oracle||²          ← 핵심 추가 loss

추론 시:
  z_vlm만 사용 (future image 필요 없음)
```

### 왜 contrastive보다 나은가

| 방식 | 신호 | 문제 |
|------|------|------|
| Contrastive (InfoNCE) | "다른 task끼리 달라져라" | 방향만 제시, 어느 방향? 몰라 |
| **z-Distillation** | "M2 oracle z처럼 돼라" | 명확한 타겟, 신호 강함 |

---

## 이번 주 태스크

### Day 1 (4/7): M5 진단 문서화 + M6 설계 확정

- [ ] M6 아키텍처 설계 (이 문서 기반 코드 스펙 작성)
- [ ] M6 config yaml 스펙 결정
  - `distill_alpha` 파라미터 추가
  - M2 oracle encoder weight path 지정 방법
- [ ] α sweep 범위 결정: `[0.1, 1.0, 10.0]`

### Day 2-3 (4/8~4/9): M6 구현

- [ ] `models/latent_vla.py` — distillation loss 추가
  - `compute_distill_loss(z_vlm, z_oracle)` 구현
  - oracle encoder (M2 MLP) forward 통합
- [ ] `training/trainer_vlm.py` — distill_alpha loss 반영
- [ ] `configs/vlm_paligemma_distill.yaml` 작성
- [ ] smoke test: distill loss 흐르는지 확인

### Day 4-5 (4/10~4/11): M6 실험 실행

- [ ] **M6a**: α=0.1 — 100 epoch
- [ ] **M6b**: α=1.0 — 100 epoch
- [ ] **M6c**: α=10.0 — 100 epoch

각 실험 모니터링 포인트:
- `z_shuffle_gap` 추이 (M5 0.016 → 목표 0.15+)
- `future_cosine_sim` 추이 (M5 0.9945 → 낮아져야 함)
- `mse_prior` 희생 여부 (너무 높아지면 α 줄여야)

### Day 6 (4/12): 결과 분석

- [ ] M5 vs M6a/b/c z_shuffle_gap 비교표 작성
- [ ] z_oracle과 z_vlm의 cosine similarity 측정 (distillation 품질 확인)
- [ ] best α 결정
- [ ] table_ko.md M6 결과 추가

### Day 7 (4/13): 버퍼 + 다음 주 준비

- [ ] 약한 결과면: α 범위 확장 or 구조적 문제 재진단
- [ ] 강한 결과면 (z_shuffle_gap > 0.3): vlm_sfp_last, vlm_sfp_pool 실험 준비
- [ ] paper.md Aha Moment 2 스토리 업데이트
- [ ] Apr_W3 계획 초안

---

## 성공 기준

| 기준 | 목표값 | 의미 |
|------|--------|------|
| z_shuffle_gap (M6 best) | > 0.15 | 방향이 맞음, 계속 진행 |
| z_shuffle_gap (M6 best) | > 0.30 | 강한 결과, paper story 성립 |
| MSE prior (M6 best) | < 0.65 | M4보다 나빠지지 않아야 함 |

**z_shuffle_gap > 0.3이면:** M6 = Aha Moment 2 증명 완료 → 다음 단계(vlm_sfp_last/pool ablation, real robot 준비)

**z_shuffle_gap 0.05~0.15이면:** 방향은 맞으나 α 튜닝 또는 구조 보완 필요

**z_shuffle_gap < 0.05이면:** oracle encoder freezing 방식 또는 distillation 타겟 재검토

---

## 논문 스토리 연결

```
Aha 1 (완료): M1~M4로 증명 — z_shuffle_gap ↑ → 행동 품질 ↑
              M2 oracle: posterior MSE 0.0017, z_shuffle_gap 0.784

Aha 2 (이번 주 목표):
  M5: VLM naive z는 discriminative하지 않음 (cosine_sim 0.9945, gap 0.016)
      → "VLM을 그냥 쓰면 안 된다" — 중요한 negative result
  M6: z-Distillation으로 VLM이 oracle z를 모방하도록 훈련
      → z_shuffle_gap이 크게 오름 → "VLM z는 설계해야 한다"

Aha 3 (4~5월): z_distill + last/pool/plan ablation → 최선의 z form 확정
```

M5의 "실패"는 논문에서 오히려 핵심 contribution이 된다:
> *"VLM을 naive하게 쓰면 z가 discriminative하지 않다는 것을 발견했고, 이를 해결하는 z-Distillation 방법을 제안한다."*

---

## 실험 디렉토리 (예정)

| 실험 | 경로 | 상태 |
|------|------|------|
| M6a (α=0.1)  | `outputs/runs/vlm_sfp_distill_a01_100ep/` | 예정 |
| M6b (α=1.0)  | `outputs/runs/vlm_sfp_distill_a10_100ep/` | 예정 |
| M6c (α=10.0) | `outputs/runs/vlm_sfp_distill_a100_100ep/` | 예정 |

---

## 참고

- M2 oracle encoder: `outputs/runs/det_latent_100ep_20260405/ckpt_final.pt`
- M5 학습 코드: `configs/vlm_paligemma.yaml` + `scripts/train_vlm.py`
- distillation loss 추가 위치: `training/trainer_vlm.py` → `_compute_loss()`
- oracle encoder 구조: `models/encoders.py` → `FutureEncoder`
