# M6: VLM SFP Plan + z-InfoNCE

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-14 |
| **베이스** | M5 (VLM SFP Plan, 100ep) |
| **목적** | z 공간에 task-discriminative 목표를 명시적으로 추가해 z_shuffle_gap 개선 |
| **상태** | 🟡 학습 완료, 오프라인 eval 미실행 |
| **출력 경로** | `outputs/runs/vlm_sfp_infonce_20260414/` |

---

## 2. 동기 (Why)

M5 z_analysis (2026-04-14) 결과:

```
z_shuffle_gap = 0.016  (M4: 0.043, M2: 0.784)
gap_task   = 43%  of full shuffle gap
gap_motion = 43%  of full shuffle gap
```

- t-SNE: task별 클러스터 시각적으로 확인 → z에 task 구조 존재
- 하지만 클러스터 경계가 흐릿 → action flow가 z의 task 정보를 충분히 활용 못함
- **결론**: task 정보가 z에 있긴 하지만 약하다. InfoNCE로 클러스터를 선명하게 밀어주면 유효할 것

---

## 3. 핵심 아이디어

같은 task의 z는 가깝게, 다른 task의 z는 멀어지도록 InfoNCE loss를 z_star에 직접 추가.

```python
def z_infonce_loss(z, task_ids, temperature=0.07):
    z_norm = F.normalize(z, dim=-1)           # (B, z_dim)
    sim = z_norm @ z_norm.T / temperature     # (B, B)
    pos_mask = (task_ids.unsqueeze(0) == task_ids.unsqueeze(1))
    pos_mask.fill_diagonal_(False)            # 자기 자신 제외
    loss = -torch.log(
        (sim.exp() * pos_mask).sum(1) /
        sim.exp().sum(1)
    ).mean()
    return loss

# 학습 loss
total = flow_loss + λ * z_infonce_loss(z_star, task_ids)
```

**task_ids 추출:** `batch["file"]` 파일명 → 정수 인덱스 (LiberoDataset에 `task_id` 추가)

---

## 4. 구현 범위

| 파일 | 변경 내용 |
|------|----------|
| `data/libero_dataset.py` | `task_id` (int) 필드 추가 — file_idx 기반 |
| `models/stoch_latent_flow_prior.py` | `z_infonce_loss` 함수 추가 |
| `models/latent_vla.py` | `compute_loss`에 InfoNCE 항 추가 |
| `configs/vlm_paligemma_infonce.yaml` | M5 기반, `infonce_weight` / `infonce_temperature` 추가 |

---

## 5. Hyperparameter 탐색

| 파라미터 | 값 | 비고 |
|---------|:--:|------|
| `infonce_weight` (λ) | **0.1** (기본) | 0.01, 0.1, 1.0 ablation |
| `infonce_temperature` | 0.07 | SimCLR 표준값 |
| 나머지 | M5와 동일 | — |

> λ 너무 크면 flow loss와 충돌 가능 → 0.1부터 시작

---

## 6. 성공 기준

| 지표 | 목표 | 비고 |
|------|:----:|------|
| z_shuffle_gap | **> 0.043** | M4 초과 |
| MSE prior | ≤ 0.608 | M5 수준 유지 |
| t-SNE 클러스터 | M5보다 선명 | 시각적 확인 |

---

## 7. 리스크

| 리스크 | 대응 |
|--------|------|
| flow prior loss와 InfoNCE 충돌 | λ=0.01로 낮춰서 재시도 |
| batch 내 task 다양성 부족 | batch_size=16, 10 task → 평균 1.6개/task → shuffle 시 pos pair 부족 가능 → 확인 필요 |
| z_shuffle_gap은 올라도 MSE prior 악화 | λ sweep으로 sweet spot 탐색 |

---

## 8. 학습 커맨드

```bash
# 기본 (λ=0.1)
screen -S m6_infonce
torchrun --nproc_per_node=<N> scripts/train_vlm.py \
    --config configs/vlm_paligemma_infonce.yaml

# λ ablation
torchrun --nproc_per_node=<N> scripts/train_vlm.py \
    --config configs/vlm_paligemma_infonce.yaml \
    --override loss.infonce_weight=0.01

torchrun --nproc_per_node=<N> scripts/train_vlm.py \
    --config configs/vlm_paligemma_infonce.yaml \
    --override loss.infonce_weight=1.0
```

---

## 9. 결과

### 9.1 로그 기준 관찰 (학습 중 validation)

- 학습은 epoch 21 로그까지 기록됐지만, validation은 epoch 20이 마지막이다.
- 최고 `z_shuffle_gap`은 **epoch 5의 0.136956**이었다.
- 이후 `z_shuffle_gap`은 빠르게 붕괴했고, 마지막 validation(epoch 20)에서는 **0.004115**였다.
- `future_cosine_sim`은 0.964684 → 0.976565로 다시 높아져, z collapse가 완전히 해소되지 않았다.
- `action_mse_prior`는 1.741231 → 0.849226까지 개선됐지만, 목표였던 **M5 prior 0.608 유지**는 달성하지 못했다.

### 9.2 오프라인 eval (evaluate_offline_vlm.py, 2026-04-16)

> ckpt_10, ckpt_20 eval 모두 완료 (2026-04-16).

| 지표 | M5 | M6 ckpt_10 | M6 ckpt_20 | 달성? |
|------|:--:|:----------:|:----------:|:-----:|
| action_mse_prior ↓ | **0.6084** | 1.2946 | 0.8588 | ❌ |
| action_mse_posterior ↓ | **0.5318** | 1.1883 | 0.8168 | ❌ |
| prior_posterior_gap ↑ | 0.0766 | 0.1063 | 0.0420 | ✅ / ⚠️ |
| z_shuffle_gap ↑ | 0.0163 | **0.0537** | 0.0268 | ✅ / ⚠️ |
| best_of_1 ↓ | — | 1.3012 | 0.8585 | — |
| best_of_5 ↓ | — | 0.6338 | 0.4362 | — |
| best_of_10 ↓ | — | 0.5033 | **0.3517** | — |
| sampling_diversity ↑ | — | 0.7027 | 0.5076 | — |
| future_cosine_sim ↓ | 0.9945 | **0.9678** | 0.9766 | ✅ / ⚠️ |

**쉽게 말하면:**

ckpt_10과 ckpt_20이 서로 다른 방향으로 좋다.

- **ckpt_10**: z 구조가 살아있음 (z_shuffle_gap=0.054, 목표 달성). 근데 prior가 비효율적 — K를 많이 뽑아야 좋은 action이 나옴 (best_of_10=0.503).
- **ckpt_20**: prior MSE가 낮아지고 best_of_10=0.352로 더 좋음. 근데 z_shuffle_gap이 0.027로 절반 이하로 줄었음.

즉 학습이 진행될수록 **"prior 효율↑, z 구조↓"** 트레이드오프가 발생.

- InfoNCE가 z 구조를 밀어주는 건 맞지만, 시간이 지나면서 prior가 그 z에 맞춰가는 대신 z가 prior한테 끌려가서 구조가 무너짐.
- 결론: **λ를 낮춰서 z 구조는 유지하되 prior 학습을 방해하지 않게 균형 맞추는 게 M7의 핵심.**

### 9.3 M5 대비 전체 비교

| 지표 | M5 | M6 val best (ep5) | M6 val last (ep20) | M6 offline (ckpt_10) |
|------|:--:|:-----------------:|:------------------:|:--------------------:|
| action_mse_prior ↓ | **0.6084** | 1.7412 | 0.8492 | 1.2946 |
| action_mse_posterior ↓ | **0.5318** | 1.4570 | 0.8208 | 1.1883 |
| z_shuffle_gap ↑ | 0.0163 | **0.1370** | 0.0041 | **0.0537** |
| prior_posterior_gap ↑ | 0.0766 | **0.2842** | 0.0284 | 0.1063 |
| future_cosine_sim ↓ | 0.9945 | **0.9647** | 0.9766 | **0.9678** |

### 9.4 해석

- **부분 성공:** InfoNCE가 z를 task-discriminative하게 밀어주는 효과는 실제로 있었다 (z_shuffle_gap 목표 달성).
- **주요 문제:** prior MSE가 M5 대비 2배 이상 나쁨 → prior flow가 InfoNCE와 경합해 학습 효율 저하.
- **핵심 발견:** z 구조와 prior 품질이 trade-off 관계 → λ를 낮추면 prior를 지키면서 z 구조도 어느 정도 챙길 수 있을 것.
- **다음 방향:** λ=0.01 + task-balanced sampler로 M7 설계.

---

## 10. 저장 파일 목록

```
outputs/runs/vlm_sfp_infonce_20260414/
├── ckpt_10.pt
├── ckpt_20.pt
├── train.log
├── train_log.jsonl
├── eval_ckpt_10_plan.json   ← 오프라인 eval 결과 (2026-04-16)
└── eval_ckpt_20_plan.json   ← 오프라인 eval 결과 (2026-04-16)
```
