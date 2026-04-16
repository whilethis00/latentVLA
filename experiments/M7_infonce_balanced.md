# M7: VLM SFP Plan + z-InfoNCE (λ=0.01, Task-Balanced Sampler)

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-16 |
| **베이스** | M6 (VLM SFP + InfoNCE, λ=0.1) |
| **목적** | M6의 두 가지 실패 원인을 동시에 수정해 z 구조와 prior 품질을 함께 달성 |
| **상태** | 🔴 설계 완료, 학습 미실행 |
| **출력 경로** | `outputs/runs/vlm_sfp_infonce_balanced_20260416/` |

---

## 2. M6에서 배운 것 (Why M7)

M6 오프라인 eval (2026-04-16) 결과:

| 지표 | M5 | M6 ckpt_10 | M6 ckpt_20 | 목표 |
|------|:--:|:----------:|:----------:|:----:|
| action_mse_prior ↓ | **0.608** | 1.295 | 0.859 | ≤ 0.608 |
| z_shuffle_gap ↑ | 0.016 | **0.054** | 0.027 | > 0.043 |
| best_of_10 ↓ | — | 0.503 | **0.352** | — |

**핵심 발견:**
- z 구조(z_shuffle_gap)와 prior 품질(action_mse_prior)이 트레이드오프 — 학습이 진행될수록 prior가 좋아지면서 z 구조가 무너짐
- 두 가지 원인:
  1. **λ=0.1이 너무 큼** → prior flow loss와 경합해 prior 학습 불안정
  2. **batch 내 positive pair 부족** → batch_size=16, 10 task → 평균 1.6개/task → InfoNCE가 collapse로 빠짐

---

## 3. 변경 사항

### 3.1 λ 축소: 0.1 → 0.01

InfoNCE를 "보조 신호"로만 쓰는 것. prior flow loss 대비 10분의 1 수준으로 낮춰서 경합 제거.

```yaml
loss:
  infonce_weight: 0.01   # M6: 0.1 → M7: 0.01
  infonce_temperature: 0.07
```

### 3.2 Task-Balanced Sampler 도입

batch 내 같은 task 데이터가 평균 4개 이상 들어오도록 보장. InfoNCE positive pair 부족 문제 해결.

**구현 위치:** `data/task_balanced_sampler.py` (신규)

```python
class TaskBalancedSampler(Sampler):
    """
    각 batch에 task당 최소 samples_per_task개 이상 포함되도록 샘플링.
    - num_tasks_per_batch: batch에 포함할 task 수 (기본: 4)
    - samples_per_task: task당 샘플 수 (기본: 4)
    → effective batch_size = num_tasks_per_batch * samples_per_task = 16 (M6와 동일)
    """
```

batch 구성 예시 (batch_size=16):
- M6: task 최대 10개, 평균 1.6개/task → positive pair 거의 없음
- M7: task 4개 × 4샘플 → task당 positive pair 6쌍 보장

**변경 파일:** `training/builder.py` — `build_dataloaders_vlm`에 sampler 옵션 추가

```python
# builder.py
if cfg["data"].get("use_task_balanced_sampler", False):
    from data.task_balanced_sampler import TaskBalancedSampler
    train_sampler = TaskBalancedSampler(
        dataset=train_ds,
        num_tasks_per_batch=cfg["data"].get("num_tasks_per_batch", 4),
        samples_per_task=cfg["data"].get("samples_per_task", 4),
    )
    shuffle = False
```

---

## 4. 구현 범위

| 파일 | 변경 내용 |
|------|----------|
| `data/task_balanced_sampler.py` | TaskBalancedSampler 신규 작성 |
| `training/builder.py` | `build_dataloaders_vlm`에 task_balanced_sampler 옵션 추가 |
| `configs/vlm_paligemma_infonce_balanced.yaml` | M6 config 기반, λ=0.01 + sampler 설정 추가 |

나머지 (모델, loss 계산 등)는 M6와 동일.

---

## 5. Hyperparameter

| 파라미터 | M6 | M7 | 비고 |
|---------|:--:|:--:|------|
| `infonce_weight` (λ) | 0.1 | **0.01** | 핵심 변경 |
| `infonce_temperature` | 0.07 | 0.07 | 동일 |
| `use_task_balanced_sampler` | False | **True** | 핵심 변경 |
| `num_tasks_per_batch` | — | **4** | batch당 task 수 |
| `samples_per_task` | — | **4** | task당 샘플 수 |
| `batch_size` | 16 | 16 | 동일 (4×4=16) |
| `num_epochs` | 100 | 100 | 동일 |
| 나머지 | M6와 동일 | — | — |

---

## 6. 성공 기준

| 지표 | 목표 | 비고 |
|------|:----:|------|
| z_shuffle_gap ↑ | **> 0.043** | M4 수준 초과, M6 ckpt_10(0.054) 이상 유지 |
| action_mse_prior ↓ | **≤ 0.608** | M5 수준 유지 |
| 두 지표 동시 달성 | — | M6의 트레이드오프 해소가 핵심 목표 |

---

## 7. 리스크

| 리스크 | 대응 |
|--------|------|
| λ=0.01이 너무 작아 z 구조 효과 없음 | val z_shuffle_gap 모니터링, 효과 없으면 0.05로 재시도 |
| TaskBalancedSampler에서 task 수 부족한 경우 (val set) | val은 기존 random sampler 유지, train만 적용 |
| 샘플 편향으로 학습 데이터 다양성 감소 | epoch 내 전체 데이터 커버되는지 확인 |

---

## 8. 학습 커맨드

```bash
screen -S m7_balanced
cd /home/introai4/.agile/users/hsjung/projects/VLA
torchrun --nproc_per_node=<N> scripts/train_vlm.py \
    --config configs/vlm_paligemma_infonce_balanced.yaml
```

---

## 9. 결과

> 학습 미실행

---

## 10. 저장 파일 목록

```
outputs/runs/vlm_sfp_infonce_balanced_20260416/
├── ckpt_*.pt
├── train.log
├── train_log.jsonl
└── result.md
```
