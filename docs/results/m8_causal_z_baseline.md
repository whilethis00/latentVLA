# M8 causal-z baseline

## 목적

M9를 시작하기 전에 M8 checkpoint를 새 causal-z metric으로 재평가했다. 핵심 질문은 posterior `z_q`와 prior `z_p`가 action decoder에 같은 방식으로 쓰이는지, 그리고 inference-time prior `z_p`가 샘플별 plan 내용을 실제로 통제하는지다.

- checkpoint: `outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt`
- 평가 스크립트: `scripts/eval_causal_z.py`
- mode: `both`
- max_batches: full validation set
- 생성일: 2026-05-12

## 핵심 결론

M8은 단순한 "decoder가 z를 전혀 안 쓴다" 케이스가 아니다.

정확한 판정은 다음이다.

> Decoder는 z의 존재와 분포 스케일에는 민감하지만, inference-time prior z의 샘플별 plan 내용에는 약하게만 민감하다.

근거:

1. `null`과 `random` intervention에서는 prior/posterior 모두 MSE가 크게 악화된다.
2. 그러나 `shuffle`과 `task_negative`에서는 prior delta가 매우 작다.
3. posterior도 `shuffle/task_negative`에는 modest하게만 민감하다.
4. `probe_ratio=0.9936`으로 posterior `mu_q`가 task/future를 강하게 분리하지 못한다.

따라서 M9의 목적은 action MSE를 낮추는 것이 아니라, prior `z_p`가 decoder에 샘플별 plan으로 쓰이도록 만드는 것이다.

## 결과 요약

| intervention | mode | mse_original | mse_intervened | delta | ratio | z_norm | z_var | probe_ratio |
|--------------|------|--------------|----------------|-------|-------|--------|-------|-------------|
| shuffle | posterior | 0.461227 | 0.488851 | +0.027624 | 1.0728 | 4.164060 | 0.016820 | 0.9936 |
| shuffle | prior | 0.509995 | 0.517939 | +0.007945 | 1.0216 | 4.017860 | 0.013556 | - |
| null | posterior | 0.461049 | 0.715664 | +0.254615 | 1.7623 | 4.164194 | 0.016823 | 0.9936 |
| null | prior | 0.510675 | 0.715347 | +0.204671 | 1.5862 | 4.014611 | 0.013520 | - |
| random | posterior | 0.460517 | 1.372104 | +0.911587 | 4.2219 | 4.163909 | 0.016822 | 0.9936 |
| random | prior | 0.512677 | 1.411882 | +0.899205 | 4.0177 | 4.018957 | 0.013676 | - |
| task_negative | posterior | 0.460142 | 0.488417 | +0.028275 | 1.0737 | 4.164060 | 0.016822 | 0.9936 |
| task_negative | prior | 0.511164 | 0.514166 | +0.003001 | 1.0151 | 4.017887 | 0.013618 | - |
| motion_negative | posterior | 0.460199 | 0.526780 | +0.066581 | 1.1718 | 4.164009 | 0.016822 | 0.9936 |
| motion_negative | prior | 0.510402 | 0.528476 | +0.018074 | 1.0562 | 4.017384 | 0.013587 | - |

## 해석

### 1. Prior path는 완전히 죽은 것은 아니다

`null` prior delta는 `+0.204671`, `random` prior delta는 `+0.899205`다. 즉 decoder는 prior `z`가 완전히 없어지거나 off-manifold random vector로 바뀌면 크게 망가진다.

이것은 "decoder가 z를 전혀 무시한다"는 해석을 배제한다.

### 2. 하지만 prior z의 plan content는 약하다

정작 sample-level plan intervention인 `shuffle`과 `task_negative`에서는 prior delta가 매우 작다.

- `shuffle`: `+0.007945`
- `task_negative`: `+0.003001`
- `motion_negative`: `+0.018074`

이는 prior `z_p`를 다른 샘플의 prior `z_p`로 바꿔도 action prediction이 거의 변하지 않는다는 뜻이다. 따라서 inference-time prior `z_p`는 decoder가 기대하는 plan-specific control signal로 충분히 쓰이지 않는다.

### 3. Posterior path는 prior보다 낫지만 아직 강하지 않다

posterior는 모든 content intervention에서 prior보다 민감하다.

- `shuffle`: posterior `+0.027624` vs prior `+0.007945`
- `task_negative`: posterior `+0.028275` vs prior `+0.003001`
- `motion_negative`: posterior `+0.066581` vs prior `+0.018074`

하지만 posterior도 `shuffle/task_negative` delta가 아주 크지는 않다. posterior `z_q`도 future/task separation이 충분히 강하지 않다.

### 4. Probe ratio는 posterior separation 약함을 지지한다

`probe_ratio=0.9936`은 same-task pair와 random pair의 posterior distance가 거의 같다는 뜻이다. 이 값은 posterior `mu_q`가 task/future를 명확히 구조화하지 못하고 있음을 시사한다.

## M9에 대한 직접 결론

M9는 새 architecture를 크게 만들기보다 먼저 다음 failure mode를 겨냥해야 한다.

> Decoder가 prior z의 manifold presence에는 민감하지만, prior z의 sample-specific plan content에는 약하게 민감하다.

따라서 M9는 `VLM-SFP + Prior-Action Co-training`으로 시작한다.

목표 metric:

| metric | M8 baseline | M9 목표 |
|--------|-------------|---------|
| `prior_z_shuffle_gap` | 0.007945 | 상승 |
| `prior_task_negative_gap` | 0.003001 | 상승 |
| `prior_motion_negative_gap` | 0.018074 | 상승 |
| `delta_null_prior` | 0.204671 | 유지 또는 완만 상승 |
| `action_mse_prior` | 약 0.51 | 크게 악화되지 않음 |
| `probe_ratio` | 0.9936 | M10-M12에서 개선 대상 |

M9 성공 기준은 MSE 단독 개선이 아니다. 최소 성공 조건은 prior content intervention에 대한 민감도 상승이다.

## 다음 실행

```bash
# M9: VLM-SFP + Prior-Action Co-training
# 목적: decoder를 training 중 prior z에도 노출하여 prior path를 action decoder에 묶기
bash scripts/run_m9_prior_action_cotrain.sh
```
