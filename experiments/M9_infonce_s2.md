# M9: InfoNCE Throughout (S1 + S2)

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-21 |
| **베이스** | M8 (`vlm_sfp_infonce_s1only_20260418`, ckpt_90.pt) |
| **핵심 변경** | `infonce_stage1_only: false` — S2에서도 InfoNCE 유지 |
| **상태** | 🔴 설계 완료, 학습 미실행 |
| **출력 경로** | `outputs/runs/vlm_sfp_infonce_s2also_20260421/` |

---

## 2. 왜 FiLM/VQ 이전에 이 실험인가 — 방향 전환 경위

M8 이전까지 예정된 흐름은 `M8 = FiLM`, `M9 = VQ`였다.
그런데 실제 M8(`s1only`)을 돌리고 진단을 해보니, FiLM/VQ로 가기 전에 더 근본적인 문제가 있었다.

### 두 연구 질문이 섞여 있었다

지금까지 실험들은 겉으로는 "prior MSE를 낮추는 법"을 찾는 것처럼 움직였다.
그런데 실제로는 전혀 다른 두 가지 문제를 동시에 건드리고 있었다.

1. **좋은 latent plan을 만들 수 있는가** — posterior z가 미래/task에 따라 다른 정보를 담는가
2. **만든 latent plan을 policy가 실제로 쓰는가** — action decoder가 z에 의존해서 행동을 만드는가

이 두 가지가 섞이면, MSE가 좋아져도 "왜 좋아졌는지"를 알 수 없다.
M8도 그랬다. `mse_prior=0.5120`, `mse_posterior=0.4606`으로 개선됐지만,
`z_shuffle_gap=0.0086`은 끝까지 낮았고, `prior_flow_loss`는 epoch 20 이후 다시 올라갔다.

### 진짜 핵심 주장

이 연구가 논문이 되려면, 다음 문장을 증명해야 한다.

> **"latent plan이 action generation에 causally necessary하다."**

이 문장이 안 서면 FiLM이든 VQ든 그냥 "surface metric을 움직이는 트릭"에 불과하다.

---

## 3. M8 판결 실험 결과 (2026-04-21)

### 무엇을 측정했나

`ckpt_90.pt`로 `scripts/eval_z_diag.py` 실행 (N=800, max_batches=50).

```bash
conda run -n vla python3 scripts/eval_z_diag.py \
    --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
    --max_batches 50
```

### 결과 수치

| 지표 | 값 | 의미 |
|------|-----|------|
| `z_mu_var_mean` | **0.0345** | posterior z의 배치 내 분산 (낮을수록 수축) |
| `delta_null` | **+0.2255** | posterior z를 zeros로 교체 시 MSE 상승 |
| `delta_shuffle` | +0.1112 | posterior z를 셔플 시 MSE 상승 |
| `probe_ratio` | 0.9198 | same-task / random 거리 비율 |
| `z_shuffle_gap` (M8 ep90) | 0.0086 | prior z 셔플 시 MSE 변화 |

> `delta_null`과 `z_shuffle_gap`은 측정 경로가 다르다.
> - `z_shuffle_gap`: **prior z** 셔플 기준
> - `delta_null`: **posterior z** 교체 기준

### 판정: 케이스 C

세 숫자를 같이 읽으면 구조가 보인다.

**posterior → decoder 경로는 살아 있다.**
`delta_null = +0.2255` — posterior z를 zeros로 바꾸면 MSE가 크게 오른다.
decoder는 posterior z를 실제로 사용하고 있다.
`z_shuffle_gap`만 보고 "decoder가 z를 안 쓴다"고 해석한 것은 틀렸다.

**prior → decoder 경로는 죽어 있다.**
`z_shuffle_gap = 0.0086` — prior flow가 뽑은 z를 셔플해도 MSE가 거의 안 변한다.
inference time에 prior로 샘플링한 z는 decoder가 사실상 무시한다.
즉 문제는 decoder가 z를 안 쓰는 게 아니라, **prior z가 decoder가 기대하는 공간에 없다**는 것이다.

**posterior z 자체는 충분히 퍼지지 않았다 (수축 의심, 확정은 아님).**
`z_mu_var_mean = 0.0345`는 낮은 편으로 보이며 contraction 의심을 준다.
단 baseline이 없으므로 이 숫자만으로 확정하기는 이르다.
`probe_ratio = 0.9198`까지 같이 읽으면 해석이 강해진다 — same-task 샘플끼리 z 거리가 random pair와 거의 같다는 건, z가 task/미래 정보를 구분해서 인코딩하지 않는다는 뜻이다.

**한 줄 요약:**
decoder는 posterior z를 쓰고 있지만, 그 z가 future/task에 따라 충분히 달라지지 않는다.
prior z는 그 posterior 공간을 inference time에 재현하지 못하고 있다.

---

## 4. 원인 분석 — 왜 z-space가 수축했는가

### InfoNCE가 Stage 1에서만 켜져 있었다

`train_log.jsonl`을 보면:

```
infonce_loss 있는 epoch: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

M8의 config: `infonce_stage1_only: true` — Stage 2 진입 시 InfoNCE 자동 비활성화.

이것이 M8의 핵심 설계 선택이었다.

### 시간축으로 보면 인과가 맞물린다

| 구간 | 상태 | 결과 |
|------|------|------|
| ep 1~9 (Stage 1) | InfoNCE 활성 + semantic loss 학습 중 | z가 어느 정도 구분됨 |
| ep 10 (Stage 2 시작) | **InfoNCE 꺼짐** | `z_shuffle_gap = 0.0059` 이미 약함 |
| ep 11~20 | semantic_future_loss만 남아 수렴 중 | 0.02 → 0.003으로 빠르게 saturation |
| ep 20 이후 | z를 future/task-discriminative하게 유지하는 직접 압력 약화 | `prior_flow_loss` 역증가 시작 |

**ep 20 이후 z에 남는 신호:**
- action_flow_loss: z를 specific하게 만들 유인이 없음 (어떤 z든 action 맞추면 됨)
- prior_flow_loss: z의 내용이 아니라 z의 위치를 따라가는 것
- semantic_future_loss: **0.0028으로 saturate** — cosine loss가 방향만 맞으면 되므로 z magnitude나 diversity가 collapse해도 loss가 0에 수렴

즉 **ep 20 이후 z를 informative하게 유지하는 직접 압력이 사실상 없다.**

### InfoNCE의 역할

이 코드베이스의 InfoNCE는 language embedding 기반이 아니라 **task_id 기반 contrastive**다.

```python
# latent_vla.py
def _z_infonce_loss(z, task_ids, temperature):
    z_norm = F.normalize(z, dim=-1)
    sim = z_norm @ z_norm.T / temperature
    pos_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)  # same task = positive
    ...
```

같은 task 샘플의 z는 가깝게, 다른 task 샘플의 z는 멀게 만든다.
이게 꺼지면, z들이 한 점으로 모여도 loss가 줄기 때문에 z-space가 수축할 자연적 유인이 생긴다.

LoRA drift 우려: InfoNCE가 language embedding이 아니라 task_id를 key로 쓰기 때문에,
S2에서 VLM LoRA가 바뀌어도 contrastive target(task_id)은 흔들리지 않는다.
생각보다 안전하다.

---

## 5. M9 설계

### 핵심 변경: 단 하나

```yaml
# M8: infonce_stage1_only: true
# M9: infonce_stage1_only: false
infonce_stage1_only: false
```

나머지는 M8과 완전히 동일하게 둔다.

**왜 이게 맞는 첫 수인가:**
- 변수가 하나다 — config 플래그 하나
- 인과 관계가 명확하다 — InfoNCE 꺼진 시점과 z collapse 시점이 맞물린다
- 실패해도 해석이 된다 — "S2에서 InfoNCE를 켜도 z가 안 살아난다"면 다른 원인이 있다는 뜻
- 성공해도 해석이 된다 — "S2 InfoNCE가 z-space를 유지하는 핵심 압력이었다"

### 코드 변경 2개

**① `stoch_latent_flow_prior.py`**: `_z_mu_var` 추가 (train loop에서 z-space 수축 궤적 추적용)

```python
# compute_loss return dict에 추가
"_z_mu_var": mu_q.var(dim=0).mean(),   # 기존에 없던 것 — 배치 방향 분산
```

**② Config**: 새 yaml 생성

```yaml
# configs/vlm_paligemma_infonce_s2also.yaml
# M8(s1only) 기반, infonce_stage1_only만 false로 변경
loss:
  infonce_weight: 0.1
  infonce_temperature: 0.07
  infonce_stage1_only: false    # ← 핵심 변경

training:
  output_dir: "outputs/runs/vlm_sfp_infonce_s2also_20260421"
```

### LoRA warm-start 추가 여부

S2 시작 시 VLM LoRA가 unfreezing되면서 f_tilde가 빠르게 변한다.
이때 InfoNCE가 갑자기 켜지면 z_star가 불안정해질 수 있다.

결론: **별도 warm-start 불필요.**
- 기존 `s2_lora_warmup_steps`가 LR 안정화를 이미 담당
- InfoNCE weight 0.1은 S1에서 이미 켜져 있던 값 — S2에서 "계속 유지"하는 것이므로 갑작스러운 충격 없음

---

## 6. 성공 기준

MSE 수치가 아니라 **z-space 품질의 M8 대비 상대 개선**으로 본다.

| 지표 | M8 (ep90) | M9 목표 | 의미 |
|------|-----------|---------|------|
| `z_mu_var_mean` | 0.0345 | **유의미한 상승** | posterior z가 더 퍼졌는가 |
| `probe_ratio` | 0.9198 | **하락** | same-task z가 더 뭉쳤는가 (task 구분 개선) |
| `z_shuffle_gap` | 0.0086 | **상승, 가능하면 0.02+** | prior z가 action에 기여하기 시작했는가 |
| `prior_flow_loss` 추세 | ep20 이후 역증가 | **단조 감소 유지 여부** | prior가 posterior를 따라가는가 |
| `mse_posterior` | 0.4606 | 유지 또는 개선 | representation 개선이 quality도 올리는가 |

> z_mu_var_mean의 절대값 기준(> 0.1 등)은 아직 baseline이 없어 방어력이 약하다.
> M8 대비 상대 개선이 더 정직한 기준이다.

### 실패 시 해석

| 결과 | 해석 |
|------|------|
| z_mu_var_mean 안 오름 | InfoNCE가 주원인이 아님. semantic loss saturation이나 posterior encoder capacity 문제 |
| z_mu_var_mean 오르는데 z_shuffle_gap 안 오름 | posterior z는 퍼졌지만 prior-posterior mismatch 별도 해결 필요 |
| z_mu_var_mean 오르고 z_shuffle_gap도 오름 | **원인 확정: S2 InfoNCE 제거가 z collapse의 주원인** |

---

## 7. 측정 추가 — `z_mu_var_mean` train loop 기록

M8 train_log.jsonl에 `z_mu_var_mean`이 없어서 수축 궤적을 볼 수 없었다.
M9부터는 이 숫자를 epoch마다 찍는다.

`stoch_latent_flow_prior.py`의 compute_loss에 추가:

```python
with torch.no_grad():
    _z_mu_norm    = mu_q.norm(dim=-1).mean()
    _z_mu_var     = mu_q.var(dim=0).mean()      # ← 신규 추가
    _z_var_mean   = logvar_q.exp().mean()
    _z_var_std    = logvar_q.exp().std()
    _z_sample_var = z_star.var(dim=0).mean()
```

JSONL에 `train/_z_mu_var`로 자동 기록된다 (trainer가 scalar tensor를 전부 누적).
z_shuffle_gap과 같이 보면 "수축 시점"과 "binding 실패 시점"을 동시에 추적 가능하다.

---

## 8. 하지 말 것 (이 실험에서)

M9 실행 전후에 다음을 함께 바꾸지 않는다.

- FiLM 구조 추가 (M8_film.md의 계획)
- VQ bottleneck 추가 (M9_vq.md의 계획)
- InfoNCE temperature 변경
- semantic_weight 변경
- posterior encoder 구조 변경

이 실험은 변수 하나짜리다. 변수 추가 시 해석력이 사라진다.

FiLM / VQ는 M9 결과를 보고 나서 결정한다.
- z_mu_var가 올라가면 → representation이 살아났다 → 그다음에 binding(FiLM 등) 검토
- z_mu_var가 안 올라가면 → 다른 representation 강화 방법 먼저

---

## 9. 기존 M8_film.md / M9_vq.md와의 관계

`M8_film.md`와 `M9_vq.md`는 M8(`s1only`) 결과 전에 설계된 계획이다.
판결 실험 이후 방향이 바뀐 이유를 정리하면:

| 이전 계획 | 전제 | 실제 관측 | 판단 |
|-----------|------|----------|------|
| FiLM z-Modulation | decoder가 z를 안 써서 binding 강화 필요 | decoder는 posterior z를 씀 (`delta_null=0.225`) | 틀린 전제 |
| VQ-z Binding | z를 discrete하게 만들어 우회 차단 | 우회가 문제가 아니라 z 자체가 uninformative | 순서가 틀림 |

FiLM과 VQ가 필요 없다는 게 아니다.
z-space representation이 먼저 살아나야 binding 강화가 의미를 갖는다는 뜻이다.
**representation이 먼저, binding은 그 다음.**

---

## 10. 학습 커맨드

```bash
python -m torch.distributed.run --nproc_per_node=1 scripts/train_vlm.py \
    --config configs/vlm_paligemma_infonce_s2also.yaml
```

---

## 11. 결과

> 학습 미실행

---

## 12. 저장 파일

```
outputs/runs/vlm_sfp_infonce_s2also_20260421/
├── ckpt_10.pt ~ ckpt_100.pt
├── train.log
├── train_log.jsonl        ← train/_z_mu_var 포함
└── result.md
```
