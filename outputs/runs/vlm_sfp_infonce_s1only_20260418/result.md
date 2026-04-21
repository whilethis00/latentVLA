# vlm_sfp_infonce_s1only_20260418 실험 결과

- **날짜**: 2026-04-18 ~ 2026-04-21
- **모델**: vlm_sfp_plan (PaliGemma-3b-pt-224 기반)
- **에포크**: 92/100 완료 (PBS walltime 초과로 epoch 93 도중 중단)
- **마지막 체크포인트**: ckpt_90.pt
- **현황**: 미완료 (walltime 259249s > 한계 259200s, 49초 초과)

### 설정

| 항목 | 값 |
|------|-----|
| z_form | plan |
| batch_size | 16 (x4 grad accum = 64 eff) |
| world_size | 1 (GPU 1장) |
| train / val | 56,578 / 6,959 |
| Stage 2 시작 | epoch 10 |
| InfoNCE 적용 | O |
| 데이터 | **S1 only** (Stage 1 demonstrations만 사용) |

---

## 무엇을 검증하나

이전 실험(`vlm_sfp_infonce_balanced_20260416`)은 S1+S2 혼합 데이터를 사용했다.  
이번 실험은 **S1 demonstrations만** 사용해 학습했을 때 action flow / prior 품질이 어떻게 달라지는지 확인한다.  
가설: S2(다단계) 데이터 없이 S1(단일 단계)만으로도 충분한 prior 표현력을 얻을 수 있는가?

---

## 학습 손실 곡선 (주요 epoch)

| Epoch | total_loss | action_flow | prior_flow | semantic |
|-------|-----------|-------------|------------|----------|
| 1     | 2.3524    | 1.0992      | 1.0586     | 0.0717   |
| 10    | 0.7030    | 0.5246      | 0.1769     | 0.0157   |
| 20    | 0.5355    | 0.4232      | 0.1118     | 0.0053   |
| 30    | 0.4601    | 0.3391      | 0.1206     | 0.0040   |
| 40    | 0.4124    | 0.2792      | 0.1328     | 0.0035   |
| 50    | 0.3842    | 0.2429      | 0.1410     | 0.0032   |
| 60    | 0.3646    | 0.2157      | 0.1486     | 0.0030   |
| 70    | 0.3475    | 0.1940      | 0.1533     | 0.0029   |
| 80    | 0.3359    | 0.1793      | 0.1564     | 0.0029   |
| 90    | 0.3293    | 0.1706      | 0.1585     | 0.0028   |
| 92    | 0.3249    | 0.1685      | 0.1561     | 0.0028   |

> **주목**: prior_flow_loss가 epoch 20 (0.1118) 이후 오히려 증가해 epoch 90에 0.1585로 상승.  
> action_flow_loss는 꾸준히 감소. 두 loss 간 역방향 추세 발생.

---

## 검증 지표 (Val)

| Epoch | mse_prior | mse_posterior | pp_gap | best_of_1 | best_of_5 | cosine_sim | z_shuffle_gap |
|-------|-----------|---------------|--------|-----------|-----------|------------|---------------|
| 5     | 1.1448    | 1.0955        | 0.0493 | 1.1579    | 0.5399    | 0.9718     | -0.0092       |
| 10    | 0.8283    | 0.7949        | 0.0333 | 0.8354    | 0.3981    | 0.9869     | 0.0140        |
| 15    | 0.7117    | 0.6405        | 0.0712 | 0.7173    | 0.3571    | 0.9930     | 0.0183        |
| 20    | 0.6564    | 0.5974        | 0.0590 | 0.6608    | 0.3499    | 0.9947     | 0.0110        |
| 25    | 0.6111    | 0.5453        | 0.0658 | 0.6168    | 0.3582    | 0.9954     | 0.0170        |
| 30    | 0.5916    | 0.5143        | 0.0773 | 0.5945    | 0.3734    | 0.9959     | 0.0143        |
| 35    | 0.5903    | 0.5090        | 0.0813 | 0.5924    | 0.4007    | 0.9962     | 0.0069        |
| 40    | 0.5517    | 0.4862        | 0.0655 | 0.5546    | 0.3891    | 0.9964     | 0.0129        |
| 45    | 0.5497    | 0.4849        | 0.0649 | 0.5517    | 0.4027    | 0.9966     | 0.0060        |
| 50    | 0.5387    | 0.4757        | 0.0631 | 0.5407    | 0.4063    | 0.9967     | 0.0106        |
| 55    | 0.5376    | 0.4723        | 0.0653 | 0.5397    | 0.4118    | 0.9968     | 0.0032        |
| 60    | 0.5216    | 0.4648        | 0.0568 | 0.5243    | 0.4101    | 0.9969     | 0.0062        |
| 65    | 0.5230    | 0.4673        | 0.0556 | 0.5255    | 0.4152    | 0.9969     | 0.0053        |
| 70    | 0.5155    | 0.4637        | 0.0517 | 0.5174    | 0.4173    | 0.9970     | 0.0035        |
| 75    | 0.5063    | 0.4589        | 0.0474 | 0.5082    | 0.4157    | 0.9970     | 0.0092        |
| 80    | 0.5110    | 0.4623        | 0.0487 | 0.5132    | 0.4234    | 0.9970     | 0.0061        |
| 85    | 0.5113    | 0.4607        | 0.0505 | 0.5117    | 0.4252    | 0.9971     | 0.0092        |
| 90    | 0.5120    | 0.4606        | 0.0514 | 0.5109    | 0.4257    | 0.9971     | 0.0086        |

---

## 최종 Val 지표 (epoch 90 기준)

| 지표 | 값 |
|------|-----|
| action_mse_prior | 0.5120 |
| action_mse_posterior | 0.4606 |
| prior_posterior_gap | 0.0514 |
| best_of_1 | 0.5109 |
| best_of_5 | 0.4257 |
| future_cosine_sim | 0.9971 |
| z_shuffle_gap | 0.0086 |

---

## 이전 실험 대비 비교

| 지표 | infonce_20260414 | infonce_balanced_20260416 | **s1only_20260418** |
|------|-----------------|--------------------------|---------------------|
| mse_prior (ep90) | - | 0.5723 | **0.5120** |
| mse_posterior (ep90) | - | 0.5303 | **0.4606** |
| best_of_5 (ep90) | - | 0.3907 | 0.4257 |
| pp_gap (ep90) | - | 0.0420 | 0.0514 |
| z_shuffle_gap (ep90) | - | 0.0047 | 0.0086 |

---

## 결과 해석 및 인사이트

### 긍정적
- **mse_prior, mse_posterior 모두 balanced 대비 향상** (0.5120 vs 0.5723, 0.4606 vs 0.5303): S1 only 데이터가 더 깔끔한 action distribution을 갖는 것으로 보임
- future_cosine_sim이 0.9971로 수렴: semantic future prediction은 안정적으로 동작
- 학습이 꾸준히 수렴 중 (총 loss 계속 감소, plateau 없음)

### 주의
- **best_of_5가 balanced보다 나쁨** (0.4257 vs 0.3907): prior의 **다양성(diversity)** 이 부족. S1 only라 action 분포 자체가 좁아진 것으로 추정
- **prior_flow_loss가 epoch 20 이후 역증가** (0.1118 → 0.1585): action_flow와 prior_flow 간 gradient 경쟁 가능성. 또는 posterior가 빠르게 좁아지면서 prior가 따라가기 어려워진 것
- **z_shuffle_gap이 0.0086으로 매우 작음**: z(language plan)가 action에 실질적으로 기여하는 정도가 낮음. InfoNCE가 충분히 z를 활용하지 못하고 있을 수 있음
- pp_gap(0.0514)이 balanced(0.0420)보다 큼: posterior 정보가 prior와 차이를 더 많이 만들어야 하는 상황 → prior가 아직 부족

### 종합
S1 only 데이터는 평균 MSE(prior/posterior) 측면에서는 유리하지만 prior 다양성이 부족하다.  
z_shuffle_gap이 작아 language conditioning 효과가 약한 것이 주요 한계로 보인다.

---

## 다음 스텝

- [ ] 100 epoch 완주를 위해 PBS walltime 늘려서 ckpt_90.pt에서 resume
- [ ] z_shuffle_gap 개선 방법 탐색 (InfoNCE temperature 조정, z conditioning 강화)
- [ ] best_of_5 개선을 위한 prior diversity 향상 방안 (S1+S2 혼합 or temperature sampling)
- [ ] balanced vs s1only를 동일 epoch에서 직접 비교 (ep90 체크포인트 기준 rollout eval)

---

## M8 이후 연구 방향 재정립 (2026-04-21)

> 이 섹션은 M8 결과를 보고 난 뒤 연구 방향을 다시 잡는 과정을 기록한 것이다.
> 나중에 방향을 잃지 않기 위해, 어떤 생각의 흐름을 거쳐 다음 실험을 정했는지 남겨둔다.

### 왜 막혔는가 — 표면적 이유 vs 진짜 이유

표면적으로는 "M9를 어떻게 설계할까"가 막힌 것처럼 보인다.
그런데 실제로 막힌 이유는 따로 있다.

**연구 질문이 두 개 섞여 있었다.**

지금까지 실험들은 겉으로는 "prior MSE를 낮추는 법"을 찾는 것처럼 움직였다.
그런데 실제로는 전혀 다른 두 가지 문제를 동시에 건드리고 있었다.

1. **좋은 latent plan을 만들 수 있는가** — 즉 posterior z가 의미 있는 미래 정보를 담고 있는가
2. **만든 latent plan을 policy가 실제로 쓰는가** — 즉 action decoder가 z에 의존해서 행동을 만드는가

이 두 가지가 섞이면, MSE가 좋아져도 "왜 좋아졌는지"를 알 수가 없다.
M8도 그랬다. `mse_prior=0.5120`, `mse_posterior=0.4606`으로 balanced 대비 분명히 나아졌는데,
`z_shuffle_gap=0.0086`은 끝까지 낮고, `prior_flow_loss`는 epoch 20 이후 다시 올라갔다.
숫자는 좋아졌지만, **왜 좋아졌는지는 여전히 불분명**하다.
그리고 정확히 같은 패턴이 M7에서도 반복됐다.

---

### 진짜 핵심 주장을 다시 정의해야 한다

이 연구가 논문이 되려면, 증명해야 할 핵심 문장이 하나 있다.

> **"latent plan이 action generation에 causally necessary하다."**

이 문장을 풀어쓰면 이렇다:
- 로봇이 행동을 생성할 때, 단순히 현재 이미지와 언어를 보는 것으로는 부족하고
- "어떤 계획을 세울 것인가"를 나타내는 latent vector z가 행동에 **인과적으로** 개입해야 한다
- z를 바꾸면 행동이 바뀌어야 하고, z를 제거하면 성능이 떨어져야 한다

이 문장 하나가 살아 있으면, 그 다음에 선택하는 기법들(FiLM이든 VQ든 CFG든)이 모두 의미를 갖는다.
반대로 이 문장이 안 서면, 어떤 기법을 얹어도 그냥 "surface metric을 움직이는 트릭"에 불과해진다.

---

### 지금 어디에 있는가 — 세 가지 가능한 세계

M8 데이터를 보면, 이 핵심 주장이 아직 서 있지 않다.
그 이유를 두 갈래로 나눌 수 있다.

**세계 A: z는 퍼져 있는데 decoder가 무시한다 (binding failure)**

posterior encoder는 제대로 된 z를 만들고 있다.
z를 보면 샘플마다 다르고, task마다 다르고, 미래에 따라 다른 z가 나온다.
그런데 action flow (decoder)는 z를 condition으로 받으면서도 실제로 z에 의존하지 않도록 학습됐다.
z가 있든 없든, 어떤 z든 비슷한 행동을 출력한다.
이건 마치 두 사람이 협업하는데, 한 명이 아무리 좋은 계획을 짜줘도 다른 한 명이 그 계획을 안 보는 상황이다.

**세계 B/C: z 자체가 수축했다 (posterior collapse / z-space contraction)**

posterior encoder가 만드는 z가 샘플마다 거의 똑같다.
즉 어떤 상황이든, 어떤 미래가 와도, z가 거의 같은 값을 출력한다.
이 경우 decoder가 z를 무시하는 건 합리적이다 — z에 아무 정보가 없으니까.
이건 마치 계획을 짜는 사람이 항상 "그냥 대충 하면 됩니다"라는 동일한 계획만 내놓는 상황이다.
decoder가 그 계획을 무시해도 행동이 안 달라지는 게 당연하다.

지금까지 로그를 보면, **세계 C (collapse + non-usage 동시 발생)** 가 가장 가능성 높다.
`z_shuffle_gap`이 8 epoch 내내 0에 가깝다는 건 "decoder가 z를 구분하지 않는다"는 직접 증거다.
그런데 그 이유가 세계 A인지 B/C인지는 아직 모른다.

---

### Claim 2는 이미 깨졌다 — 이건 새로 증명할 필요가 없다

위의 두 가지 주장을 다시 쓰면:

- **Claim 1**: posterior z는 실제로 의미 있게 퍼져 있다 (각 상황마다 다른 z가 나온다)
- **Claim 2**: decoder는 그 z를 실제로 사용한다 (z를 바꾸면 행동이 바뀐다)

`z_shuffle_gap = 0.0086`이 이미 말해주는 것이 있다.

`z_shuffle_gap`이 어떻게 계산되는지 설명하면:
1. 같은 val 배치에서 prior flow로 z를 생성한다
2. 정상적으로 z와 context를 붙여서 action을 예측하고 MSE를 잰다
3. z를 배치 내에서 셔플(다른 샘플의 z와 교체)하고 다시 action MSE를 잰다
4. `z_shuffle_gap = MSE(셔플) - MSE(정상)`

만약 z가 실제로 action에 기여하고 있다면, 셔플했을 때 성능이 확 나빠져야 한다.
즉 `z_shuffle_gap`이 크게 양수여야 한다.
그런데 M8에서는 `0.0086`이다. 거의 0이다.
z를 다른 사람 것으로 바꿔줘도 예측 성능이 거의 안 변한다.

이건 **"decoder가 z를 실질적으로 사용하지 않는다"는 관측이 이미 있다**는 뜻이다.
Claim 2는 새로 실험해서 확인할 대상이 아니라, 이미 반례가 관측된 상태다.

---

### 그러면 진짜 미지수는 하나다

Claim 2가 왜 깨졌는가.

이걸 가르는 최소 판별 변수가 **`z_mu_var_mean`** 이다.

`z_mu_var_mean`이 뭔지 설명하면:
- posterior encoder는 각 입력에 대해 `mu_q` (z의 평균값)와 `logvar_q` (z의 분산 로그)를 출력한다
- `mu_q`는 해당 샘플에서 "z가 어디쯤에 있는가"를 나타내는 벡터다
- `mu_q.var(dim=0).mean()`은 "val 배치 전체에서 샘플마다 mu_q가 얼마나 다른가"를 측정한다
- 이 값이 크면 → posterior가 샘플마다 다른 z를 만들고 있다 (퍼져 있다)
- 이 값이 작으면 → posterior가 어떤 샘플이든 거의 같은 z를 만들고 있다 (수축했다)

| `z_mu_var_mean`이 낮으면 | `z_mu_var_mean`이 높으면 |
|--------------------------|--------------------------|
| z-space contraction 가능성 높음 | global collapse는 아닐 가능성 높음 |
| decoder가 z를 무시하는 게 합리적 | binding failure 쪽으로 기울어짐 |
| → posterior/encoder 강화 먼저 | → FiLM/CFG 같은 conditioning 강화 먼저 |

이 숫자 하나가 다음 방향을 결정한다.

단, `z_mu_var_mean`이 높다고 해서 "posterior는 완전히 괜찮다"는 뜻은 아니다.
높은 variance가 task/미래와 무관한 noise일 수도 있기 때문이다.
그래서 이 숫자는 **"다음 실험 방향을 정하는 1차 판별자"** 로 쓰되,
최종 확정 진단으로는 쓰지 않는다.

---

### 중요: `z_mu_var_mean`은 학습 로그에 한 번도 찍힌 적이 없다

`compute_loss`에서 `_z_mu_norm`, `_z_var_mean`, `_z_var_std`, `_z_sample_var`는 반환하고 있지만,
`train_log.jsonl`을 보면 그 키들이 없다.
즉 trainer_vlm.py가 해당 필드를 JSONL에 기록하지 않고 있다.

그 말은: **ep1 ~ ep92 동안 이 숫자가 어떻게 변해왔는지 궤적이 아예 없다.**

지금 ckpt_90.pt 스냅샷에서 한 번 뽑는 건 맞지만,
- 이 숫자가 처음부터 낮았는지
- 학습 중간에 떨어진 건지
- 아니면 학습 내내 높은데 decoder만 안 쓰는 건지

이건 현재 로그만으로는 알 수 없다.

따라서 다음 실험(M9든 뭐든)부터는 **`z_mu_var_mean`을 train loop에서 주기적으로 기록해야 한다.**
z_shuffle_gap과 함께 epoch마다 찍어놓으면, 언제 z-space가 수축하는지 궤적을 볼 수 있다.

---

### M8에서 배운 것과 아직 모르는 것

**이미 아는 것 (관측 사실)**

- `z_shuffle_gap`이 전 epoch 구간에서 0에 가깝다 → decoder가 z를 구분하지 않는다 (Claim 2 실패)
- `prior_flow_loss`가 epoch 20 이후 역증가한다 → prior flow가 posterior를 따라가기 어려워지고 있다
- `mse_prior`, `mse_posterior`는 balanced 대비 개선됐다 → action 예측 자체는 나아졌다
- `best_of_5`는 balanced보다 나쁘다 → prior의 diversity는 떨어졌다

**아직 모르는 것 (미지수)**

- posterior z가 실제로 의미 있게 퍼져 있는가 (`z_mu_var_mean` 미측정)
- z-space contraction이 학습 초반부터 발생했는가, 아니면 특정 epoch 이후 무너진 것인가
- Claim 2 실패의 원인이 "z에 정보가 없어서"인가, "decoder가 z를 학습적으로 무시하게 됐는가"인가

---

### 다음 한 단계 (판결 전 예상)

M9 설계를 먼저 하지 않는다.
FiLM, VQ, CFG, temperature 튜닝 — 이것들을 먼저 올리지 않는다.

지금 당장 해야 할 것은 딱 하나다.

**M8 `ckpt_90.pt`로 val 한 번 돌려서 `z_mu_var_mean` 하나를 뽑는다.**

```
z_mu_var_mean = mu_q.var(dim=0).mean()
```

이 숫자가 낮으면 → z-space 먼저 살리는 방향으로 M9 설계  
이 숫자가 높으면 → binding 강화 방향으로 M9 설계

---

### 판결 실험 결과 (2026-04-21, ckpt_90.pt, max_batches=50, N=800)

실행 커맨드:
```bash
conda run -n vla python3 scripts/eval_z_diag.py \
    --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
    --max_batches 50
```

#### 결과 수치

| 지표 | 값 | 의미 |
|------|-----|------|
| `z_mu_var_mean` | **0.0345** | posterior z의 배치 내 분산 (0.1 이하 = 수축) |
| `z_sample_var`  | 0.0345    | z 샘플의 분산 (mu_var와 거의 동일 → noise 기여 미미) |
| `delta_null`    | **+0.2255** | posterior z를 zeros로 교체 시 MSE 상승량 |
| `delta_shuffle` | +0.1112   | posterior z를 배치 내 셔플 시 MSE 상승량 |
| `probe_ratio`   | 0.9198    | same-task / random 거리 비율 (1에 가까울수록 task 구분 없음) |

참고: `z_shuffle_gap`(= 0.0086)은 **prior z** 셔플 기준이고, `delta_null`/`delta_shuffle`은 **posterior z** 교체 기준이다. 두 숫자가 측정하는 경로가 다르다.

#### 판정: 케이스 C

> posterior 약함, decoder는 z 쓰려 함 → **posterior/encoder 강화 우선**

---

#### 결과 해석

세 숫자를 같이 읽어야 한다.

**posterior → decoder 경로는 살아 있다.**

`delta_null = +0.2255`는 posterior z를 zeros로 바꿨을 때 MSE가 0.225나 오른다는 뜻이다.
이건 decoder가 posterior z를 실제로 사용하고 있다는 직접 증거다.
이전에 `z_shuffle_gap`만 보고 "decoder가 z를 안 쓴다"고 해석한 것은 틀렸다.

**prior → decoder 경로는 죽어 있다.**

`z_shuffle_gap = 0.0086`은 prior flow가 뽑은 z를 셔플해도 MSE가 거의 안 변한다는 뜻이다.
inference time에 prior로 샘플링한 z는 decoder가 거의 무시한다.
즉 문제는 decoder가 z를 안 쓰는 게 아니라, **prior z가 decoder가 기대하는 공간에 없다**는 것이다.

**posterior z 자체는 충분히 퍼지지 않았다 (수축 의심, 단 확정은 아님).**

`z_mu_var_mean = 0.0345`는 낮은 편으로 보이고 contraction 의심을 강하게 준다.
단 baseline(정상 범위가 어느 수준인지)이 없으므로, 이 숫자 하나만으로 "거의 다 같다"를 확정하기는 이르다.

대신 `probe_ratio = 0.9198`과 같이 읽으면 해석이 강해진다.
probe_ratio가 1에 가깝다는 건, 같은 task 내 샘플들의 z 거리가 random pair와 거의 같다는 뜻이다.
즉 z가 task나 미래 정보에 따라 다르게 인코딩되지 않고 있다.
이 두 숫자를 합치면, **posterior z의 future-separation이 약하다**는 해석은 충분히 강하다.

#### 왜 prior_flow_loss가 epoch 20 이후 역증가했는가 (가설)

다음은 가장 그럴듯한 설명이지만, 아직 가설이다. posterior 수축 외에 S2 data drift나 action_flow / prior_flow 간 loss competition이 섞였을 가능성도 배제할 수 없다.

**가설 (posterior 수축 → prior mismatch 경로):**

1. posterior z가 좁은 공간으로 수축한다
2. prior flow는 그 수축된 분포를 따라가야 하는데, 수축이 계속될수록 타겟 분포 자체가 불안정해진다
3. prior flow가 따라가기 점점 더 어려워지면서 loss가 다시 올라간다
4. inference time의 prior z는 posterior z가 있는 좁은 공간에서 벗어나게 되고, decoder는 그 prior z를 무시하는 게 합리적이 된다

이 가설이 맞는지 확인하려면, z_mu_var_mean을 train loop에 기록해서 posterior 수축이 실제로 epoch 20 전후에 발생했는지 봐야 한다.

**한 줄 요약:**
decoder는 posterior z를 쓰고 있지만, 그 z가 future/task에 따라 충분히 달라지지 않는다.
prior z는 그 posterior 공간을 inference time에 재현하지 못하고 있다.

---

#### 다음 실험 방향

posterior z가 수축한 게 주원인이므로, **z-space를 먼저 살리는 것**이 우선이다.
FiLM이나 CFG는 그 다음이다 — z가 펴지지 않으면 binding을 강화할 공간 자체가 없다.

구체적으로 검토할 방향:
- posterior z가 왜 수축하는가 — semantic_weight, InfoNCE temperature, posterior encoder capacity 점검
- z-space를 펴는 방법 — KL-like regularizer, z spread loss, contrastive objective 강화
- prior flow가 posterior를 더 잘 따라가게 하는 방법 — prior 학습 안정성 개선

그리고 다음 실험부터는 **`z_mu_var_mean`을 train loop에서 주기적으로 기록**해야 한다.
z_shuffle_gap과 함께 epoch마다 찍어두면 수축이 언제 시작되는지 궤적을 볼 수 있다.

---

### 하지 말 것 (방향 잃지 않기 위해)

- M9 바로 시작하지 말 것
- temperature 튜닝부터 하지 말 것
- FiLM 코드 먼저 넣지 말 것
- "이번엔 뭘 추가하면 좋아질까"로 생각 시작하지 말 것

지금은 **개선이 아니라 식별**이다.
원인이 무엇인지 모르는 채로 새 변수를 추가하면, 또 "뭔가 좋아졌는데 왜 좋아졌는지 모름"으로 끝난다.

---

### 한 문장 요약

> 현재 병목은 action 예측 정확도가 낮아서가 아니라,
> **latent action-planning channel이 인과적으로 작동하지 않기 때문**이다.
> 다음 실험의 목적은 성능 향상이 아니라,
> **그 원인이 z-space collapse인지 binding failure인지 식별하는 것**이다.

---

## 저장 파일

| 파일 | 내용 |
|------|------|
| `ckpt_10.pt` ~ `ckpt_90.pt` | 10 epoch 단위 체크포인트 |
| `train_log.jsonl` | Epoch별 전체 loss / val 지표 |
| `train.log` | 전체 학습 로그 (텍스트) |
