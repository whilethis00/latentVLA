# LatentVLA Idea Bank

> 브레인스토밍 문서. 코드 구현 없음. 아이디어 본질 + 장단점 + 우선순위 정리.
> 업데이트: 2026-04-16

---

## 핵심 진단 (모든 아이디어의 출발점)

> **우리 문제는 representation 문제가 아니라 interface + routing + causality 문제다.**

- z가 "좋은 정보를 담는가"보다 **"decoder가 z 없이 살 수 있는가"** 가 더 근본적인 질문
- 지금까지의 실패(M6/M7)는 모두 z를 optional로 남겨둔 결과
- 해결 방향: z를 mandatory하게 만드는 것 → 방법이 다를 뿐

---

## 아이디어 우선순위 요약

| 순위 | 아이디어 | 시점 | 이유 |
|:----:|---------|:----:|------|
| 1 | **Classifier-Free Guidance (z dropout)** | 지금 (M8에 추가) | 구현 5줄, 리스크 없음, 효과 직접적 |
| 2 | **Soft VQ / Prototype** | M8 후 (M9-1) | 이미 설계됨, VQ 없는 soft binding |
| 3 | **MoE (z → expert routing)** | M9 후 (M10) | VQ 성공 시 자연스러운 확장 |
| 4 | **World Model (predictive z)** | 중기 | semantic loss 강화 버전, 강력함 |
| 5 | **Hard VQ** | M9 후 (M9-2) | discrete binding 완성, prior 재설계 필요 |
| 6 | **Information Bottleneck** | 후기 | 이론적으로 제일 강하지만 아키텍처 전반 변경 |
| 7 | **Causal Representation** | 측정 방법으로 이미 사용 중 | 실험 철학 |
| 8 | **Option Learning / Hierarchical** | 후기 | 방향 맞지만 현재 문제 해결 후 |
| 9 | **Slot Attention** | 후기 | 과설계, 현재 단계 아님 |

---

## 1. Classifier-Free Guidance (z dropout)

### 본질
Stable Diffusion의 CFG에서 차용.
학습 시 일정 확률로 z를 null로 대체 → decoder가 "z 있을 때"와 "없을 때"를 모두 학습.
결과적으로 z 유무에 따른 차이가 명시적으로 학습됨.

```
p=0.1로 z를 zeros/learnable_null로 대체
→ decoder: "z 없으면 평균적인 행동, z 있으면 task-specific 행동"
→ z sensitivity가 loss 추가 없이 자연스럽게 생김
```

### 왜 저평가됐냐
"dropout이랑 비슷한 거 아냐?" → 아님.
일반 dropout은 z의 일부 차원을 끄는 것.
CFG dropout은 z 전체를 null로 바꿈 → decoder가 아예 z 없는 조건도 경험.
이 차이가 z dependency를 만드는 핵심.

### 장점
- 구현 5줄 (학습 루프에서 `if random() < p: z = null_z`)
- 기존 아키텍처 변경 없음
- M8(FiLM)과 즉시 결합 가능
- 추론 시 guidance scale로 z 영향력 조절 가능 (`z_guided = z_null + w*(z - z_null)`)
- 부작용 거의 없음

### 단점
- null_z 설계 선택 필요 (zeros vs learnable vector)
- drop probability p 튜닝 필요
- 단독으로는 z의 정보량을 늘리지는 않음 → FiLM/VQ와 함께 써야 제대로 작동

### 우선순위: ★★★★★
M8 구현 시 같이 넣어야 함. 안 넣을 이유가 없음.

---

## 2. Soft VQ / Prototype Learning

### 본질
Metric Learning + VQ-VAE의 중간.
continuous z를 유지하되 codebook prototype으로 당기는 commitment loss 추가.
z가 "흐릿한 연속 공간"이 아니라 "prototype 근처에 모이는 구조화된 공간"에 있게 됨.

### 왜 중요한가
M6/M7에서 z_shuffle_gap이 붕괴하는 이유 중 하나:
z가 연속 공간에서 점점 퍼지거나 한 점으로 몰림.
prototype이 있으면 z가 갈 수 있는 곳이 제한됨 → collapse 방향 차단.

### 장점
- prior flow 구조 유지 (Hard VQ처럼 prior 재설계 불필요)
- straight-through로 gradient 흐름 유지
- codebook이 task prototype을 자연스럽게 학습할 가능성
- dead code 문제가 Hard VQ보다 완화됨

### 단점
- commitment weight λ 튜닝 필요
- prototype saturation (모두 한 점으로) 가능성
- 여전히 "완전한 hard binding"은 아님

### 우선순위: ★★★★☆
M9-1로 이미 설계됨. M8 결과 확인 후 진행.

---

## 3. Mixture of Experts (MoE)

### 본질
z가 "정보를 담는 벡터"가 아니라 **"어떤 전문가 decoder를 쓸지 결정하는 선택 신호"**.

```
z → gating network → expert weights (K개)
action = Σ expert_k(context) * gate_k(z)
```

또는 hard routing:
```
z → argmax → expert_k
action = expert_k(context)
```

### VLA에서의 번역
- task마다 optimal action pattern이 다름 (pick vs place vs push)
- 하나의 decoder가 모든 mode를 평균화하려 함 → z 무시해도 됨
- expert가 분리되면 z가 "어느 expert를 쓸지" 결정해야만 함 → mandatory

### 장점
- z가 routing signal이 되는 순간 구조적으로 mandatory
- expert별로 task-specific pattern 학습 가능
- 해석 가능성 높음 ("z=3번 expert = pick task")
- VQ + MoE 결합 시 z=codebook index=expert index로 깔끔하게 통일

### 단점
- expert 수 K 설계 필요 (task 수 10개 → K=10? 16? 32?)
- load balancing 필요 (특정 expert만 쓰이는 collapse)
- 학습 초기 불안정 (모든 expert가 비슷하게 시작)
- 지금 action flow (VelocityMLP) 전체를 K개로 복제 → 파라미터 K배

### 우선순위: ★★★★☆
VQ 방향 확인 후 자연스러운 다음 스텝. M10 후보.

---

## 4. World Model / Predictive Coding

### 본질
Dreamer, DreamerV3에서 차용.
**좋은 plan이라면 미래를 잘 예측해야 한다.**
z를 단순 feature가 아니라 **future state rollout을 설명하는 predictive variable**로 만들기.

현재 semantic future loss는 SigLIP 이미지 feature cosine similarity만 봄 → 너무 약함.

강화 방향:
```
z → predict: [future_image_feat, future_proprio, future_lang_alignment]
loss = MSE(z_pred_future_state, actual_future_state)
```

더 강하게 가면:
```
z + current_state → rollout K steps → predicted_states
loss = Σ MSE(predicted_state_t, actual_state_t)
```

### 왜 강한가
z가 미래를 예측해야 하면 task-discriminative 정보를 담을 수밖에 없음.
"pick apple"과 "open drawer"는 미래 state가 완전히 다름.
현재 semantic loss만으로는 이미지 feature 수준만 구분 → proprio/state 수준 구분이 안 됨.

### 장점
- z가 "좋은 plan = 미래를 잘 설명"이라는 직접적인 학습 신호
- task discrimination이 자연스럽게 z에 인코딩됨
- 논문 스토리가 강함 ("predictive plan variable")
- LIBERO 데이터에 proprio sequence가 있으므로 추가 데이터 수집 불필요

### 단점
- future proprio prediction head 추가 필요
- rollout 방식이면 데이터 구조 변경 (multi-step label)
- semantic future loss와의 weight 밸런싱 복잡
- prediction error가 z 학습을 지배할 수 있음

### 우선순위: ★★★★☆
semantic future loss를 proprio 예측으로 확장하는 건 지금도 가능. 중기.

---

## 5. Hard VQ (Discrete Binding)

### 본질
z를 완전히 이산 코드로 만들기. 이미 M9-2로 설계됨.

### 요약
- 가장 강한 binding
- prior 아키텍처 재설계 필요 (flow → categorical)
- dead code, codebook collapse 리스크

### 우선순위: ★★★☆☆
M9-1 (Soft VQ) 확인 후 진행. 단독으로 점프는 리스크 큼.

---

## 6. Information Bottleneck

### 본질
Tishby의 Information Bottleneck 이론에서 차용.
목표: `z가 task에 필요한 정보만 담고, 나머지는 버리게 강제`

```
maximize: I(z; task)    ← task 정보는 유지
minimize: I(z; state)   ← 불필요한 state 정보는 버림
```

VLA 적용:
- state/image 전체를 action flow에 직접 주지 않음
- **z를 반드시 통과해야만 task 정보가 action flow에 도달**
- context encoder의 출력을 z 없이는 action decoder에 못 쓰게 구조 변경

### 왜 제일 근본적인가
지금 구조는 `action_flow(context, z)`인데 context에 이미 task 정보가 있음.
z가 없어도 context만으로 task를 구분할 수 있음 → z optional.
Information Bottleneck은 이 context 경로를 강제로 좁혀서 z를 필수화.

### 장점
- 이론적으로 가장 깨끗한 해법
- "z가 task의 충분통계량"이라는 강한 주장 가능
- 논문 contribution으로 제일 강력

### 단점
- 아키텍처 전반 변경 필요 (context를 z로만 전달하도록)
- action quality 하락 가능 (context 정보 손실)
- IB 구현 자체가 어려움 (MINE, VIB 등 별도 estimator 필요)
- 안정적인 학습이 어려움

### 우선순위: ★★★☆☆
후기. VQ/FiLM 방향이 검증된 후 논문의 이론적 프레임으로 활용 가능.

---

## 7. Causal Representation Learning

### 본질
z가 task의 상관 feature인가, 아니면 **원인 변수(causal variable)**인가.

```
진짜 causal z: z를 바꾸면 action이 바뀜
단순 correlation: z와 action이 같이 움직이지만 z→action 인과 없음
```

### VLA 적용
- z를 task A → task B로 교체 → action chunk가 실제로 바뀌어야 함
- z를 zero로 대체 → 성능이 하락해야 함
- 같은 task, 다른 state → z가 비슷해야 함

### 현재 상태
이미 실험 측정 방법으로 사용 중 (z intervention test).
별도 아키텍처 아이디어라기보다 **실험 철학이자 검증 프레임**.

### 우선순위: ★★★★★ (측정 방법으로)
아키텍처 변경은 아니지만 모든 실험의 평가 기준으로 써야 함.

---

## 8. Option Learning / Hierarchical Policy

### 본질
Sutton의 Options Framework에서 차용.
z를 "한 번 뽑고 끝나는 feature"가 아니라 **"subpolicy를 지정하는 option key"**로.

```
high-level: z = option_key (어떤 subpolicy를 쓸지)
low-level: subpolicy_z(state) → action chunk
```

현재 M시리즈와의 관계:
- 지금: z → action flow (z가 step마다 같은 조건으로 들어감)
- Option: z가 subpolicy를 결정 → subpolicy 내에서는 z 없이 실행

### 왜 말이 되는가
LIBERO task는 "pick object → move → place" 같은 subgoal 구조.
z가 "어떤 subgoal 시퀀스를 실행할지" 결정하면 훨씬 의미 있는 plan variable이 됨.

### 장점
- 계층 구조가 로봇 task의 자연스러운 분해와 맞음
- z의 해석 가능성 높음
- long-horizon generalization에 유리

### 단점
- 지금 문제(z optional)가 해결 안 된 상태에서 계층 추가는 과함
- subgoal 정의, 종료 조건 설계 복잡
- 학습 난이도 크게 증가

### 우선순위: ★★☆☆☆
방향은 맞지만 현재 단계에서 과설계. z mandatory 문제 해결 후.

---

## 9. Slot Attention / Object-centric z

### 본질
Locatello의 Slot Attention에서 차용.
z 하나에 모든 걸 욱여넣지 말고, **역할별로 분리된 z 여러 개** 사용.

```
z_task   : 어떤 task인가
z_object : 어떤 object를 다루는가
z_motion : 어떤 motion 패턴인가
```

### 장점
- 각 z가 담당 역할이 명확 → collapse 방향 분리
- 해석 가능성 높음
- generalization 이론적으로 강함

### 단점
- 지금 z 하나도 제대로 못 쓰는데 여러 개로 쪼개는 건 문제 회피
- 학습 난이도 급증
- 어떻게 분리할지 supervision 없음

### 우선순위: ★★☆☆☆
후기. z mandatory 문제 해결 + VQ 방향 검증 후.

---

## 연결 구조 (아이디어 간 관계)

```
현재 문제: z optional
    │
    ├─ 구조적 해결
    │   ├─ M8: FiLM (soft binding)
    │   ├─ CFG z-dropout ← 지금 당장 M8에 추가
    │   ├─ M9-1: Soft VQ (commitment)
    │   └─ M9-2: Hard VQ (discrete)
    │           │
    │           └─ MoE (z → expert routing) ← VQ 후 자연스러운 확장
    │
    ├─ 정보 구조 해결
    │   ├─ Information Bottleneck (z만 통과하게 강제)
    │   ├─ World Model (z가 미래 예측)
    │   └─ Slot Attention (z 역할 분리)
    │
    └─ 측정/철학
        ├─ Causal Representation (z intervention test)
        └─ Option Learning (z = subpolicy key, 장기 목표)
```

---

## 다음 액션

1. **M8 구현 시 CFG z-dropout 같이 추가** (p=0.1)
2. **M8 완료 후** z intervention test + per-block alpha 확인
3. **M9-1 (Soft VQ)** — M8 검증 후
4. **World Model 강화** (semantic loss → proprio prediction) — 중기
5. **MoE** — VQ 검증 후
