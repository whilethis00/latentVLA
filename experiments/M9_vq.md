# M9: VQ-z Binding (Soft → Hard)

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-16 |
| **베이스** | M8 (FiLM z-Modulation) 결과 확인 후 진행 |
| **목적** | z binding을 soft(FiLM)에서 hard(VQ)로 강화해 decoder의 z 우회를 구조적으로 차단 |
| **상태** | 🔴 설계 완료, M8 완료 후 진행 |

---

## 2. VQ-VAE에서 배워야 할 진짜 인사이트

### 핵심은 "이산이 좋다"가 아님

VQ-VAE의 진짜 교훈:

> **decoder가 latent를 우회하지 못하게 interface를 설계하라.**

VAE는 continuous z를 넘기기 때문에 decoder가 "평균 z"로 학습해도 loss가 내려감.
VQ-VAE는 z가 아니라 **index**를 넘김 → index는 평균낼 수 없음 → decoder가 특정 mode에 특화됨.

### 우리 문제에 번역하면

| VQ-VAE | VLA |
|--------|-----|
| image reconstruction mode | task execution mode |
| codebook entry | plan prototype |
| "어떤 code를 쓸 것인가" | "어떤 task를 실행할 것인가" |
| decoder가 code index에 특화 | action flow가 plan code에 특화 |

LIBERO는 task가 10개로 정해져 있음 → codebook이 task별 plan prototype을 자연스럽게 학습할 수 있는 조건.

### M8(FiLM)과의 관계

| | binding 방식 | prior 구조 | 리스크 |
|--|--|--|--|
| M8 FiLM | soft (modulation) | flow 유지 | constant modulation collapse |
| M9-1 Soft VQ | soft-hard (commitment) | flow 유지 | prototype saturation |
| M9-2 Hard VQ | hard (index) | classifier/categorical으로 변경 | dead code, prior 재설계 |

> M8 → M9-1 → M9-2 순서로 binding을 강화. 이전 단계 성공 확인 후 진행.

---

## 3. M9-1: Soft VQ (Commitment Loss + Prototype Clustering)

### 핵심 아이디어

continuous z는 유지하되, z가 **prototype 근처로 모이도록 강제**.
Hard VQ처럼 index를 강제하지는 않지만, codebook이 존재하고 z가 그 중 하나에 commit하도록 loss를 추가.

```
L_commit = ||z_encoder_output - sg(nearest_prototype)||²
L_total  = L_flow + λ_commit * L_commit
```

prior 구조는 기존 flow 유지 → 변수 최소화.

### 구조 변경

```
[현재 M8]
posterior → z_star (continuous) → FiLM modulation

[M9-1]
posterior → z_raw → quantize_soft(z_raw, codebook) → z_commit → FiLM modulation
                                  ↓
                         L_commit = ||z_raw - sg(nearest)||²
```

```python
class SoftVQBottleneck(nn.Module):
    def __init__(self, z_dim: int, num_codes: int = 64):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, z_dim)
        nn.init.uniform_(self.codebook.weight, -1/num_codes, 1/num_codes)

    def forward(self, z: torch.Tensor):
        # nearest prototype
        dists = torch.cdist(z.unsqueeze(1), self.codebook.weight.unsqueeze(0))
        indices = dists.squeeze(1).argmin(dim=-1)          # (B,)
        nearest = self.codebook(indices)                    # (B, z_dim)

        # commitment loss (encoder는 prototype 쪽으로, codebook은 encoder 쪽으로)
        loss_commit = F.mse_loss(z, nearest.detach())
        loss_codebook = F.mse_loss(nearest, z.detach())

        # straight-through: forward는 nearest, backward는 z 그대로
        z_q = z + (nearest - z).detach()

        return z_q, loss_commit + 0.25 * loss_codebook, indices
```

### Hyperparameter

| 파라미터 | 값 | 비고 |
|---------|:--:|------|
| `num_codes` | 64 | task 10개 → 여유있게 6배 |
| `commit_weight` | 0.1 | flow loss와 경합 방지 |
| `infonce_weight` | 0.0 | commitment이 clustering 담당 |
| prior, FiLM | M8와 동일 | — |

### 성공 기준

| 지표 | 목표 |
|------|:----:|
| z_shuffle_gap ↑ | > 0.054 (M6 ckpt_10 초과) |
| action_mse_prior ↓ | ≤ 0.608 |
| codebook usage | 10개 이상 code 활성화 (dead code < 50%) |
| task별 code 분포 | 같은 task → 같은 code cluster |

### 리스크

| 리스크 | 대응 |
|--------|------|
| prototype saturation (모든 z가 하나로 몰림) | codebook loss + commit weight 균형 확인 |
| commit loss가 flow loss와 경합 | λ_commit = 0.1부터 시작, sweep |
| prior flow가 clustered z 분포를 못 따라감 | prior flow 학습률 별도 조정 가능 |

---

## 4. M9-2: Hard VQ (Full Discrete Binding)

### M9-1과의 차이

M9-1은 z가 여전히 연속 공간에 있고 prototype 근처로 "당겨지는" 것.
M9-2는 z가 **반드시 codebook vector 중 하나**여야 함 → decoder가 완전히 index에 특화됨.

### 구조 변경

```
[M9-2]
posterior → z_raw → hard_quantize(z_raw) → z_q (= nearest codebook vector) → FiLM
prior     → categorical distribution over K codes → sample index → z_q
```

**Prior 구조 변경 필요:**
- 기존: `flow(noise, context) → z` (continuous)
- 변경: `MLP(context) → logits over K codes → argmax/gumbel-softmax → z_q`

```python
class VQPrior(nn.Module):
    """prior: context → categorical distribution over codebook"""
    def __init__(self, context_dim: int, num_codes: int, z_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, num_codes),
        )

    def forward(self, context: torch.Tensor, temperature: float = 1.0):
        logits = self.head(context)                          # (B, K)
        # training: gumbel-softmax (differentiable)
        if self.training:
            return F.gumbel_softmax(logits, tau=temperature, hard=True)
        # inference: argmax
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
```

### Prior 재설계 이유

Hard VQ에서 prior가 "어떤 code를 고를 것인가"를 학습해야 함.
flow prior(연속 분포)는 이 역할에 맞지 않음. categorical prior가 더 자연스러움.

### Hyperparameter

| 파라미터 | 값 | 비고 |
|---------|:--:|------|
| `num_codes` | 64 | ablation: 32 / 64 / 128 |
| `gumbel_temperature` | 1.0 → 0.1 annealing | 학습 후반에 hard selection으로 수렴 |
| `ema_update` | True | codebook EMA 업데이트 (안정성) |
| prior | VQPrior (MLP) | flow prior 대체 |

### 성공 기준

| 지표 | 목표 |
|------|:----:|
| z_shuffle_gap ↑ | > M9-1 수준 |
| action_mse_prior ↓ | ≤ 0.608 |
| codebook usage | dead code < 20% |
| task-code alignment | task당 dominant code 존재 |

### 리스크

| 리스크 | 대응 |
|--------|------|
| dead code | EMA codebook update + code restart (사용 안 되는 code 재초기화) |
| codebook collapse | codebook diversity loss 추가 |
| prior 재설계로 학습 불안정 | gumbel temperature annealing으로 점진적 이산화 |
| action quality 하락 (z 표현력 손실) | num_codes 늘리거나 residual VQ 고려 |

---

## 5. 구현 범위

### M9-1 구현 범위

| 파일 | 변경 내용 |
|------|----------|
| `models/vq_bottleneck.py` | `SoftVQBottleneck` 신규 작성 |
| `models/stoch_latent_flow_prior.py` | `compute_loss`에서 z_star에 SoftVQBottleneck 통과, `loss_commit` 반환 |
| `models/latent_vla.py` | `compute_loss`에서 `loss_commit` 수신 후 total loss에 추가 |
| `configs/vlm_paligemma_soft_vq.yaml` | M8 기반, `use_soft_vq: true`, `num_codes`, `commit_weight` 추가 |

`trainer_vlm.py`, `evaluate_offline_vlm.py` — **변경 없음**

#### `stoch_latent_flow_prior.py` 변경 포인트

```python
# __init__에 추가
if use_soft_vq:
    self.vq = SoftVQBottleneck(z_dim=z_dim, num_codes=num_codes)

# compute_loss 변경
z_star = self.reparameterize(mu_q, logvar_q)
loss_commit = torch.tensor(0.0, device=context.device)
if hasattr(self, 'vq'):
    z_star, loss_commit, _ = self.vq(z_star)   # z_star가 prototype 근처로 이동

# return에 추가
return {
    ...,
    "_loss_commit": loss_commit,   # latent_vla.py에서 total에 추가
}
```

#### `latent_vla.py` 변경 포인트

```python
# compute_loss에서 commit loss 수신
loss_commit = loss_dict.pop("_loss_commit", torch.tensor(0.0))
commit_weight = cfg.get("loss", {}).get("commit_weight", 0.1)
loss_dict["commit_loss"] = loss_commit
loss_dict["total_loss"] = loss_dict["total_loss"] + commit_weight * loss_commit
```

---

### M9-2 구현 범위

| 파일 | 변경 내용 |
|------|----------|
| `models/vq_bottleneck.py` | `HardVQBottleneck` 추가 (EMA codebook update 포함) |
| `models/vq_prior.py` | `VQPrior` 신규 작성 (MLP categorical prior) |
| `models/stoch_latent_flow_prior.py` | `prior_flow` → `VQPrior`로 교체 옵션 추가, `predict`에서 categorical sampling |
| `configs/vlm_paligemma_hard_vq.yaml` | M9-1 기반, `use_hard_vq: true`, `gumbel_temperature`, `ema_update` 추가 |

#### `HardVQBottleneck` (vq_bottleneck.py에 추가)

```python
class HardVQBottleneck(nn.Module):
    """EMA codebook update + code restart 지원"""
    def __init__(self, z_dim: int, num_codes: int = 64, ema_decay: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.z_dim = z_dim
        self.ema_decay = ema_decay

        self.register_buffer('codebook', torch.randn(num_codes, z_dim))
        self.register_buffer('ema_cluster_size', torch.ones(num_codes))
        self.register_buffer('ema_embed_sum', torch.randn(num_codes, z_dim))

    def forward(self, z: torch.Tensor):
        # nearest code
        dists = torch.cdist(z, self.codebook)
        indices = dists.argmin(dim=-1)              # (B,)
        z_q = self.codebook[indices]                # (B, z_dim)

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(z, indices)

        # commitment loss (encoder → codebook, codebook frozen)
        loss_commit = F.mse_loss(z, z_q.detach())

        # straight-through
        z_q_st = z + (z_q - z).detach()

        return z_q_st, loss_commit, indices

    def _ema_update(self, z, indices):
        one_hot = F.one_hot(indices, self.num_codes).float()   # (B, K)
        self.ema_cluster_size = (
            self.ema_decay * self.ema_cluster_size
            + (1 - self.ema_decay) * one_hot.sum(0)
        )
        self.ema_embed_sum = (
            self.ema_decay * self.ema_embed_sum
            + (1 - self.ema_decay) * one_hot.T @ z
        )
        # normalize
        n = self.ema_cluster_size.clamp(min=1e-5)
        self.codebook = self.ema_embed_sum / n.unsqueeze(1)

        # code restart: 사용 안 되는 code는 랜덤 z로 재초기화
        dead = self.ema_cluster_size < 1.0
        if dead.any():
            self.codebook[dead] = z[torch.randperm(z.shape[0])[:dead.sum()]]
```

#### `predict` 변경 (stoch_latent_flow_prior.py)

```python
# Hard VQ prior inference
if hasattr(self, 'vq_prior'):
    one_hot = self.vq_prior(planner_context)     # (B, K) one-hot
    z = one_hot @ self.vq.codebook               # (B, z_dim)
else:
    z = euler_integrate(self.prior_flow, ...)    # 기존 flow
```

---

## 6. 전체 실험 로드맵

```
M7 (InfoNCE Balanced)        ← 현재 학습 중
    ↓ 결과 확인
M8 (FiLM soft binding)       ← z binding 효과 검증
    ↓ z_shuffle_gap + intervention test 통과 시
M9-1 (Soft VQ)               ← binding 강화, prior 구조 유지
    ↓ codebook 활성화 확인 시
M9-2 (Hard VQ)               ← 완전한 discrete binding
```

> M8에서 z binding 효과가 없으면 M9로 넘어가도 같은 문제 반복 가능.
> 반드시 M8 검증 먼저.

---

## 6. 공통 측정 지표 (M9-1/M9-2 모두)

| 측정 | 방법 | 목적 |
|------|------|------|
| codebook usage | 전체 K개 중 활성 code 수 | dead code 감지 |
| task-code alignment | task별 dominant code 분포 | "plan selection" 실제로 됐는지 |
| z intervention | z를 task A→B code로 교체 | hard binding 효과 확인 |
| action_mse_prior | 기존과 동일 | quality 하락 여부 |
| z_shuffle_gap | 기존과 동일 | z 의존성 정량화 |

---

## 7. 핵심 한 줄 요약

> **VQ-VAE의 교훈은 "이산이 좋다"가 아니라 "decoder가 latent를 우회하지 못하게 interface를 설계하라"다.**
> M8이 soft binding, M9-1이 soft-hard binding, M9-2가 완전한 hard binding.
> binding 강도와 action quality 사이 trade-off를 단계적으로 탐색한다.
