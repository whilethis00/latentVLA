# M8: VLM SFP Plan + FiLM z-Modulation

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-16 |
| **베이스** | M7 (VLM SFP + InfoNCE Balanced) |
| **목적** | z를 side information(concat)에서 computation modulator(FiLM)로 전환해 decoder의 z 의존성을 구조적으로 보장 |
| **상태** | 🔴 설계 완료, 학습 미실행 |
| **출력 경로** | `outputs/runs/vlm_sfp_film_20260416/` |

---

## 2. 왜 M7만으로 부족한가

### 현재 문제 진단

M6/M7의 공통 실패 원인:

> **z를 concat으로 넣으면 decoder가 z를 무시해도 학습이 된다.**

현재 action_flow 구조:

```
action_cond = cat([context, z])          # (B, context_dim + z_dim)
h = input_proj(cat([x_t, t_emb, cond]))  # z가 입력단에서 한 번만 들어감
for block in residual_blocks:
    h = h + block(h)                     # z와 무관하게 진행
output = output_proj(h)
```

z가 `input_proj`에서 한 번 섞이고 나면 이후 레이어에서 z 정보가 희석됨.
모델이 `input_proj` weight를 작게 만들면 z를 사실상 무시 가능.

### 목표: z as computation modulator

```
z가 각 residual block의 스케일·오프셋을 직접 결정
→ z 없이는 블록이 다른 방식으로 작동
→ z를 무시하는 것이 구조적으로 불가능
```

---

## 3. 핵심 아이디어: FiLM-style z modulation

### 기본 FiLM 수식

각 residual block l에 대해:

```
h_l  = ResidualBlock_l(h_{l-1})          # 기존 연산
γ_l, β_l = MLP_l(z_feat)                # z → scale, shift
h_l' = γ_l ⊙ h_l + β_l                 # FiLM 적용
h_l' = h_{l-1} + α_l * (h_l' - h_{l-1}) # residual 보호 (identity 초기화)
```

`α_l`: learnable scalar, 초기값 0.1 (처음엔 거의 원래 모델처럼 작동, 학습되며 커짐)

### Identity 초기화 전략

```python
# MLP_l의 마지막 레이어를 zero-init
nn.init.zeros_(film_mlp[-1].weight)
nn.init.ones_(film_mlp[-1].bias[:hidden])   # gamma 부분 → 1
nn.init.zeros_(film_mlp[-1].bias[hidden:])  # beta 부분  → 0
```

→ 학습 초기에 FiLM이 항등 변환 → 기존 모델 성능 유지하며 시작

---

## 4. 구조 변경 상세

### 현재 vs M8 아키텍처

```
[현재 M5/M6/M7]
z ──┐
    cat → input_proj → Block_0 → Block_1 → Block_2 → Block_3 → output
ctx─┘

[M8]
z ──→ ZEncoder → z_feat
                    │
                  FiLM_1  FiLM_2  FiLM_3      (middle 3 blocks)
                    │       │       │
ctx → input_proj → Block_0 → Block_1 → Block_2 → Block_3 → output
```

**변경 포인트:**
- z가 `action_flow`의 cond에서 **제거됨** (`cond_dim = context_dim`)
- z는 별도 `ZEncoder`를 통해 `z_feat`으로 변환
- depth=4 기준: block 1, 2, 3에 FiLM 적용 (block 0은 low-level, 제외)

---

## 5. 구현 범위

| 파일 | 변경 내용 |
|------|----------|
| `models/flow_utils.py` | `FiLMResidualBlock`, `FiLMVelocityMLP` 추가 |
| `models/stoch_latent_flow_prior.py` | `action_flow`를 `FiLMVelocityMLP`로 교체, z 전달 방식 변경 |
| `configs/vlm_paligemma_film.yaml` | M7 기반, `use_film_z: true` 추가 |

`latent_vla.py`, `trainer_vlm.py` — **변경 없음**

---

## 6. Pseudo-code

### `FiLMVelocityMLP` (flow_utils.py에 추가)

```python
class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden: int, z_feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(z_feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 2),  # gamma, beta concat
        )
        # Identity init: gamma→1, beta→0
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.ones_(self.film_mlp[-1].bias[:hidden])
        nn.init.zeros_(self.film_mlp[-1].bias[hidden:])

        self.alpha = nn.Parameter(torch.tensor(0.1))  # modulation 강도

    def forward(self, x: torch.Tensor, z_feat: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        film_out = self.film_mlp(z_feat)
        gamma, beta = film_out.chunk(2, dim=-1)
        h_film = gamma * h + beta
        return x + self.alpha * h_film


class FiLMVelocityMLP(nn.Module):
    def __init__(self, x_dim, cond_dim, z_dim, hidden=512, depth=4,
                 time_dim=128, film_start_block=1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
        )
        z_feat_dim = hidden // 2

        in_dim = x_dim + time_dim + cond_dim  # z는 여기서 제외
        self.input_proj = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i >= film_start_block:
                self.blocks.append(FiLMResidualBlock(hidden, z_feat_dim))
            else:
                self.blocks.append(ResidualBlock(hidden))

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, x_dim),
        )
        self.film_start_block = film_start_block

    def forward(self, x, t, cond, z):
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([x, t_emb, cond], -1))
        z_feat = self.z_encoder(z)
        for i, blk in enumerate(self.blocks):
            if i >= self.film_start_block:
                h = blk(h, z_feat)
            else:
                h = blk(h)
        return self.output_proj(h)
```

### `stoch_latent_flow_prior.py` 변경

```python
# __init__ 변경
self.action_flow = FiLMVelocityMLP(
    x_dim=self.x_dim,
    cond_dim=context_dim,   # z 제거 (기존: context_dim + z_dim)
    z_dim=z_dim,
    hidden=flow_hidden,
    depth=flow_depth,
    film_start_block=1,     # block 0은 FiLM 제외
)

# compute_loss 변경
loss_action = film_flow_matching_loss(
    self.action_flow, actions.reshape(B, -1),
    cond=context,           # z 제거
    z=z_star.detach(),      # z는 별도로 전달
)

# predict 변경
x = euler_integrate_film(self.action_flow, context, z, self.x_dim, steps)
```

---

## 7. Hyperparameter

| 파라미터 | M7 | M8 | 비고 |
|---------|:--:|:--:|------|
| `use_film_z` | False | **True** | 핵심 변경 |
| `film_start_block` | — | **1** | block 0 제외 |
| `alpha` 초기값 | — | **0.1** | learnable, per-block |
| `infonce_weight` | 0.01 | **0.0** | FiLM이 z binding 담당, InfoNCE 제거 |
| 나머지 | M7과 동일 | — | — |

> InfoNCE 제거 이유: FiLM이 구조적으로 z를 강제하므로 loss-level 추가 불필요. 변수 격리.

---

## 8. 성공 기준

| 지표 | 목표 | 비고 |
|------|:----:|------|
| z_shuffle_gap ↑ | **> 0.043** | M4 수준 초과 |
| action_mse_prior ↓ | **≤ 0.608** | M5 수준 유지 |
| z-drop 민감도 | **MSE 증가** | z=0 대체 시 성능 하락 확인 |
| alpha 수렴값 | **> 0.1** | 학습 후 modulation이 실제로 커졌는지 |

---

## 9. 검증 지표 (기존 + 추가)

### 기존 (계속 측정)
- `action_mse_prior`, `action_mse_posterior`
- `z_shuffle_gap`, `prior_posterior_gap`
- `best_of_K`, `sampling_diversity`

### M8 신규 측정 (별도 분석 스크립트)

| 측정 | 방법 | 목적 |
|------|------|------|
| **z intervention** | z를 task A → task B로 교체 후 hidden 변화량 측정 | z가 실제로 computation을 바꾸는지 |
| **z-drop test** | z=0 대체 시 MSE 변화 | z 의존성 정량화 |
| **per-block alpha** | 학습 완료 후 α_l 값 로깅 | modulation이 어느 층에서 활성화됐는지 |
| **task-conditional modulation** | 동일 task 샘플들의 γ, β 분산 측정 | constant modulation collapse 감지 |

---

## 10. 리스크

| 리스크 | 대응 |
|--------|------|
| alpha가 0으로 수렴 (modulation 죽음) | alpha에 lower bound 0.01 또는 정규화 추가 |
| FiLM이 task-unrelated noise 학습 | z intervention test로 조기 감지 |
| prior flow 불안정 (prior는 FiLM 없음) | prior_flow는 기존 VelocityMLP 유지 |
| M7 대비 파라미터 증가 | ZEncoder + FiLM MLP → 약 +1M params, 허용 범위 |

---

## 11. 학습 커맨드

```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/train_vlm.py --config configs/vlm_paligemma_film.yaml
```

---

## 12. 결과

> 학습 미실행

---

## 13. 저장 파일 목록

```
outputs/runs/vlm_sfp_film_20260416/
├── ckpt_*.pt
├── train.log
├── train_log.jsonl
└── result.md
```
