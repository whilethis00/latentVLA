# LatentVLA 코드 구현 계획 (4월)

> 목표: ICRA Aha 1~3을 실험할 수 있는 코드를 4월 안에 완성한다
> 기간: 2026-04-01 ~ 2026-04-30 (4주)
> 전제: 기존 코드(M1~M4) 건드리지 않고, 위에 VLM 레이어를 쌓는다

---

## 전체 구조 변화 한눈에 보기

```
[기존]
  ContextEncoder (SigLIP + proprio) → C_t
  + StochFlowPrior (C_t → z → actions)

[새로 추가]
  System2VLM (PaliGemma-3B + LoRA) → f̃
  + LatentVLA (f̃ → StochFlowPrior → z → actions)
```

**원칙:**
- 기존 M1~M4 코드는 손대지 않는다 (Exp 1 베이스라인으로 그대로 사용)
- 새 파일만 추가한다
- 인터페이스는 기존 Trainer가 그대로 쓸 수 있게 맞춘다

---

## 추가/수정 파일 목록

```
추가:
  models/system2_vlm.py          ← PaliGemma wrapper (핵심)
  models/latent_vla.py           ← System2 + StochFlowPrior 통합
  training/trainer_vlm.py        ← 2-stage 학습 루프
  configs/vlm_paligemma.yaml     ← VLM 전용 설정
  scripts/smoke_test_vlm.py      ← VLM 동작 확인
  scripts/train_vlm.py           ← VLM 학습 진입점
  scripts/run_exp1.sh            ← Exp 1 자동화
  scripts/run_exp2.sh            ← Exp 2 자동화

수정:
  training/builder.py            ← VLM 모델 빌드 추가
  requirements.txt               ← peft, bitsandbytes 추가
```

---

## Week 1 (4/1 ~ 4/7): PaliGemma Wrapper 구현

**목표:** PaliGemma를 로드하고 이미지+언어 → f̃ (256-dim) 까지 동작 확인

### 구현할 것: `models/system2_vlm.py`

```python
class System2VLM(nn.Module):
    """
    PaliGemma-3B를 System 2로 사용하는 VLM wrapper.
    이미지 + 언어 → semantic feature f̃ (context_dim)

    z_form 옵션:
      'last'  : 마지막 토큰 hidden state → Linear → f̃
      'pool'  : 마지막 K 토큰 평균 → Linear → f̃
      'plan'  : [PLAN] 특수 토큰 위치 hidden state → Linear → f̃
    """

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        context_dim: int = 256,
        z_form: str = "plan",       # 'last' | 'pool' | 'plan'
        pool_k: int = 8,            # z_form='pool'일 때 평균 낼 토큰 수
        freeze_backbone: bool = True,  # Stage 1은 True, Stage 2는 False (LoRA)
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list = None,  # ['q_proj', 'v_proj']
    ):

    def forward(
        self,
        pixel_values: torch.Tensor,    # (B, 3, 224, 224) - PaliGemma processor 전처리 필요
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        """
        Returns: f̃ (B, context_dim)
        """

    def tokenize(
        self,
        texts: list[str],
        images: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PaliGemmaProcessor로 이미지+텍스트 토크나이즈"""

    def enable_lora(self):
        """Stage 2 진입 시 호출: backbone 언프리즈 + LoRA 활성화"""
```

**구현 포인트:**
- `z_form='plan'` : 입력 시퀀스 끝에 `"\nPLAN:"` 프롬프트 추가, 그 위치 hidden state 사용
- `z_form='pool'` : `last_hidden_state[:, -pool_k:, :].mean(dim=1)`
- `z_form='last'` : `last_hidden_state[:, -1, :]`
- LoRA는 `peft.get_peft_model()` 사용, target: `q_proj`, `v_proj`
- `output_hidden_states=True` 로 forward 해서 last layer hidden state 추출
- Projection: `nn.Sequential(nn.Linear(2048, 512), nn.SiLU(), nn.Linear(512, context_dim))`

**주의사항:**
- PaliGemma processor가 SigLIP과 이미지 전처리 방식이 다름 → 데이터셋에서 raw image 따로 저장 필요
- `google/paligemma-3b-pt-224` : pretrained (사용할 것)
- `google/paligemma-3b-mix-224` : instruction-tuned (비교 실험용)

### Week 1 완료 기준

```bash
# 이 명령어가 에러 없이 돌아야 함
python scripts/smoke_test_vlm.py
# 출력:
# [System2VLM] PaliGemma-3B loaded (3.0B params, 1.2M LoRA params)
# [System2VLM] z_form=plan, f̃ shape: (2, 256) ✓
# [System2VLM] z_form=pool, f̃ shape: (2, 256) ✓
# [System2VLM] z_form=last, f̃ shape: (2, 256) ✓
```

---

## Week 2 (4/8 ~ 4/14): LatentVLA 통합 + 데이터로더

**목표:** 데이터 → System2VLM → StochFlowPrior 전체 forward pass 성공

### 구현할 것 1: `models/latent_vla.py`

```python
class LatentVLA(nn.Module):
    """
    System 2 (VLM) + System 1 (StochFlowPrior) 통합 모델.

    기존 StochFlowPrior와 인터페이스 동일하게 맞춤
    → 기존 Trainer 코드 최소 수정으로 사용 가능

    학습:
      1. System2VLM(image, lang) → f̃
      2. StochFlowPrior.compute_loss(context=f̃, actions, future_feat, planner_context=f̃)

    추론:
      1. System2VLM(image, lang) → f̃
      2. StochFlowPrior.predict(context=f̃, planner_context=f̃)
    """

    def __init__(
        self,
        system2: System2VLM,
        action_dim: int,
        action_horizon: int = 8,
        z_dim: int = 128,
        context_dim: int = 256,      # System2 출력 차원
        future_feat_dim: int = 768,  # SigLIP embed dim (semantic loss용)
        prior_weight: float = 1.0,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
    ):
        # System 1: 기존 StochFlowPrior 그대로 재사용
        self.system2 = system2
        self.system1 = StochLatentFlowPrior(
            context_dim=context_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            z_dim=z_dim,
            future_feat_dim=future_feat_dim,
            prior_weight=prior_weight,
            flow_hidden=flow_hidden,
            flow_depth=flow_depth,
            flow_steps=flow_steps,
        )
        # Semantic loss용 SigLIP (frozen, 기존과 동일하게 유지)
        self.siglip_encoder = ...

    def compute_loss(self, batch: dict, device, semantic_weight=0.1) -> dict:
        """
        batch에서 직접 받아서 처리 (Trainer 수정 최소화)
        """
        # 1. System 2: VLM → f̃
        pixel_values, input_ids, attn_mask = self.system2.tokenize(
            batch['language'], batch['image'], device
        )
        f_tilde = self.system2(pixel_values, input_ids, attn_mask)  # (B, 256)

        # 2. Future semantic feat (frozen SigLIP, 기존 방식 그대로)
        future_feat = self.siglip_encoder(batch['future_image'])

        # 3. System 1 loss
        return self.system1.compute_loss(
            context=f_tilde,
            actions=batch['actions'],
            future_feat=future_feat,
            planner_context=f_tilde,
            semantic_weight=semantic_weight,
        )

    def predict(self, batch: dict, device, **kwargs) -> torch.Tensor:
        pixel_values, input_ids, attn_mask = self.system2.tokenize(
            batch['language'], batch['image'], device
        )
        f_tilde = self.system2(pixel_values, input_ids, attn_mask)
        return self.system1.predict(context=f_tilde, planner_context=f_tilde, **kwargs)
```

### 구현할 것 2: `data/libero_dataset.py` 수정

- 기존 `__getitem__`이 반환하는 dict에 `raw_image` 추가
  - SigLIP용: 기존 `image` (224×224, normalized)
  - PaliGemma용: `raw_image` (PIL Image or uint8 tensor) → System2VLM.tokenize에서 처리
- 이유: PaliGemma processor와 SigLIP processor의 정규화 파라미터가 다름

```python
# 기존 반환 dict에 추가:
{
    "image": ...,           # 기존 SigLIP 전처리
    "raw_image": img_uint8, # (H, W, 3) uint8 ← 추가
    ...
}
```

### 구현할 것 3: `training/builder.py` 수정

```python
def build_vlm_model(cfg: dict, action_dim: int) -> LatentVLA:
    """VLM 모델 빌드"""
    system2 = System2VLM(
        model_name=cfg['system2']['model_name'],
        context_dim=cfg['encoder']['context_dim'],
        z_form=cfg['system2']['z_form'],
        freeze_backbone=True,  # 항상 Stage 1으로 시작
        lora_rank=cfg['system2']['lora_rank'],
    )
    return LatentVLA(system2=system2, action_dim=action_dim, ...)

def build_vlm_optimizer(model: LatentVLA, cfg: dict):
    """
    2그룹 optimizer:
    - VLM LoRA params: lr=3e-5 (작게)
    - Flow policy params: lr=3e-4 (기존과 동일)
    """
    lora_params = [p for n, p in model.system2.named_parameters() if 'lora' in n]
    flow_params = list(model.system1.parameters())
    return AdamW([
        {'params': lora_params, 'lr': cfg['training']['lora_lr']},
        {'params': flow_params, 'lr': cfg['training']['learning_rate']},
    ], weight_decay=cfg['training']['weight_decay'])
```

### Week 2 완료 기준

```bash
# LIBERO 데이터 배치 → LatentVLA forward → loss 계산까지 동작
python -c "
from training.builder import build_vlm_model
# ... 배치 만들고 compute_loss 호출
print('Loss:', loss_dict['total_loss'].item())  # 숫자 나와야 함
"
```

---

## Week 3 (4/15 ~ 4/21): 2-stage Trainer + 첫 학습

**목표:** LIBERO에서 Stage 1 학습 → loss 하강 확인

### 구현할 것: `training/trainer_vlm.py`

기존 `Trainer`를 상속해서 VLM 특화 기능만 추가

```python
class VLMTrainer(Trainer):
    """
    기존 Trainer 상속. 추가 기능:
      - 2-stage 학습 (Stage1: VLM frozen, Stage2: LoRA)
      - gradient checkpointing
      - bf16 mixed precision
      - 2그룹 optimizer (VLM LR != flow LR)
    """

    def __init__(self, model: LatentVLA, ...):
        # stage 경계
        self.stage2_epoch = cfg['training']['stage2_epoch']  # default=10

    def train(self):
        for epoch in range(1, num_epochs + 1):
            # Stage 2 진입
            if epoch == self.stage2_epoch:
                print("[VLMTrainer] Stage 2 시작: VLM LoRA 활성화")
                self.model.system2.enable_lora()
                self._rebuild_optimizer()  # LoRA params 추가

            self._train_epoch(epoch)
            ...

    def _train_step(self, batch):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = self.model.compute_loss(batch, self.device, ...)
        ...

    def _save_checkpoint(self, tag):
        # LoRA 가중치만 저장 (전체 PaliGemma 저장 안 함 → 용량 절약)
        torch.save({
            'system2_lora': get_peft_model_state_dict(self.model.system2.vlm),
            'system1': self.model.system1.state_dict(),
            'optimizer': ...,
            'cfg': ...,
        }, path)
```

### `configs/vlm_paligemma.yaml` 추가

```yaml
system2:
  model_name: "google/paligemma-3b-pt-224"
  z_form: "plan"          # 'last' | 'pool' | 'plan'
  pool_k: 8
  lora_rank: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj"]

encoder:
  context_dim: 256

training:
  stage2_epoch: 10        # 이 epoch부터 LoRA 활성화
  lora_lr: 3.0e-5         # VLM LoRA LR (flow LR보다 10배 작게)
  learning_rate: 3.0e-4   # flow policy LR
  num_epochs: 100
  batch_size: 16          # PaliGemma 때문에 기존(64)보다 줄임
  grad_clip: 1.0
  grad_accum_steps: 4     # effective batch = 16 * 4 = 64

data:
  dataset_type: libero
  dataset_path: "/home/introai4/home_lustre/introai4/libero/libero_object_openvla_processed/"
  action_horizon: 8
  action_dim: 7
```

### Week 3 완료 기준

```bash
# Stage 1 학습 10 epoch 돌고, loss 하강 로그 나와야 함
python scripts/train_vlm.py --config configs/vlm_paligemma.yaml \
    --override training.num_epochs=10

# 출력 예시:
# [VLMTrainer] Stage 1 (VLM frozen)
# [Epoch  1] total_loss=2.8341  action_flow_loss=1.3211  prior_flow_loss=1.4821
# [Epoch  5] total_loss=1.2411  action_flow_loss=0.7123  prior_flow_loss=0.5011
# [Epoch 10] total_loss=0.8932  ...
# [VLMTrainer] Stage 2 시작: VLM LoRA 활성화 ← 11epoch부터
```

---

## Week 4 (4/22 ~ 4/30): 전체 학습 + Exp 1 준비

**목표:** 3가지 z_form 모두 100 epoch 학습 + Exp 1 오프라인 데이터 준비

### 학습 실행: `scripts/run_exp2.sh`

```bash
#!/bin/bash
# Exp 2: VLM-z 3가지 form 학습
for Z_FORM in last pool plan; do
  python scripts/train_vlm.py \
    --config configs/vlm_paligemma.yaml \
    --override system2.z_form=$Z_FORM \
               training.output_dir=outputs/runs/vlm_sfp_$Z_FORM
  echo "[$Z_FORM] 학습 완료"
done
```

### Exp 1 준비: `scripts/run_exp1.sh`

```bash
#!/bin/bash
# Exp 1: M1~M4 + VLM-z 모델들의 z_shuffle_gap 측정
# (LIBERO 온라인 평가는 5월에, 오프라인은 지금)
for MODEL in flat_flow det_latent stoch_vae stoch_flow_prior; do
  python scripts/evaluate_offline.py \
    --run_dir outputs/runs/$MODEL \
    --best_of_k 1 3 5 10 \
    --save_path outputs/exp1/${MODEL}_metrics.json
done

# VLM 모델 평가
for Z_FORM in last pool plan; do
  python scripts/evaluate_offline.py \
    --run_dir outputs/runs/vlm_sfp_$Z_FORM \
    --model_type vlm \
    --save_path outputs/exp1/vlm_sfp_${Z_FORM}_metrics.json
done

# scatter plot 생성
python scripts/plot_exp1.py --results_dir outputs/exp1/
```

### 추가 구현: `scripts/plot_exp1.py`

```python
"""
Exp 1 scatter plot:
  x축: z_shuffle_gap
  y축: action_mse_prior (online success rate는 5월에 추가)
  점: M1(FlatFlow), M2(DetLatent), M3(StochVAE), M4(StochFlowPrior),
      VLM-last, VLM-pool, VLM-plan
"""
```

### Week 4 완료 기준

```bash
# 이 3개 파일이 생성되어야 함
ls outputs/runs/vlm_sfp_last/ckpt_final.pt    ✓
ls outputs/runs/vlm_sfp_pool/ckpt_final.pt    ✓
ls outputs/runs/vlm_sfp_plan/ckpt_final.pt    ✓

# Exp 1 scatter plot 생성
ls outputs/exp1/scatter_z_quality.png          ✓
```

---

## 의존성 추가 (`requirements.txt`)

```
# 기존 유지
torch>=2.1.0
transformers>=4.40.0   # PaliGemma 지원 버전

# 추가
peft>=0.10.0           # LoRA
bitsandbytes>=0.43.0   # 8-bit optimizer (메모리 절약)
accelerate>=0.28.0     # gradient checkpointing
```

---

## 주요 리스크 & 대응

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| PaliGemma OOM (A100 80GB) | 낮음 | batch_size=16 + grad_accum=4 + bf16 |
| PaliGemma processor 이미지 형식 충돌 | 중간 | `raw_image` 따로 저장으로 해결 |
| Stage 1→2 전환 시 loss 폭발 | 중간 | stage2 LR warmup 100 steps 추가 |
| LIBERO HDF5 구조가 기존 loader와 안 맞음 | 낮음 | Week 2에서 확인, 필요 시 수정 |
| 3가지 z_form 학습 시간 부족 | 낮음 | A100에서 LIBERO 100epoch ≈ 8시간/run |

---

## 4월 말 체크리스트

```
Week 1:
  □ models/system2_vlm.py 구현
  □ smoke_test_vlm.py 통과 (3가지 z_form)

Week 2:
  □ models/latent_vla.py 구현
  □ data/libero_dataset.py raw_image 추가
  □ training/builder.py build_vlm_model() 추가
  □ forward pass 전체 동작 확인

Week 3:
  □ training/trainer_vlm.py 구현 (2-stage)
  □ configs/vlm_paligemma.yaml 작성
  □ scripts/train_vlm.py 작성
  □ Stage 1 10 epoch loss 하강 확인

Week 4:
  □ vlm_sfp_last / pool / plan 전체 100 epoch 학습
  □ M1~M4 Exp 1 오프라인 평가
  □ Exp 1 scatter plot 생성
  □ 코드 + 결과 GitHub push
```

---

## 5월 예고

4월 코드가 완성되면 5월은:
- LIBERO 시뮬레이터 설치 → 온라인 성공률 측정 (Exp 1 완성)
- VLM-z vs MLP-z 정량 비교표 완성 (Exp 2 완성)
- t-SNE 시각화 (Exp 3)
- 논문 figure 초안 작성 시작

---

*코드 작업 중 설계가 바뀌면 이 문서를 업데이트한다.*
