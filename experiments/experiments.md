# LatentVLA 실험 인덱스

| ID | 실험명 | 상태 | 핵심 변경 | 문서 |
|----|--------|:----:|----------|------|
| M1 | FlatFlow | ✓ 완료 | z 없음, 베이스라인 | `outputs/runs/flat_flow_100ep_20260404/` |
| M2 | DetLatent | ✓ 완료 | Oracle z (미래 이미지 MLP) | `outputs/runs/det_latent_100ep_20260405/` |
| M3 | StochVAE | ✓ 완료 | VAE posterior z | `outputs/runs/stoch_vae_100ep_20260405/` |
| M4 | StochFlowPrior | ✓ 완료 | Flow-based prior z | `outputs/runs/sfp_100ep_20260401/` |
| M5 | VLM SFP Plan | ✓ 완료 | VLM plan token z | `outputs/runs/vlm_sfp_plan_100ep_20260405/` |
| M6 | VLM SFP + InfoNCE | ✓ 완료 | M5 + z-InfoNCE loss (λ=0.1) | [M6_infonce.md](M6_infonce.md) |
| M7 | VLM SFP + InfoNCE Balanced | 🟡 학습 중 | M6 + λ=0.01 + Task-Balanced Sampler | [M7_infonce_balanced.md](M7_infonce_balanced.md) |
| M8 | VLM SFP + FiLM z-Modulation | 🔴 설계 완료 | z concat → FiLM per-block modulation | [M8_film.md](M8_film.md) |

## 핵심 지표 요약

| 모델 | MSE prior ↓ | MSE post ↓ | z_shuffle_gap ↑ | prior_post_gap ↑ |
|------|:-----------:|:----------:|:---------------:|:----------------:|
| M1 FlatFlow | 0.5530 | — | — | — |
| M2 DetLatent | **0.4776** | **0.0017** | **0.7837** | **0.4759** |
| M3 StochVAE | 0.6321 | 0.6205 | -0.0039 | 0.0116 |
| M4 StochFlowPrior | 0.6540 | 0.5395 | 0.0428 | 0.1144 |
| M5 VLM SFP Plan | 0.6084 | 0.5318 | 0.0163 | 0.0766 |
| M6 VLM + InfoNCE (ckpt_10) | 1.2946 | 1.1883 | 0.0537 | 0.1063 |
| M6 VLM + InfoNCE (ckpt_20) | 0.8588 | 0.8168 | 0.0268 | 0.0420 |
| M7 VLM + InfoNCE Balanced | — | — | — | — |

M6: ckpt_10과 ckpt_20이 z 구조 vs prior 품질 트레이드오프. M7에서 동시 달성 목표.
