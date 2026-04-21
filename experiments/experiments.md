# LatentVLA 실험 인덱스

| ID | 실험명 | 상태 | 핵심 변경 | 문서 |
|----|--------|:----:|----------|------|
| M1 | FlatFlow | ✓ 완료 | z 없음, 베이스라인 | `outputs/runs/flat_flow_100ep_20260404/` |
| M2 | DetLatent | ✓ 완료 | Oracle z (미래 이미지 MLP) | `outputs/runs/det_latent_100ep_20260405/` |
| M3 | StochVAE | ✓ 완료 | VAE posterior z | `outputs/runs/stoch_vae_100ep_20260405/` |
| M4 | StochFlowPrior | ✓ 완료 | Flow-based prior z | `outputs/runs/sfp_100ep_20260401/` |
| M5 | VLM SFP Plan | ✓ 완료 | VLM plan token z | `outputs/runs/vlm_sfp_plan_100ep_20260405/` |
| M6 | VLM SFP + InfoNCE | ✓ 완료 | M5 + z-InfoNCE loss (λ=0.1) | [M6_infonce.md](M6_infonce.md) |
| M7 | VLM SFP + InfoNCE Balanced | ✓ 완료 | M6 + λ=0.01 + Task-Balanced Sampler | [M7_infonce_balanced.md](M7_infonce_balanced.md) |
| M8 | VLM SFP + InfoNCE S1 Only | ✓ 완료 (ep92, walltime 초과) | S1 data only + infonce_stage1_only=True | `outputs/runs/vlm_sfp_infonce_s1only_20260418/` |
| M9 | VLM SFP + InfoNCE S1+S2 | 🔴 설계 완료 | M8 + infonce_stage1_only=False | [M9_infonce_s2.md](M9_infonce_s2.md) |
| (보류) M8-FiLM | VLM SFP + FiLM z-Modulation | ⏸ 보류 | z-space 수축 해결 후 검토 | [M8_film.md](M8_film.md) |
| (보류) M9-VQ | Soft/Hard VQ Binding | ⏸ 보류 | representation 먼저 살린 뒤 검토 | [M9_vq.md](M9_vq.md) |

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
| M7 VLM + InfoNCE Balanced (ep100) | 0.5694 | 0.5252 | 0.0047 | 0.0442 |
| M8 VLM + InfoNCE S1 Only (ep90) | 0.5120 | 0.4606 | 0.0086 | 0.0514 |
| M9 VLM + InfoNCE S1+S2 | — | — | — | — |

**M8 판결 실험 결과 (2026-04-21)**: `z_mu_var_mean=0.0345`, `delta_null=+0.225`, `probe_ratio=0.919`
→ posterior→decoder 경로는 살아 있음. z-space 수축 + prior-posterior mismatch가 주원인.
→ M9 방향: representation 먼저, binding(FiLM/VQ)은 그 다음.
