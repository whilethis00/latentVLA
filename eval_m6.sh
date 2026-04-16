#!/bin/bash
cd "$(dirname "$0")"
conda run -n vla python scripts/evaluate_offline_vlm.py --checkpoint outputs/runs/vlm_sfp_infonce_20260414/ckpt_10.pt --config configs/vlm_paligemma_infonce.yaml --sweep_k
conda run -n vla python scripts/evaluate_offline_vlm.py --checkpoint outputs/runs/vlm_sfp_infonce_20260414/ckpt_20.pt --config configs/vlm_paligemma_infonce.yaml --sweep_k
