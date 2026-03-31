"""
LatentVLA 오프라인 평가 스크립트

Usage:
    # 기본 평가
    python scripts/evaluate_offline_vlm.py --checkpoint outputs/runs/vlm_plan/ckpt_final.pt

    # best-of-K sweep
    python scripts/evaluate_offline_vlm.py --checkpoint ... --sweep_k

    # z_form override
    python scripts/evaluate_offline_vlm.py --checkpoint ... --z_form last

출력 지표:
    action_mse_prior      : prior z로 예측한 행동 MSE
    action_mse_posterior  : posterior z로 예측한 행동 MSE (상한)
    prior_posterior_gap   : 두 MSE 차이 (gap이 클수록 z가 유용)
    z_shuffle_gap         : z shuffle 후 MSE 증가폭 (클수록 z가 의미 있음)
    best_of_{k}           : K샘플 중 최선 MSE
    sampling_diversity    : 샘플 간 표준편차
    future_cosine_sim     : semantic head의 미래 예측 cosine similarity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import yaml

from training.builder import build_vlm_datasets, build_dataloaders_vlm, build_vlm_model
from training.trainer_vlm import VLMOfflineEvaluator


def _deep_merge(base: dict, override: dict) -> dict:
    """override 키로 base를 덮어씀 (중첩 dict는 재귀 병합)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_model(checkpoint_path: str, device: torch.device,
               z_form_override: str = None, data_path_override: str = None,
               base_config: str = None):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]

    # yaml config를 base로 사용하고, 체크포인트 cfg로 덮어씀
    if base_config:
        with open(base_config) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg = _deep_merge(yaml_cfg, cfg)

    # z_form override (ablation 비교용)
    if z_form_override:
        cfg["system2"]["z_form"] = z_form_override

    # data_path override
    if data_path_override:
        if "data" not in cfg:
            cfg["data"] = {
                "dataset_type": "libero",
                "action_horizon": 8,
                "image_size": 224,
                "num_workers": 4,
            }
        cfg["data"]["dataset_path"] = data_path_override

    if "data" not in cfg:
        raise ValueError(
            "체크포인트 cfg에 'data' 키가 없습니다.\n"
            "  --config configs/vlm_paligemma.yaml  또는\n"
            "  --data_path /path/to/libero_object/  를 지정하세요."
        )

    train_ds, val_ds = build_vlm_datasets(cfg)
    _, val_loader = build_dataloaders_vlm(train_ds, val_ds, cfg)

    model = build_vlm_model(cfg, action_dim=train_ds.action_dim,
                            proprio_dim=train_ds.proprio_dim)

    # LoRA 활성화 후 가중치 로드 (Stage 2 체크포인트)
    saved = ckpt["model"]
    if saved.get("system2_lora"):
        model.system2.enable_lora()

    model.load_state_dict_from_save(saved)
    model = model.to(device)
    model.eval()

    return model, val_loader, cfg


def run_eval(model, val_loader, device, best_of_ks, max_batches=None):
    evaluator = VLMOfflineEvaluator(model=model, device=device, best_of_ks=best_of_ks)
    return evaluator.evaluate(val_loader, max_batches=max_batches)


def print_metrics(metrics: dict, z_form: str):
    print(f"\n{'='*55}")
    print(f"  LatentVLA Offline Evaluation  (z_form={z_form})")
    print(f"{'='*55}")
    order = [
        "action_mse_prior", "action_mse_posterior", "prior_posterior_gap",
        "z_shuffle_gap", "best_of_1", "best_of_3", "best_of_5", "best_of_10",
        "sampling_diversity", "future_cosine_sim",
    ]
    for k in order:
        if k in metrics:
            print(f"  {k:<28s}: {metrics[k]:.4f}")
    # 혹시 추가 지표 있으면
    for k, v in metrics.items():
        if k not in order and isinstance(v, float):
            print(f"  {k:<28s}: {v:.4f}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="LatentVLA 체크포인트 경로")
    parser.add_argument("--sweep_k", action="store_true",
                        help="K=1,3,5,10 best-of-K sweep")
    parser.add_argument("--z_form", default=None,
                        choices=["last", "pool", "plan"],
                        help="z_form override (기본: 체크포인트에 저장된 값)")
    parser.add_argument("--output", default=None,
                        help="결과 저장 JSON 경로 (기본: 체크포인트 디렉토리)")
    parser.add_argument("--config", default=None,
                        help="yaml config 경로 (체크포인트 cfg에 누락된 키를 보충)")
    parser.add_argument("--data_path", default=None,
                        help="데이터셋 경로 override")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="평가할 최대 배치 수 (빠른 smoke 검증용, 기본: 전체)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (기본: 자동)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[evaluate_offline_vlm] device={device}")

    best_of_ks = [1, 3, 5, 10] if args.sweep_k else [1, 5]

    model, val_loader, cfg = load_model(
        args.checkpoint, device, args.z_form, args.data_path, args.config
    )
    z_form = cfg["system2"]["z_form"]

    metrics = run_eval(model, val_loader, device, best_of_ks, args.max_batches)
    metrics["z_form"] = z_form
    metrics["checkpoint"] = args.checkpoint

    print_metrics(metrics, z_form)

    # JSON 저장
    out_path = args.output
    if out_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(ckpt_dir, f"eval_{ckpt_stem}_{z_form}.json")

    with open(out_path, "w") as f:
        json.dump({k: round(v, 6) if isinstance(v, float) else v
                   for k, v in metrics.items()}, f, indent=2)
    print(f"[evaluate_offline_vlm] 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
