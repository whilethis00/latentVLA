"""
LatentVLA (VLM + StochFlowPrior) 학습 진입점

사용법:
  python scripts/train_vlm.py --config configs/vlm_paligemma.yaml
  python scripts/train_vlm.py --config configs/vlm_paligemma.yaml \
      --override system2.z_form=last training.output_dir=outputs/runs/vlm_sfp_last
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import random
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[],
                   help="key=value 형태로 config 덮어쓰기. 예: system2.z_form=last")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list):
    for ov in overrides:
        key, val = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # 타입 자동 추론
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        d[keys[-1]] = val
    return cfg


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, args.override)

    # output_dir을 z_form에 맞게 자동 설정 (override 없으면)
    if not any("output_dir" in ov for ov in args.override):
        z_form = cfg["system2"]["z_form"]
        cfg["training"]["output_dir"] = f"outputs/runs/vlm_sfp_{z_form}"

    print("=" * 60)
    print(f"[train_vlm] z_form     : {cfg['system2']['z_form']}")
    print(f"[train_vlm] output_dir : {cfg['training']['output_dir']}")
    print(f"[train_vlm] epochs     : {cfg['training']['num_epochs']}")
    print(f"[train_vlm] batch_size : {cfg['training']['batch_size']} "
          f"(x{cfg['training']['grad_accum_steps']} accum = "
          f"{cfg['training']['batch_size'] * cfg['training']['grad_accum_steps']} eff)")
    print("=" * 60)

    # 시드
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_vlm] device: {device}")

    # 데이터셋
    from training.builder import build_vlm_datasets, build_dataloaders_vlm
    train_ds, val_ds = build_vlm_datasets(cfg)
    train_loader, val_loader = build_dataloaders_vlm(train_ds, val_ds, cfg)
    print(f"[train_vlm] train: {len(train_ds)}  val: {len(val_ds)}")

    # 모델
    from training.builder import build_vlm_model
    model = build_vlm_model(cfg, action_dim=train_ds.action_dim,
                            proprio_dim=train_ds.proprio_dim)
    print(f"[train_vlm] 모델 생성 완료")

    # 학습
    from training.trainer_vlm import VLMTrainer
    trainer = VLMTrainer(model, train_loader, val_loader, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
