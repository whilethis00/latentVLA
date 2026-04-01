"""
LatentVLA (VLM + StochFlowPrior) 학습 진입점

단일 GPU:
  python scripts/train_vlm.py --config configs/vlm_paligemma.yaml

DDP (멀티 GPU):
  torchrun --nproc_per_node=4 scripts/train_vlm.py --config configs/vlm_paligemma.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.distributed as dist
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

    # ── DDP 초기화 ────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1
    is_main    = rank == 0

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, args.override)

    if not any("output_dir" in ov for ov in args.override):
        z_form = cfg["system2"]["z_form"]
        cfg["training"]["output_dir"] = f"outputs/runs/vlm_sfp_{z_form}"

    if is_main:
        print("=" * 60)
        print(f"[train_vlm] z_form     : {cfg['system2']['z_form']}")
        print(f"[train_vlm] output_dir : {cfg['training']['output_dir']}")
        print(f"[train_vlm] epochs     : {cfg['training']['num_epochs']}")
        print(f"[train_vlm] batch_size : {cfg['training']['batch_size']} "
              f"(x{cfg['training']['grad_accum_steps']} accum = "
              f"{cfg['training']['batch_size'] * cfg['training']['grad_accum_steps']} eff)")
        print(f"[train_vlm] world_size : {world_size}  device: {device}")
        print("=" * 60)

    # ── 시드 (rank별로 다르게) ────────────────────────────────────
    seed = cfg["training"].get("seed", 42) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── 데이터셋 ─────────────────────────────────────────────────
    from training.builder import build_vlm_datasets, build_dataloaders_vlm
    train_ds, val_ds = build_vlm_datasets(cfg)
    train_loader, val_loader, train_sampler = build_dataloaders_vlm(
        train_ds, val_ds, cfg, rank=rank, world_size=world_size
    )
    if is_main:
        print(f"[train_vlm] train: {len(train_ds)}  val: {len(val_ds)}")

    # ── 모델 ─────────────────────────────────────────────────────
    from training.builder import build_vlm_model
    model = build_vlm_model(cfg, action_dim=train_ds.action_dim,
                            proprio_dim=train_ds.proprio_dim)
    model.to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    if is_main:
        print("[train_vlm] 모델 생성 완료")

    # ── 학습 ─────────────────────────────────────────────────────
    from training.trainer_vlm import VLMTrainer
    trainer = VLMTrainer(
        model, train_loader, val_loader, cfg, device,
        is_main=is_main, train_sampler=train_sampler,
    )
    trainer.train()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
