"""
Main training script.

Usage:
    python scripts/train.py --config configs/default.yaml [--overrides key=value ...]

Examples:
    # Flat-Flow baseline on robomimic
    python scripts/train.py \\
        --config configs/default.yaml \\
        --override model.type=flat_flow \\
                   data.dataset_path=/path/to/lift.hdf5 \\
                   data.dataset_type=robomimic

    # Det-Latent on LIBERO-Object
    python scripts/train.py \\
        --config configs/default.yaml \\
        --override model.type=det_latent \\
                   data.dataset_type=libero \\
                   data.dataset_path=/path/to/libero_object/

    # Stoch-FlowPrior on LIBERO-Long
    python scripts/train.py \\
        --config configs/default.yaml \\
        --override model.type=stoch_flow_prior \\
                   data.dataset_type=libero \\
                   data.dataset_path=/path/to/libero_long/
"""

import sys
import os

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list):
    """Apply key=value overrides to nested config dict."""
    for ov in overrides:
        key, val = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Type coercion
        orig = d.get(keys[-1])
        if isinstance(orig, bool):
            val = val.lower() in ("true", "1", "yes")
        elif isinstance(orig, int):
            val = int(val)
        elif isinstance(orig, float):
            val = float(val)
        d[keys[-1]] = val
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # ── DDP 초기화 (torchrun 시 LOCAL_RANK 환경변수 존재) ──────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    set_seed(cfg["training"]["seed"] + (dist.get_rank() if is_ddp else 0))

    if is_main:
        print(f"[train.py] Device: {device}  DDP: {is_ddp}"
              + (f"  world_size: {dist.get_world_size()}" if is_ddp else ""))
        print(f"[train.py] Model type: {cfg['model']['type']}")

    # Build datasets
    from training.builder import build_datasets, build_dataloaders, build_model
    if is_main:
        print("[train.py] Loading datasets...")
    train_ds, val_ds = build_datasets(cfg)

    # DDP용 DistributedSampler
    train_sampler = None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg, train_sampler=train_sampler)

    if is_main:
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Build model
    if is_main:
        print("[train.py] Building model...")
    context_encoder, planner_encoder, policy = build_model(cfg, train_ds.action_dim, train_ds.proprio_dim, train_ds)

    # DDP 래핑
    if is_ddp:
        same_encoder = (planner_encoder is context_encoder)
        from torch.nn.parallel import DistributedDataParallel as DDP
        context_encoder = DDP(context_encoder.to(device), device_ids=[local_rank], find_unused_parameters=True)
        policy = DDP(policy.to(device), device_ids=[local_rank], find_unused_parameters=True)
        if same_encoder:
            planner_encoder = context_encoder  # 동일 객체 유지
        else:
            planner_encoder = DDP(planner_encoder.to(device), device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        n_pol = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"  Policy trainable: {n_pol:,}")

    # Train
    from training.trainer import Trainer
    trainer = Trainer(
        context_encoder, planner_encoder, policy,
        train_loader, val_loader, cfg, device,
        is_main=is_main, train_sampler=train_sampler,
    )
    trainer.train()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
