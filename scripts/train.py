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

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Using device: {device}")
    print(f"[train.py] Model type: {cfg['model']['type']}")

    # Build datasets
    from training.builder import build_datasets, build_dataloaders, build_model
    print("[train.py] Loading datasets...")
    train_ds, val_ds = build_datasets(cfg)
    print(f"  Train: {train_ds}")
    print(f"  Val:   {val_ds}")

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg)

    # Build model
    print("[train.py] Building model...")
    context_encoder, planner_encoder, policy = build_model(cfg, train_ds.action_dim, train_ds.proprio_dim, train_ds)
    n_params_enc = sum(p.numel() for p in context_encoder.parameters() if p.requires_grad)
    n_params_plan = sum(p.numel() for p in planner_encoder.parameters() if p.requires_grad) if planner_encoder is not context_encoder else 0
    n_params_pol = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Context encoder trainable: {n_params_enc:,}")
    if n_params_plan:
        print(f"  Planner encoder trainable: {n_params_plan:,}")
    print(f"  Policy trainable:          {n_params_pol:,}")

    # Train
    from training.trainer import Trainer
    trainer = Trainer(context_encoder, planner_encoder, policy, train_loader, val_loader, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
