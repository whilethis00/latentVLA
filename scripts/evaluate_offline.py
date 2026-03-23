"""
Offline evaluation script with best-of-K and temperature sweep.

Usage:
    # Single eval
    python scripts/evaluate_offline.py --checkpoint outputs/runs/stoch_flow_prior/ckpt_final.pt

    # best-of-K sweep (K=1,3,5,10)
    python scripts/evaluate_offline.py --checkpoint outputs/runs/stoch_flow_prior/ckpt_final.pt --sweep_k

    # temperature sweep
    python scripts/evaluate_offline.py --checkpoint outputs/runs/stoch_flow_prior/ckpt_final.pt --sweep_temp

    # both
    python scripts/evaluate_offline.py --checkpoint outputs/runs/stoch_flow_prior/ckpt_final.pt --sweep_k --sweep_temp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch

from training.builder import build_datasets, build_dataloaders, build_model
from evaluation.metrics import OfflineEvaluator

DATA_PATH = "/home/introai4/home_lustre/introai4/datasets/lift/ph/low_dim_v141.hdf5"


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    cfg["data"]["dataset_path"] = DATA_PATH
    cfg["training"]["batch_size"] = 32

    train_ds, val_ds = build_datasets(cfg)
    _, val_loader = build_dataloaders(train_ds, val_ds, cfg)

    enc, policy = build_model(cfg, train_ds.action_dim, train_ds.proprio_dim, train_ds)
    enc.load_state_dict(ckpt["context_encoder"])
    policy.load_state_dict(ckpt["policy"])
    enc = enc.to(device)
    policy = policy.to(device)

    return enc, policy, val_loader, cfg


def run_eval(enc, policy, val_loader, device, best_of_ks, std_scale):
    evaluator = OfflineEvaluator(
        enc, policy, device,
        n_diversity_samples=max(best_of_ks),
        best_of_ks=best_of_ks,
        std_scale=std_scale,
    )
    return evaluator.evaluate(val_loader)


def print_metrics(metrics, indent="  "):
    for k, v in metrics.items():
        print(f"{indent}{k:35s} {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sweep_k", action="store_true", help="Sweep best-of-K for K=1,3,5,10")
    parser.add_argument("--sweep_temp", action="store_true", help="Sweep std_scale=1.0,0.7,0.5,0.3")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--temps", nargs="+", type=float, default=[1.0, 0.7, 0.5, 0.3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    enc, policy, val_loader, cfg = load_model(args.checkpoint, device)
    model_type = cfg["model"]["type"]
    print(f"Model: {model_type}\n")

    results = {}

    if args.sweep_k:
        print("=== best-of-K sweep (std_scale=1.0) ===")
        metrics = run_eval(enc, policy, val_loader, device, best_of_ks=args.ks, std_scale=1.0)
        print_metrics(metrics)
        results["best_of_k_sweep"] = metrics
        print()

    if args.sweep_temp:
        print("=== temperature sweep (best-of-5) ===")
        for temp in args.temps:
            print(f"  std_scale={temp}")
            metrics = run_eval(enc, policy, val_loader, device, best_of_ks=[1, 5], std_scale=temp)
            key_metrics = {k: v for k, v in metrics.items()
                           if k in ("action_mse_prior", "best_of_1", "best_of_5", "sampling_diversity")}
            print_metrics(key_metrics, indent="    ")
            results[f"temp_{temp}"] = metrics
        print()

    if not args.sweep_k and not args.sweep_temp:
        print("=== Single eval (K=1,3,5,10 | std_scale=1.0) ===")
        metrics = run_eval(enc, policy, val_loader, device, best_of_ks=[1, 3, 5, 10], std_scale=1.0)
        print_metrics(metrics)
        results["single"] = metrics

    # Save
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
