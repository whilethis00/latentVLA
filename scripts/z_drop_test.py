"""
z_drop test — decoder의 z 의존도 측정.

prior 경로와 posterior 경로 각각에서 z를 null(zeros)로 교체했을 때
action MSE 변화를 측정한다. z를 빼도 MSE가 거의 안 오르면 decoder non-usage 확인.

사용법:
  conda run -n vla python3 scripts/z_drop_test.py \\
      --config configs/vlm_paligemma_infonce_balanced.yaml \\
      --ckpt   outputs/runs/vlm_sfp_infonce_balanced_20260416/ckpt_80.pt \\
      --n_batches 50 --seed 42
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--steps",     type=int, default=None, help="ODE steps (default: model default)")
    p.add_argument("--out",       default=None, help="결과 저장 json 경로")
    return p.parse_args()


@torch.no_grad()
def run_z_drop_test(model, val_loader, device, n_batches, steps, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    system1 = model.system1
    flow_steps = steps or system1.flow_steps
    z_dim = system1.z_dim

    results = {
        "prior_baseline":    [],
        "prior_null_z":      [],
        "posterior_baseline":[],
        "posterior_null_z":  [],
    }

    model.eval()
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break

        actions  = batch["actions"].to(device)
        proprio  = batch["proprio"].to(device)
        raw_image = batch.get("raw_image", batch["image"])
        B = actions.shape[0]

        # VLM encoding
        pv, ids, mask = model.system2.prepare_inputs(raw_image, batch["language"], device)
        f_tilde = model.system2(pv, ids, mask, proprio)

        null_z = torch.zeros(B, z_dim, device=device)

        # ── Prior path ────────────────────────────────────────────────
        # baseline: z from prior flow
        from models.flow_utils import euler_integrate
        torch.manual_seed(seed + i)
        z_prior = euler_integrate(system1.prior_flow, f_tilde, z_dim, flow_steps)
        cond_prior = torch.cat([f_tilde, z_prior], dim=-1)
        torch.manual_seed(seed + i + 10000)
        a_prior = euler_integrate(system1.action_flow, cond_prior, system1.x_dim, flow_steps)
        a_prior = a_prior.reshape(B, system1.action_horizon, -1)
        results["prior_baseline"].append(F.mse_loss(a_prior, actions).item())

        # null: z → zeros
        cond_null = torch.cat([f_tilde, null_z], dim=-1)
        torch.manual_seed(seed + i + 10000)  # same noise seed as baseline
        a_null = euler_integrate(system1.action_flow, cond_null, system1.x_dim, flow_steps)
        a_null = a_null.reshape(B, system1.action_horizon, -1)
        results["prior_null_z"].append(F.mse_loss(a_null, actions).item())

        # ── Posterior path ────────────────────────────────────────────
        # baseline: z from posterior (mu, no sampling noise for consistency)
        future_feat = None
        if "future_image" in batch and system1.use_future:
            future_feat = model._siglip.encode_image_only(batch["future_image"].to(device))

        parts = [f_tilde, actions.reshape(B, -1)]
        if future_feat is not None:
            parts.append(future_feat)
        q_in = torch.cat(parts, -1)
        mu_q, _ = system1.posterior_enc(q_in)  # use mu (deterministic)

        cond_post = torch.cat([f_tilde, mu_q], dim=-1)
        torch.manual_seed(seed + i + 20000)
        a_post = euler_integrate(system1.action_flow, cond_post, system1.x_dim, flow_steps)
        a_post = a_post.reshape(B, system1.action_horizon, -1)
        results["posterior_baseline"].append(F.mse_loss(a_post, actions).item())

        # null: posterior z → zeros
        cond_post_null = torch.cat([f_tilde, null_z], dim=-1)
        torch.manual_seed(seed + i + 20000)  # same noise seed
        a_post_null = euler_integrate(system1.action_flow, cond_post_null, system1.x_dim, flow_steps)
        a_post_null = a_post_null.reshape(B, system1.action_horizon, -1)
        results["posterior_null_z"].append(F.mse_loss(a_post_null, actions).item())

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from training.builder import build_vlm_datasets, build_dataloaders_vlm, build_vlm_model
    from torch.utils.data import DataLoader

    _, val_ds = build_vlm_datasets(cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    model = build_vlm_model(cfg, action_dim=val_ds.action_dim, proprio_dim=val_ds.proprio_dim)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict_from_save(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"[z_drop_test] checkpoint: {args.ckpt}")
    print(f"[z_drop_test] n_batches={args.n_batches}  seed={args.seed}")
    print()

    res = run_z_drop_test(model, val_loader, device, args.n_batches, args.steps, args.seed)

    prior_delta    = res["prior_null_z"]    - res["prior_baseline"]
    posterior_delta = res["posterior_null_z"] - res["posterior_baseline"]

    print("=" * 55)
    print(f"  Prior path")
    print(f"    baseline (z from prior) : {res['prior_baseline']:.4f}")
    print(f"    null z (zeros)          : {res['prior_null_z']:.4f}")
    print(f"    Δ MSE (↑ = z 의존)      : {prior_delta:+.4f}")
    print()
    print(f"  Posterior path")
    print(f"    baseline (z = mu_q)     : {res['posterior_baseline']:.4f}")
    print(f"    null z (zeros)          : {res['posterior_null_z']:.4f}")
    print(f"    Δ MSE (↑ = z 의존)      : {posterior_delta:+.4f}")
    print("=" * 55)
    print()

    if prior_delta < 0.01 and posterior_delta < 0.01:
        print("판정: ❌ decoder non-usage — z를 빼도 MSE 거의 안 변함")
        print("       FiLM/CFG 등 binding 장치 없이는 InfoNCE 튜닝이 의미 없음")
    elif prior_delta < 0.01:
        print("판정: ⚠ prior path z 미사용 — posterior는 반응하나 prior는 non-usage")
    elif posterior_delta < 0.01:
        print("판정: ⚠ posterior path z 미사용 — prior는 반응하나 posterior는 non-usage")
    else:
        print("판정: ✅ decoder가 z를 실제로 사용 중")

    full = {**res, "prior_delta": prior_delta, "posterior_delta": posterior_delta}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(full, f, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
