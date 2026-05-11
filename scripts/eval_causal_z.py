"""
Evaluate causal z interventions for LatentVLA checkpoints.

Examples:
    python scripts/eval_causal_z.py \
        --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
        --mode prior --intervention shuffle --max_batches 30

    python scripts/eval_causal_z.py \
        --checkpoint outputs/runs/vlm_sfp_infonce_s1only_20260418/ckpt_90.pt \
        --mode both --intervention null
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.flow_utils import euler_integrate
from training.builder import build_dataloaders_vlm, build_vlm_datasets, build_vlm_model


def load_model_and_data(ckpt_path, device, data_path_override=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    if data_path_override:
        cfg.setdefault("data", {})["dataset_path"] = data_path_override

    train_ds, val_ds = build_vlm_datasets(cfg)
    _, val_loader, _ = build_dataloaders_vlm(train_ds, val_ds, cfg)
    model = build_vlm_model(
        cfg,
        action_dim=train_ds.action_dim,
        proprio_dim=train_ds.proprio_dim,
    )

    saved = ckpt["model"]
    if saved.get("system2_lora"):
        model.system2.enable_lora()
    model.load_state_dict_from_save(saved)
    model = model.to(device).eval()
    return model, val_loader, cfg


@torch.no_grad()
def encode_batch(model, batch, device):
    actions = batch["actions"].to(device)
    proprio = batch["proprio"].to(device)
    raw_image = batch.get("raw_image", batch["image"])
    pixel_values, input_ids, attn_mask = model.system2.prepare_inputs(
        raw_image, batch["language"], device
    )
    context = model.system2(pixel_values, input_ids, attn_mask, proprio)

    future_feat = None
    if model.system1.use_future and "future_image" in batch:
        future_feat = model._siglip.encode_image_only(batch["future_image"].to(device))

    return context, actions, future_feat


@torch.no_grad()
def posterior_z(model, context, actions, future_feat):
    batch_size = actions.shape[0]
    parts = [context, actions.reshape(batch_size, -1)]
    if model.system1.use_future and future_feat is not None:
        parts.append(future_feat)
    q_in = torch.cat(parts, dim=-1)
    mu_q, logvar_q = model.system1.posterior_enc(q_in)
    z = mu_q + (0.5 * logvar_q).exp() * torch.randn_like(mu_q)
    return z, mu_q, logvar_q


@torch.no_grad()
def prior_z(model, context):
    return euler_integrate(
        model.system1.prior_flow,
        context,
        model.system1.z_dim,
        model.system1.flow_steps,
    )


@torch.no_grad()
def decode_actions(model, context, z):
    cond = torch.cat([context, z], dim=-1)
    pred = euler_integrate(
        model.system1.action_flow,
        cond,
        model.system1.x_dim,
        model.system1.flow_steps,
    )
    return pred.reshape(context.shape[0], model.system1.action_horizon, -1)


def shuffled_indices(batch_size, device):
    if batch_size < 2:
        return torch.arange(batch_size, device=device)
    perm = torch.randperm(batch_size, device=device)
    if torch.equal(perm, torch.arange(batch_size, device=device)):
        perm = torch.roll(perm, shifts=1)
    return perm


def task_negative_indices(batch, batch_size, device):
    task_ids = batch.get("task_id")
    if task_ids is None:
        langs = list(batch["language"])
        task_ids = torch.tensor(
            [abs(hash(x)) % 1000003 for x in langs],
            device=device,
            dtype=torch.long,
        )
    else:
        task_ids = task_ids.to(device)

    perm = shuffled_indices(batch_size, device)
    for i in range(batch_size):
        candidates = torch.where(task_ids != task_ids[i])[0]
        if candidates.numel() > 0:
            choice = torch.randint(candidates.numel(), (1,), device=device).item()
            perm[i] = candidates[choice]
    return perm


def motion_negative_indices(actions):
    batch_size = actions.shape[0]
    if batch_size < 2:
        return torch.arange(batch_size, device=actions.device)
    flat = actions.reshape(batch_size, -1).float()
    dist = torch.cdist(flat, flat)
    dist.fill_diagonal_(-1.0)
    return dist.argmax(dim=1)


def intervene_z(z, intervention, batch, actions):
    batch_size = z.shape[0]
    device = z.device
    if intervention == "shuffle":
        return z[shuffled_indices(batch_size, device)]
    if intervention == "null":
        return torch.zeros_like(z)
    if intervention == "random":
        return torch.randn_like(z)
    if intervention == "task_negative":
        return z[task_negative_indices(batch, batch_size, device)]
    if intervention == "motion_negative":
        return z[motion_negative_indices(actions)]
    raise ValueError("Unknown intervention: {}".format(intervention))


def update_probe_stats(probe_values, mu_q, batch):
    batch_size = mu_q.shape[0]
    if batch_size < 2:
        return
    device = mu_q.device
    task_ids = batch.get("task_id")
    if task_ids is not None:
        task_ids = task_ids.to(device)
        same_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)
    else:
        langs = list(batch["language"])
        same_mask = torch.tensor(
            [[a == b for b in langs] for a in langs],
            device=device,
            dtype=torch.bool,
        )
    same_mask.fill_diagonal_(False)
    offdiag = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    if same_mask.any() and offdiag.any():
        dist = torch.cdist(mu_q.float(), mu_q.float())
        probe_values.append((dist[same_mask].mean() / (dist[offdiag].mean() + 1e-8)).item())


@torch.no_grad()
def evaluate_mode(model, dataloader, device, mode, intervention, max_batches=None):
    metrics = {
        "mse_original": [],
        "mse_intervened": [],
        "delta": [],
        "ratio": [],
        "z_norm": [],
        "z_var": [],
        "probe_ratio": [],
    }

    for i, batch in enumerate(tqdm(dataloader, desc="{}:{}".format(mode, intervention), leave=False)):
        if max_batches is not None and i >= max_batches:
            break

        context, actions, future_feat = encode_batch(model, batch, device)
        if mode == "prior":
            z = prior_z(model, context)
            mu_q = None
        elif mode == "posterior":
            z, mu_q, _ = posterior_z(model, context, actions, future_feat)
            update_probe_stats(metrics["probe_ratio"], mu_q, batch)
        else:
            raise ValueError("Unknown mode: {}".format(mode))

        z_int = intervene_z(z, intervention, batch, actions)
        pred = decode_actions(model, context, z)
        pred_int = decode_actions(model, context, z_int)

        mse = F.mse_loss(pred, actions).item()
        mse_int = F.mse_loss(pred_int, actions).item()
        metrics["mse_original"].append(mse)
        metrics["mse_intervened"].append(mse_int)
        metrics["delta"].append(mse_int - mse)
        metrics["ratio"].append(mse_int / (mse + 1e-8))
        metrics["z_norm"].append(z.norm(dim=-1).mean().item())
        metrics["z_var"].append(z.var(dim=0).mean().item())

    return {
        key: float(np.mean(vals))
        for key, vals in metrics.items()
        if vals
    }


def write_outputs(results, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "causal_z_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    md_path = os.path.join(out_dir, "causal_z_report.md")
    lines = [
        "# Causal z intervention report",
        "",
        "- checkpoint: `{}`".format(args.checkpoint),
        "- intervention: `{}`".format(args.intervention),
        "- max_batches: `{}`".format(args.max_batches),
        "- generated_at: `{}`".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "",
        "| mode | mse_original | mse_intervened | delta | ratio | z_norm | z_var | probe_ratio |",
        "|------|--------------|----------------|-------|-------|--------|-------|-------------|",
    ]
    for mode in sorted(results):
        row = results[mode]
        lines.append(
            "| {mode} | {mse_original:.6f} | {mse_intervened:.6f} | {delta:+.6f} | "
            "{ratio:.4f} | {z_norm:.6f} | {z_var:.6f} | {probe_ratio} |".format(
                mode=mode,
                mse_original=row.get("mse_original", float("nan")),
                mse_intervened=row.get("mse_intervened", float("nan")),
                delta=row.get("delta", float("nan")),
                ratio=row.get("ratio", float("nan")),
                z_norm=row.get("z_norm", float("nan")),
                z_var=row.get("z_var", float("nan")),
                probe_ratio=(
                    "{:.4f}".format(row["probe_ratio"])
                    if "probe_ratio" in row else "-"
                ),
            )
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--mode", choices=["prior", "posterior", "both"], default="both")
    parser.add_argument(
        "--intervention",
        choices=["shuffle", "null", "random", "task_negative", "motion_negative"],
        default="shuffle",
    )
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, dataloader, _ = load_model_and_data(args.checkpoint, device, args.data_path)

    modes = ["prior", "posterior"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        results[mode] = evaluate_mode(
            model,
            dataloader,
            device,
            mode=mode,
            intervention=args.intervention,
            max_batches=args.max_batches,
        )

    run_name = os.path.basename(os.path.dirname(args.checkpoint))
    default_out = os.path.join(
        "outputs",
        "analyses",
        "causal_z_{}_{}_{}".format(run_name, args.intervention, datetime.now().strftime("%Y%m%d")),
    )
    out_dir = args.out_dir or default_out
    json_path, md_path = write_outputs(results, args, out_dir)

    print("[eval_causal_z] JSON: {}".format(json_path))
    print("[eval_causal_z] Report: {}".format(md_path))
    for mode in modes:
        row = results[mode]
        print(
            "[{}] mse={:.6f} intervened={:.6f} delta={:+.6f} ratio={:.4f}".format(
                mode,
                row.get("mse_original", float("nan")),
                row.get("mse_intervened", float("nan")),
                row.get("delta", float("nan")),
                row.get("ratio", float("nan")),
            )
        )


if __name__ == "__main__":
    main()
