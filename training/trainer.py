"""
Trainer
───────
Handles the full training loop for all model variants.

Training flow per batch:
  1. Encode (image, proprio, language) → C_t  [ContextEncoder]
  2. Optionally encode future_image → future_feat  [SigLIP frozen]
  3. Forward pass through policy → loss dict
  4. Backward + optimizer step

Evaluation:
  - Runs offline metrics every eval_every epochs
  - Saves checkpoint every save_every epochs
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional

from evaluation.metrics import OfflineEvaluator


class Trainer:
    def __init__(
        self,
        context_encoder: nn.Module,
        planner_encoder: nn.Module,
        policy: nn.Module,
        train_loader,
        val_loader,
        cfg: dict,
        device: torch.device = None,
        is_main: bool = True,
        train_sampler=None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_cfg = cfg["training"]
        self.loss_cfg = cfg["loss"]
        self.output_dir = self.train_cfg["output_dir"]
        self.is_main = is_main
        self.train_sampler = train_sampler
        if is_main:
            os.makedirs(self.output_dir, exist_ok=True)

        self.context_encoder = context_encoder.to(self.device)
        self.planner_encoder = planner_encoder.to(self.device)
        self.policy = policy.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._planner_input = self.cfg["model"].get("planner_input", "full")
        self._object_state_dim = self.cfg["model"].get("object_state_dim", 10)
        # DDP 래핑 시 .module로 실제 모델 접근 (커스텀 메서드용)
        self._raw_enc = getattr(context_encoder, "module", context_encoder)
        self._raw_policy = getattr(policy, "module", policy)

        # Optimizer: skip frozen params; avoid duplicate registration when encoders are shared
        trainable = list(p for p in context_encoder.parameters() if p.requires_grad)
        if planner_encoder is not context_encoder:
            trainable += list(p for p in planner_encoder.parameters() if p.requires_grad)
        trainable += list(policy.parameters())
        self.optimizer = AdamW(
            trainable,
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        total_steps = len(train_loader) * self.train_cfg["num_epochs"]
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)

        self.evaluator = OfflineEvaluator(
            context_encoder=self._raw_enc,
            policy=getattr(self.policy, "module", self.policy),
            device=self.device,
        )

        # Logging
        self.use_wandb = cfg["logging"]["use_wandb"]
        if self.use_wandb:
            try:
                import wandb
                run_name = cfg["logging"].get("run_name") or f"{cfg['model']['type']}_{int(time.time())}"
                wandb.init(project=cfg["logging"]["project"], name=run_name, config=cfg)
                self.wandb = wandb
            except ImportError:
                print("[Trainer] wandb not installed, disabling.")
                self.use_wandb = False

        self.global_step = 0
        self.warmup_steps = self.train_cfg.get("warmup_steps", 0)

        # JSONL log
        self._log_path = os.path.join(self.output_dir, "train_log.jsonl")
        self._log_file = open(self._log_path, "w")

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        print(f"[Trainer] Starting training on {self.device}")
        print(f"  Model: {self.cfg['model']['type']}")
        print(f"  Planner input: {self._planner_input}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples:   {len(self.val_loader.dataset)}")

        for epoch in range(1, self.train_cfg["num_epochs"] + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(epoch)

            if not self.is_main:
                continue

            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["epoch"] = epoch

            if epoch % self.train_cfg["eval_every"] == 0:
                val_metrics = self.evaluator.evaluate(self.val_loader)
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                self._print_metrics(epoch, train_metrics, val_metrics)
            else:
                self._print_metrics(epoch, train_metrics, None)

            if self.use_wandb:
                self.wandb.log(log_dict, step=self.global_step)

            self._write_csv(log_dict)

            if epoch % self.train_cfg["save_every"] == 0:
                self._save_checkpoint(epoch)

        if self.is_main:
            print("[Trainer] Training complete.")
            self._save_checkpoint("final")
            self._log_file.close()
            print(f"[Trainer] Log saved: {self._log_path}")
            self._generate_result()

    def _generate_result(self):
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.generate_result import generate
            generate(self.output_dir, self.cfg.get("model", {}).get("type"))
        except Exception as e:
            print(f"[Trainer] result 생성 실패 (무시): {e}")

    # ── One epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.context_encoder.train()
        if self.planner_encoder is not self.context_encoder:
            self.planner_encoder.train()
        self.policy.train()

        accum = {}
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            metrics = self._train_step(batch)

            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            pbar.set_postfix(loss=f"{metrics.get('total_loss', 0):.4f}")

        return {k: v / n for k, v in accum.items()}

    def _train_step(self, batch: dict) -> dict:
        # Support both image-mode and low_dim-mode batches
        if "image" in batch:
            image = batch["image"].to(self.device)
        else:
            image = batch["state"].to(self.device)     # low_dim: state as image slot
        proprio = batch["proprio"].to(self.device)
        actions = batch["actions"].to(self.device)
        future_input = batch["future_image"].to(self.device)  # future_image or future_state

        # Language tokenization (no-op for StateContextEncoder)
        langs = list(batch["language"])
        input_ids, attention_mask = self._raw_enc.tokenize(langs, self.device)

        # Context encoding (full state → action decoder context)
        context = self.context_encoder(image, proprio, input_ids, attention_mask)

        # Planner context (subset of state → planner prior/posterior)
        if self.planner_encoder is self.context_encoder:
            planner_context = context
        else:
            if self._planner_input == "object_only":
                planner_state = image[:, :self._object_state_dim]
            else:  # proprio_only
                planner_state = image[:, self._object_state_dim:]
            planner_context = self.planner_encoder(planner_state, planner_state)

        # Future feature for semantic loss
        future_feat = None
        if self.loss_cfg.get("semantic_future_weight", 0.0) > 0:
            with torch.no_grad():
                future_feat = self._raw_enc.encode_future_image(future_input)

        # Policy loss — kwargs vary by model type
        model_type = self.cfg["model"]["type"]
        sem_w = self.loss_cfg.get("semantic_future_weight", 0.1)
        if model_type == "flat_flow":
            loss_dict = self._raw_policy.compute_loss(context=context, actions=actions)
        elif model_type == "det_latent":
            loss_dict = self._raw_policy.compute_loss(
                context=context, actions=actions, future_feat=future_feat,
                semantic_weight=sem_w,
                prior_weight=self.loss_cfg.get("prior_weight", 1.0),
            )
        elif model_type == "stoch_vae":
            loss_dict = self._raw_policy.compute_loss(
                context=context, actions=actions, future_feat=future_feat,
                semantic_weight=sem_w,
                kl_weight=self.loss_cfg.get("kl_beta", 1.0),
            )
        elif model_type == "stoch_flow_prior":
            loss_dict = self._raw_policy.compute_loss(
                context=context, actions=actions, future_feat=future_feat,
                planner_context=planner_context,
                semantic_weight=sem_w,
                prior_weight=self.loss_cfg.get("prior_weight", 1.0),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters()) + list(self.policy.parameters()),
            self.train_cfg["grad_clip"],
        )

        # Warmup
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.train_cfg["learning_rate"] * lr_scale

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {k: v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 0}

    # ── Utilities ─────────────────────────────────────────────────────────

    def _print_metrics(self, epoch, train, val=None):
        msg = f"[Epoch {epoch:3d}]"
        for k, v in train.items():
            msg += f"  {k}={v:.4f}"
        if val:
            msg += "  |  VAL:"
            for k, v in val.items():
                if isinstance(v, float):
                    msg += f"  {k}={v:.4f}"
        print(msg)

    def _write_csv(self, log_dict: dict):
        row = {k: round(v, 6) if isinstance(v, float) else v for k, v in log_dict.items()}
        self._log_file.write(json.dumps(row) + "\n")
        self._log_file.flush()

    def _save_checkpoint(self, tag):
        path = os.path.join(self.output_dir, f"ckpt_{tag}.pt")
        # DDP 래핑 시 .module로 실제 모델 접근
        enc = getattr(self.context_encoder, "module", self.context_encoder)
        pol = getattr(self.policy, "module", self.policy)
        plan = getattr(self.planner_encoder, "module", self.planner_encoder)
        torch.save({
            "context_encoder": enc.state_dict(),
            "planner_encoder": plan.state_dict() if self.planner_encoder is not self.context_encoder else None,
            "policy": pol.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }, path)
        print(f"[Trainer] Saved checkpoint: {path}")
