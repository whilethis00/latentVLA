"""
VLMTrainer — 2-stage 학습 루프
───────────────────────────────
Stage 1 (epoch 1 ~ stage2_epoch-1): VLM frozen, flow heads만 학습
Stage 2 (epoch stage2_epoch ~     ): VLM LoRA 활성화, 두 그룹 optimizer

추가 기능:
  - bf16 mixed precision
  - gradient accumulation
  - LoRA/flow 별도 LR
  - LoRA 가중치만 저장 (전체 PaliGemma 저장 X)
"""

import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.latent_vla import LatentVLA
from evaluation.metrics import OfflineEvaluator


class _Tee:
    """stdout을 화면과 파일에 동시에 출력."""
    def __init__(self, filepath):
        self._file = open(filepath, "a", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

    # tqdm 등이 참조하는 속성 위임
    def __getattr__(self, name):
        return getattr(self._stdout, name)


class VLMOfflineEvaluator:
    """
    LatentVLA용 OfflineEvaluator 어댑터.
    기존 OfflineEvaluator는 context_encoder + policy 분리를 가정하므로,
    LatentVLA의 batch 기반 인터페이스에 맞게 래핑.
    """

    def __init__(self, model: LatentVLA, device: torch.device, best_of_ks=None):
        self.model = model
        self.device = device
        self.best_of_ks = best_of_ks or [1, 5]

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches: int = None) -> dict:
        self.model.eval()
        import numpy as np
        import torch.nn.functional as F

        metrics = {
            "action_mse_prior": [],
            "action_mse_posterior": [],
            "sampling_diversity": [],
            **{f"best_of_{k}": [] for k in self.best_of_ks},
            "future_cosine_sim": [],
            "z_shuffle_gap": [],
        }

        for i, batch in enumerate(tqdm(dataloader, desc="VLM Eval", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            actions = batch["actions"].to(self.device)
            B = actions.shape[0]

            # f̃ 인코딩 (한 번만)
            proprio = batch["proprio"].to(self.device)
            raw_image = batch.get("raw_image", batch["image"])
            pv, ids, mask = self.model.system2.prepare_inputs(
                raw_image, batch["language"], self.device
            )
            f_tilde = self.model.system2(pv, ids, mask, proprio)

            # ── action MSE (prior) ─────────────────────────────────────────
            pred_prior = self.model.system1.predict(
                context=f_tilde, planner_context=f_tilde
            )
            metrics["action_mse_prior"].append(
                F.mse_loss(pred_prior, actions).item()
            )

            # ── action MSE (posterior) ─────────────────────────────────────
            future_feat = None
            if "future_image" in batch and self.model.system1.use_future:
                future_feat = self.model._siglip.encode_image_only(
                    batch["future_image"].to(self.device)
                )
            pred_post = self.model.system1.predict(
                context=f_tilde,
                planner_context=f_tilde,
                use_posterior=True,
                actions_for_posterior=actions,
                future_feat_for_posterior=future_feat,
            )
            metrics["action_mse_posterior"].append(
                F.mse_loss(pred_post, actions).item()
            )

            # ── future cosine sim ──────────────────────────────────────────
            if future_feat is not None and hasattr(self.model.system1, "semantic_head"):
                from models.stoch_latent_vae import GaussianEncoder
                parts = [f_tilde, actions.reshape(B, -1)]
                if future_feat is not None:
                    parts.append(future_feat)
                q_in = torch.cat(parts, -1)
                mu_q, _ = self.model.system1.posterior_enc(q_in)
                pred_future = self.model.system1.semantic_head(mu_q)
                cos = F.cosine_similarity(
                    F.normalize(pred_future, dim=-1),
                    F.normalize(future_feat, dim=-1),
                    dim=-1,
                ).mean().item()
                metrics["future_cosine_sim"].append(cos)

            # ── z-shuffle gap ──────────────────────────────────────────────
            if B >= 2:
                from models.flow_utils import euler_integrate
                z_normal = euler_integrate(
                    self.model.system1.prior_flow, f_tilde,
                    self.model.system1.z_dim, self.model.system1.flow_steps
                )
                perm = torch.randperm(B, device=self.device)
                z_shuffled = z_normal[perm]

                cond_normal = torch.cat([f_tilde, z_normal], dim=-1)
                cond_shuffled = torch.cat([f_tilde, z_shuffled], dim=-1)
                x_dim = self.model.system1.x_dim

                a_normal = euler_integrate(
                    self.model.system1.action_flow, cond_normal, x_dim,
                    self.model.system1.flow_steps
                ).reshape(B, self.model.system1.action_horizon, -1)
                a_shuffled = euler_integrate(
                    self.model.system1.action_flow, cond_shuffled, x_dim,
                    self.model.system1.flow_steps
                ).reshape(B, self.model.system1.action_horizon, -1)

                gap = (F.mse_loss(a_shuffled, actions) -
                       F.mse_loss(a_normal, actions)).item()
                metrics["z_shuffle_gap"].append(gap)

            # ── diversity + best-of-K ──────────────────────────────────────
            max_k = max(self.best_of_ks)
            samples = self.model.system1.predict(
                context=f_tilde, planner_context=f_tilde,
                n_samples=max_k,
            )
            if samples.dim() == 4:  # (B, K, H, D)
                std = samples.std(dim=1).mean().item()
                metrics["sampling_diversity"].append(std)
                gt = actions.unsqueeze(1)
                mse_per_k = ((samples - gt) ** 2).mean(dim=(-2, -1))
                for k in self.best_of_ks:
                    best = mse_per_k[:, :k].min(dim=1).values.mean().item()
                    metrics[f"best_of_{k}"].append(best)

        result = {k: float(np.mean(v)) for k, v in metrics.items() if v}
        if "action_mse_prior" in result and "action_mse_posterior" in result:
            result["prior_posterior_gap"] = (
                result["action_mse_prior"] - result["action_mse_posterior"]
            )
        return result


class VLMTrainer:
    """
    LatentVLA 2-stage 학습 루프.

    Stage 1: VLM frozen → flow heads (proj + system1) 만 학습
    Stage 2: LoRA 활성화 → LoRA params (낮은 LR) + flow heads (높은 LR) 학습

    DDP: torchrun으로 실행 시 is_main=True인 rank 0만 저장/로깅 수행.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: dict,
        device: torch.device = None,
        is_main: bool = True,
        train_sampler=None,
        resume: str = None,
        resume_epoch: int = None,
    ):
        self.model = model
        # DDP로 래핑된 경우 raw model은 .module
        self.raw_model = model.module if hasattr(model, "module") else model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main = is_main
        self.train_sampler = train_sampler
        self.resume = resume
        self.resume_epoch = resume_epoch

        self.train_cfg = cfg["training"]
        self.loss_cfg = cfg.get("loss", {})
        self.output_dir = self.train_cfg["output_dir"]
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)

        self.stage2_epoch = self.train_cfg.get("stage2_epoch", 10)
        self.grad_accum = self.train_cfg.get("grad_accum_steps", 4)
        self.semantic_weight = self.loss_cfg.get("semantic_future_weight", 0.1)

        # Stage 1 optimizer (VLM frozen → non-VLM params만)
        self.optimizer = self._build_stage1_optimizer()

        total_steps = (
            len(train_loader) // self.grad_accum * self.train_cfg["num_epochs"]
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )

        self.evaluator = VLMOfflineEvaluator(
            model=self.raw_model,
            device=self.device,
            best_of_ks=self.train_cfg.get("best_of_ks", [1, 5]),
        )

        # wandb (rank 0만)
        self.use_wandb = self.is_main and cfg.get("logging", {}).get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                run_name = (cfg.get("logging", {}).get("run_name")
                            or f"vlm_{cfg['system2']['z_form']}_{int(time.time())}")
                wandb.init(
                    project=cfg.get("logging", {}).get("project", "latentVLA"),
                    name=run_name, config=cfg,
                )
                self.wandb = wandb
            except Exception:
                self.use_wandb = False

        self.global_step = 0
        self._log_path = os.path.join(self.output_dir, "train_log.jsonl")
        log_mode = "a" if (self.is_main and resume) else "w"
        self._log_file = open(self._log_path, log_mode) if self.is_main else None

        # 터미널 출력 전체를 train.log에 동시 저장 (rank 0만)
        self._tee = None
        if self.is_main:
            log_path = os.path.join(self.output_dir, "train.log")
            self._tee = _Tee(log_path)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[VLMTrainer] 로그 파일: {log_path}  (시작: {ts})")

    # ── Optimizer 빌드 ────────────────────────────────────────────────────────

    def _build_stage1_optimizer(self):
        params = list(self.raw_model.non_vlm_parameters())
        return AdamW(
            params,
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
        )

    def _build_stage2_optimizer(self):
        """Stage 2: LoRA (작은 LR) + 나머지 (큰 LR) 두 그룹."""
        lora_params = self.raw_model.vlm_lora_parameters()
        non_vlm_params = list(self.raw_model.non_vlm_parameters())
        param_groups = [
            {"params": non_vlm_params, "lr": self.train_cfg["learning_rate"]},
        ]
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": self.train_cfg.get("lora_lr", 3e-5),
            })
        return AdamW(
            param_groups,
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
        )

    # ── 메인 학습 루프 ────────────────────────────────────────────────────────

    def train(self):
        print(f"[VLMTrainer] 학습 시작 — device: {self.device}")
        print(f"  z_form: {self.cfg['system2']['z_form']}")
        print(f"  Stage 2 시작 epoch: {self.stage2_epoch}")
        print(f"  Grad accum steps: {self.grad_accum}")
        print(f"  Train: {len(self.train_loader.dataset)}  "
              f"Val: {len(self.val_loader.dataset)}")

        start_epoch = 1
        if self.resume:
            ckpt = torch.load(self.resume, map_location=self.device)
            self.raw_model.load_state_dict_from_save(ckpt["model"])
            saved_epoch = ckpt.get("epoch") or self.resume_epoch
            if saved_epoch is None:
                raise ValueError(
                    "체크포인트에 'epoch' 키가 없습니다. --resume_epoch으로 직접 지정하세요.\n"
                    "  예: --resume_epoch 80"
                )
            start_epoch = saved_epoch + 1
            # Stage 2 이후 체크포인트라면 LoRA 활성화 및 optimizer 재빌드
            if start_epoch > self.stage2_epoch:
                self.raw_model.system2.enable_lora()
                self.optimizer = self._build_stage2_optimizer()
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.is_main:
                print(f"[VLMTrainer] Resume: {self.resume}  (epoch {start_epoch}부터 재개)")

        for epoch in range(start_epoch, self.train_cfg["num_epochs"] + 1):

            # DistributedSampler는 epoch마다 shuffle seed 갱신 필요
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Stage 2 진입
            if epoch == self.stage2_epoch:
                if self.is_main:
                    print(f"\n[VLMTrainer] === Stage 2 시작 (epoch {epoch}) ===")
                self.raw_model.system2.enable_lora()
                self.optimizer = self._build_stage2_optimizer()
                total_steps = (
                    len(self.train_loader) // self.grad_accum
                    * (self.train_cfg["num_epochs"] - epoch + 1)
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=total_steps, eta_min=1e-6
                )

            train_metrics = self._train_epoch(epoch)
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["epoch"] = epoch

            if epoch % self.train_cfg.get("eval_every", 5) == 0:
                val_metrics = self.evaluator.evaluate(self.val_loader)
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                if self.is_main:
                    self._print(epoch, train_metrics, val_metrics)
            else:
                if self.is_main:
                    self._print(epoch, train_metrics)

            if self.is_main:
                if self.use_wandb:
                    self.wandb.log(log_dict, step=self.global_step)
                self._write_log(log_dict)

                if epoch % self.train_cfg.get("save_every", 10) == 0:
                    self._save(epoch)

        if self.is_main:
            self._save("final")
            self._log_file.close()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[VLMTrainer] 완료. JSONL: {self._log_path}  (종료: {ts})")
            if self._tee:
                self._tee.close()
            self._generate_result()

    def _generate_result(self):
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.generate_result import generate
            z_form = self.cfg.get("system2", {}).get("z_form", "plan")
            generate(self.output_dir, f"vlm_sfp_{z_form}")
        except Exception as e:
            print(f"[VLMTrainer] result 생성 실패 (무시): {e}")

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        accum, n = {}, 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True
        )
        for step, batch in enumerate(pbar):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                loss_dict = self.raw_model.compute_loss(
                    batch, self.device,
                    semantic_weight=self.semantic_weight,
                    prior_weight=self.loss_cfg.get("prior_weight", 1.0),
                )

            loss = loss_dict["total_loss"] / self.grad_accum
            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.train_cfg.get("grad_clip", 1.0),
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    accum[k] = accum.get(k, 0.0) + v.item()
            n += 1
            pbar.set_postfix(loss=f"{loss_dict['total_loss'].item():.4f}")

        return {k: v / n for k, v in accum.items()}

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    def _print(self, epoch, train, val=None):
        stage = "S2" if epoch >= self.stage2_epoch else "S1"
        msg = f"[{stage} Epoch {epoch:3d}]"
        for k, v in train.items():
            msg += f"  {k}={v:.4f}"
        if val:
            msg += "  |  VAL:"
            for k, v in val.items():
                if isinstance(v, float):
                    msg += f"  {k}={v:.4f}"
        print(msg)

    def _write_log(self, d: dict):
        if self._log_file is None:
            return
        self._log_file.write(json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}
        ) + "\n")
        self._log_file.flush()

    def _save(self, tag):
        path = os.path.join(self.output_dir, f"ckpt_{tag}.pt")
        epoch_val = tag if isinstance(tag, int) else None
        torch.save({
            "model": self.raw_model.state_dict_for_save(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "z_form": self.cfg["system2"]["z_form"],
            "epoch": epoch_val,
        }, path)
        print(f"[VLMTrainer] 체크포인트 저장: {path}")
