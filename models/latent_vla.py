"""
LatentVLA — System 1 + System 2 통합 모델
──────────────────────────────────────────
System 2 (PaliGemma): image + language → semantic latent f̃
System 1 (StochFlowPrior): f̃ → z → actions (Two-level flow)

기존 StochFlowPrior 인터페이스를 유지하므로
기존 Trainer/OfflineEvaluator와 호환된다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.system2_vlm import System2VLM
from models.stoch_latent_flow_prior import StochLatentFlowPrior
from models.encoders import SigLIPEncoder, ContextEncoder


class LatentVLA(nn.Module):
    """
    Args:
        system2         : System2VLM 인스턴스
        action_dim      : 로봇 행동 차원 (e.g. 7)
        action_horizon  : 행동 시퀀스 길이 (e.g. 8)
        z_dim           : 잠재 변수 차원 (e.g. 128)
        context_dim     : System2VLM 출력 차원 = System1 conditioning 차원
        siglip_model    : semantic future loss용 SigLIP 모델명
        prior_weight    : prior flow loss 가중치
        flow_hidden     : VelocityMLP hidden 차원
        flow_depth      : VelocityMLP residual block 수
        flow_steps      : 추론 시 ODE 스텝 수
    """

    def __init__(
        self,
        system2: System2VLM,
        action_dim: int,
        action_horizon: int = 8,
        z_dim: int = 128,
        context_dim: int = 256,
        siglip_model: str = "google/siglip-base-patch16-224",
        prior_weight: float = 1.0,
        flow_hidden: int = 512,
        flow_depth: int = 4,
        flow_steps: int = 10,
        distill_alpha: float = 0.0,
        oracle_ckpt_path: str = None,
        action_detach: bool = True,
    ):
        super().__init__()

        self.system2 = system2
        self.context_dim = context_dim

        # System 1: 기존 StochFlowPrior 그대로 재사용
        # future_feat_dim = SigLIP embed_dim (semantic loss용)
        self._siglip = SigLIPEncoder(siglip_model, freeze=True)
        future_feat_dim = self._siglip.embed_dim  # 768

        self.system1 = StochLatentFlowPrior(
            context_dim=context_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            z_dim=z_dim,
            future_feat_dim=future_feat_dim,
            prior_weight=prior_weight,
            flow_hidden=flow_hidden,
            flow_depth=flow_depth,
            flow_steps=flow_steps,
            action_detach=action_detach,
        )

        # OfflineEvaluator 호환용 (기존 코드가 policy.posterior_enc, policy.semantic_head 참조)
        self.posterior_enc = self.system1.posterior_enc
        self.semantic_head = self.system1.semantic_head if self.system1.use_future else None
        self.use_future = self.system1.use_future

        # ── M6 z-Distillation: oracle (frozen M2) ─────────────────────────
        self.distill_alpha = distill_alpha
        self._oracle_enc = None       # frozen ContextEncoder (M2's)
        self._oracle_posterior = None  # frozen PosteriorEncoder (M2's)
        if distill_alpha > 0.0 and oracle_ckpt_path is not None:
            self._load_oracle(oracle_ckpt_path, action_dim, action_horizon, z_dim)

    # ── Oracle 로딩 (M6 z-Distillation) ──────────────────────────────────────

    def _load_oracle(self, ckpt_path: str, action_dim: int, action_horizon: int, z_dim: int):
        """M2(DetLatent) 체크포인트에서 ContextEncoder + PosteriorEncoder를 로드해 freeze."""
        from models.det_latent import DetLatent

        ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_sd = ckpt["context_encoder"]

        # 체크포인트 weight shape으로 dim 자동 추론
        proprio_dim  = enc_sd["proprio_encoder.net.0.weight"].shape[1]
        oracle_ctx_dim = enc_sd["fusion.0.weight"].shape[0]
        future_feat_dim = self._siglip.embed_dim  # 768

        oracle_enc = ContextEncoder(proprio_dim=proprio_dim, context_dim=oracle_ctx_dim)
        oracle_enc.load_state_dict(enc_sd)
        oracle_enc.eval()
        for p in oracle_enc.parameters():
            p.requires_grad = False
        self._oracle_enc = oracle_enc

        oracle_det = DetLatent(
            context_dim=oracle_ctx_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            z_dim=z_dim,
            future_feat_dim=future_feat_dim,
        )
        oracle_det.load_state_dict(ckpt["policy"])
        oracle_det.eval()
        for p in oracle_det.parameters():
            p.requires_grad = False
        self._oracle_posterior = oracle_det.posterior  # PosteriorEncoder만 보관

        print(f"[LatentVLA] Oracle loaded from {ckpt_path} "
              f"(proprio_dim={proprio_dim}, ctx_dim={oracle_ctx_dim})")

    # ── 학습 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _z_infonce_loss(z: torch.Tensor, task_ids: torch.Tensor,
                        temperature: float = 0.07) -> torch.Tensor:
        """
        z_star에 대한 InfoNCE loss.
        같은 task_id 샘플을 positive pair로, 다른 task_id를 negative로 취급.
        batch 내에 positive pair가 없는 샘플은 loss 계산에서 제외.
        """
        z_norm = F.normalize(z, dim=-1)                                 # (B, D)
        sim = z_norm @ z_norm.T / temperature                           # (B, B)
        pos_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)       # (B, B)
        pos_mask.fill_diagonal_(False)

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=z.device)

        sim_sel = sim[has_pos]          # (M, B)
        pos_sel = pos_mask[has_pos]     # (M, B)
        loss = -torch.log(
            (sim_sel.exp() * pos_sel).sum(1) / sim_sel.exp().sum(1)
        ).mean()
        return loss

    @staticmethod
    def _hard_task_contrastive_loss(
        z: torch.Tensor,
        task_ids: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        z_norm = F.normalize(z, dim=-1)
        sim = z_norm @ z_norm.T
        pos_mask = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=z.device)

        pos_sim = sim.masked_fill(~pos_mask, 2.0).min(dim=1).values
        neg_sim = sim.masked_fill(~neg_mask, -2.0).max(dim=1).values
        return F.relu(margin + neg_sim[valid] - pos_sim[valid]).mean()

    @staticmethod
    def _negative_indices(batch: dict, actions: torch.Tensor, kind: str) -> torch.Tensor:
        B = actions.shape[0]
        device = actions.device
        if B < 2:
            return torch.arange(B, device=device)

        if kind == "shuffle":
            perm = torch.randperm(B, device=device)
            if torch.equal(perm, torch.arange(B, device=device)):
                perm = torch.roll(perm, shifts=1)
            return perm

        if kind == "task_negative":
            task_ids = batch.get("task_id")
            if task_ids is None:
                return LatentVLA._negative_indices(batch, actions, "shuffle")
            task_ids = task_ids.to(device)
            perm = LatentVLA._negative_indices(batch, actions, "shuffle")
            for i in range(B):
                candidates = torch.where(task_ids != task_ids[i])[0]
                if candidates.numel() > 0:
                    j = torch.randint(candidates.numel(), (1,), device=device).item()
                    perm[i] = candidates[j]
            return perm

        if kind == "motion_negative":
            flat = actions.reshape(B, -1).float()
            dist = torch.cdist(flat, flat)
            dist.fill_diagonal_(-1.0)
            return dist.argmax(dim=1)

        raise ValueError(f"Unknown negative kind: {kind}")

    def _counterfactual_binding_loss(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        margin: float,
    ):
        B = actions.shape[0]
        target = actions.reshape(B, -1)
        pos_cond = torch.cat([context, z_pos], dim=-1)
        neg_cond = torch.cat([context, z_neg], dim=-1)
        x0 = torch.randn_like(target)
        t_min, t_max = 1e-4, 1.0 - 1e-4
        t = torch.rand(B, device=target.device) * (t_max - t_min) + t_min
        xt = (1 - t[:, None]) * x0 + t[:, None] * target
        target_velocity = target - x0
        loss_pos = F.mse_loss(
            self.system1.action_flow(xt, t, pos_cond), target_velocity
        )
        loss_neg = F.mse_loss(
            self.system1.action_flow(xt, t, neg_cond), target_velocity
        )
        return F.relu(margin + loss_pos - loss_neg), loss_pos, loss_neg

    def compute_loss(
        self,
        batch: dict,
        device: torch.device,
        semantic_weight: float = 0.1,
        prior_weight: float = None,
        infonce_weight: float = 0.0,
        infonce_temperature: float = 0.07,
        prior_action_mix_prob: float = 0.0,
        prior_action_weight: float = 1.0,
        z_spread_weight: float = 0.0,
        z_spread_min_var: float = 0.1,
        supervised_contrastive_weight: float = 0.0,
        hard_contrastive_weight: float = 0.0,
        hard_contrastive_margin: float = 0.2,
        content_cf_weight: float = 0.0,
        content_cf_margin: float = 0.1,
        content_cf_negatives=None,
    ) -> dict:
        """
        batch 키:
          'raw_image'   : (B, H, W, 3) uint8  또는 (B, 3, H, W) float
          'image'       : (B, 3, 224, 224) float  (SigLIP 전처리)
          'proprio'     : (B, proprio_dim)
          'language'    : List[str]
          'actions'     : (B, horizon, action_dim)
          'future_image': (B, 3, 224, 224) float  (SigLIP 전처리, semantic loss용)
        """
        actions = batch["actions"].to(device)
        proprio = batch["proprio"].to(device)

        # ── System 2: VLM → f̃ ─────────────────────────────────────────────
        raw_image = batch.get("raw_image", batch["image"])  # raw 없으면 SigLIP image 사용
        pixel_values, input_ids, attn_mask = self.system2.prepare_inputs(
            raw_image, batch["language"], device
        )
        f_tilde = self.system2(pixel_values, input_ids, attn_mask, proprio)  # (B, context_dim)

        # ── Semantic future feat (frozen SigLIP) ───────────────────────────
        # distill_alpha > 0이면 oracle z 계산에도 future_feat 필요 → 항상 계산
        need_future = (semantic_weight > 0 and self.system1.use_future) or \
                      (self.distill_alpha > 0 and self._oracle_enc is not None)
        future_feat = None
        if need_future and "future_image" in batch:
            with torch.no_grad():
                future_feat = self._siglip.encode_image_only(
                    batch["future_image"].to(device)
                )

        # ── System 1: StochFlowPrior loss ─────────────────────────────────
        loss_dict = self.system1.compute_loss(
            context=f_tilde,
            actions=actions,
            future_feat=future_feat,
            planner_context=f_tilde,
            semantic_weight=semantic_weight,
            prior_weight=prior_weight,
            prior_action_mix_prob=prior_action_mix_prob,
            prior_action_weight=prior_action_weight,
        )
        z_mu   = loss_dict.pop("_z_mu")    # posterior mean, for distillation
        z_star = loss_dict.pop("_z_star")  # sampled posterior z, for InfoNCE

        # ── z-InfoNCE loss ─────────────────────────────────────────────────
        if infonce_weight > 0.0 and "task_id" in batch:
            task_ids = batch["task_id"].to(device)
            loss_infonce = self._z_infonce_loss(z_star, task_ids, infonce_temperature)
            loss_dict["infonce_loss"] = loss_infonce
            loss_dict["total_loss"] = loss_dict["total_loss"] + infonce_weight * loss_infonce

        loss_z_spread = torch.tensor(0.0, device=device)
        if z_spread_weight > 0.0:
            loss_z_spread = F.relu(z_spread_min_var - z_mu.var(dim=0).mean())
            loss_dict["z_spread_loss"] = loss_z_spread
            loss_dict["total_loss"] = loss_dict["total_loss"] + z_spread_weight * loss_z_spread

        if "task_id" in batch:
            task_ids = batch["task_id"].to(device)
            if supervised_contrastive_weight > 0.0:
                loss_supcon = self._z_infonce_loss(z_mu, task_ids, infonce_temperature)
                loss_dict["supervised_contrastive_loss"] = loss_supcon
                loss_dict["total_loss"] = (
                    loss_dict["total_loss"] + supervised_contrastive_weight * loss_supcon
                )
            if hard_contrastive_weight > 0.0:
                loss_hard = self._hard_task_contrastive_loss(
                    z_mu, task_ids, hard_contrastive_margin
                )
                loss_dict["hard_contrastive_loss"] = loss_hard
                loss_dict["total_loss"] = (
                    loss_dict["total_loss"] + hard_contrastive_weight * loss_hard
                )

        if content_cf_weight > 0.0:
            kinds = content_cf_negatives or ["shuffle", "task_negative", "motion_negative"]
            cf_losses = []
            for kind in kinds:
                neg_idx = self._negative_indices(batch, actions, kind)
                z_neg = z_star[neg_idx]
                loss_cf, loss_pos, loss_neg = self._counterfactual_binding_loss(
                    f_tilde, actions, z_star, z_neg, content_cf_margin
                )
                cf_losses.append(loss_cf)
                loss_dict[f"cf_{kind}_loss"] = loss_cf
                loss_dict[f"cf_{kind}_pos_action_loss"] = loss_pos
                loss_dict[f"cf_{kind}_neg_action_loss"] = loss_neg
                loss_dict[f"cf_{kind}_gap"] = loss_neg.detach() - loss_pos.detach()
            loss_cf_total = torch.stack(cf_losses).mean()
            loss_dict["content_cf_loss"] = loss_cf_total
            loss_dict["total_loss"] = loss_dict["total_loss"] + content_cf_weight * loss_cf_total

        # ── M6 z-Distillation loss ─────────────────────────────────────────
        if self.distill_alpha > 0 and self._oracle_enc is not None and future_feat is not None:
            with torch.no_grad():
                oracle_ids, oracle_mask = self._oracle_enc.tokenize(
                    batch["language"], device
                )
                C_t_oracle = self._oracle_enc(
                    batch["image"].to(device), proprio, oracle_ids, oracle_mask
                )
                z_oracle = self._oracle_posterior(C_t_oracle, actions, future_feat)

            loss_distill = F.mse_loss(z_mu, z_oracle)
            loss_dict["distill_loss"] = loss_distill
            loss_dict["total_loss"] = loss_dict["total_loss"] + self.distill_alpha * loss_distill

        return loss_dict

    # ── 추론 ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        device: torch.device,
        n_samples: int = 1,
        std_scale: float = 1.0,
        use_posterior: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns: (B, horizon, action_dim) 또는 (B, n_samples, horizon, action_dim)
        """
        proprio = batch["proprio"].to(device)
        raw_image = batch.get("raw_image", batch["image"])
        pixel_values, input_ids, attn_mask = self.system2.prepare_inputs(
            raw_image, batch["language"], device
        )
        f_tilde = self.system2(pixel_values, input_ids, attn_mask, proprio)

        if use_posterior and "actions" in batch:
            future_feat = None
            if self.system1.use_future and "future_image" in batch:
                future_feat = self._siglip.encode_image_only(
                    batch["future_image"].to(device)
                )
            return self.system1.predict(
                context=f_tilde,
                planner_context=f_tilde,
                use_posterior=True,
                actions_for_posterior=batch["actions"].to(device),
                future_feat_for_posterior=future_feat,
                n_samples=n_samples,
                std_scale=std_scale,
            )

        return self.system1.predict(
            context=f_tilde,
            planner_context=f_tilde,
            n_samples=n_samples,
            std_scale=std_scale,
        )

    # ── OfflineEvaluator 호환 래퍼 ────────────────────────────────────────────
    # OfflineEvaluator는 policy.predict(context, ...) 형태를 기대하므로
    # context를 이미 인코딩된 f̃로 받는 래퍼 제공

    def predict_from_context(
        self,
        context: torch.Tensor,       # (B, context_dim) — 이미 인코딩된 f̃
        use_posterior: bool = False,
        actions_for_posterior: torch.Tensor = None,
        future_feat_for_posterior: torch.Tensor = None,
        n_samples: int = 1,
        std_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """OfflineEvaluator에서 호출하는 인터페이스."""
        return self.system1.predict(
            context=context,
            planner_context=context,
            use_posterior=use_posterior,
            actions_for_posterior=actions_for_posterior,
            future_feat_for_posterior=future_feat_for_posterior,
            n_samples=n_samples,
            std_scale=std_scale,
        )

    # ── 파라미터 그룹 (optimizer용) ───────────────────────────────────────────

    def vlm_lora_parameters(self):
        return self.system2.vlm_lora_parameters()

    def non_vlm_parameters(self):
        return (
            list(self.system2.non_vlm_parameters())
            + list(self.system1.parameters())
        )

    # ── 체크포인트 ────────────────────────────────────────────────────────────

    def state_dict_for_save(self):
        """LoRA 가중치만 저장 (전체 PaliGemma 3B 저장 방지)."""
        return {
            "system2_lora": self.system2.lora_state_dict(),
            "system2_proj": self.system2.proj.state_dict(),
            "system2_proprio_encoder": self.system2.proprio_encoder.state_dict(),
            "system1": self.system1.state_dict(),
        }

    def load_state_dict_from_save(self, d: dict):
        if d.get("system2_proj"):
            self.system2.proj.load_state_dict(d["system2_proj"])
        if d.get("system2_proprio_encoder"):
            try:
                self.system2.proprio_encoder.load_state_dict(d["system2_proprio_encoder"])
            except RuntimeError as e:
                print(f"[LatentVLA] proprio_encoder 로드 스킵 (shape 불일치): {e}")
        if d.get("system1"):
            self.system1.load_state_dict(d["system1"])
        if d.get("system2_lora") and self.system2._lora_enabled:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.system2.vlm, d["system2_lora"])
