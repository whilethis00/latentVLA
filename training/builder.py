"""
Factory functions for datasets, dataloaders, and models.
"""

import torch
from torch.utils.data import DataLoader, random_split

from data import RobomimicDataset, LiberoDataset
from models import ContextEncoder, StateContextEncoder, FlatFlow, DetLatent, StochLatentVAE, StochLatentFlowPrior


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    kwargs = dict(
        action_horizon=data_cfg["action_horizon"],
        image_size=data_cfg["image_size"],
    )

    if data_cfg["dataset_type"] == "robomimic":
        train_ds = RobomimicDataset(data_cfg["dataset_path"], split="train", **kwargs)
        val_ds = RobomimicDataset(data_cfg["dataset_path"], split="valid", **kwargs)
        # Propagate normalization stats
        if hasattr(train_ds, "action_mean"):
            val_ds.action_mean = train_ds.action_mean
            val_ds.action_std = train_ds.action_std
        if hasattr(train_ds, "obs_mean"):
            val_ds.obs_mean = train_ds.obs_mean
            val_ds.obs_std = train_ds.obs_std
    elif data_cfg["dataset_type"] == "libero":
        train_ds = LiberoDataset(data_cfg["dataset_path"], split="train", **kwargs)
        val_ds = LiberoDataset(data_cfg["dataset_path"], split="valid", **kwargs)
        if hasattr(train_ds, "action_mean"):
            val_ds.action_mean = train_ds.action_mean
            val_ds.action_std = train_ds.action_std
    else:
        raise ValueError(f"Unknown dataset_type: {data_cfg['dataset_type']}")

    return train_ds, val_ds


def build_dataloaders(train_ds, val_ds, cfg: dict, train_sampler=None):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: dict, action_dim: int, proprio_dim: int, train_ds=None):
    enc_cfg = cfg["encoder"]
    lat_cfg = cfg["latent"]
    mdl_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # Choose encoder based on dataset mode
    is_low_dim = (
        data_cfg.get("encoder_type", "auto") == "state"
        or (train_ds is not None and hasattr(train_ds, "mode") and train_ds.mode == "low_dim")
    )

    if is_low_dim:
        future_state_dim = train_ds.future_state_dim if train_ds else 10
        context_encoder = StateContextEncoder(
            state_dim=proprio_dim,
            context_dim=enc_cfg["context_dim"],
            future_state_dim=future_state_dim,
            hidden=enc_cfg.get("state_hidden", 256),
        )
        # Planner encoder (may differ from context_encoder based on planner_input)
        planner_input = mdl_cfg.get("planner_input", "full")
        object_state_dim = mdl_cfg.get("object_state_dim", 10)
        if planner_input == "full":
            planner_encoder = context_encoder  # shared, no extra params
        elif planner_input == "object_only":
            planner_encoder = StateContextEncoder(
                state_dim=object_state_dim,
                context_dim=enc_cfg["context_dim"],
                future_state_dim=future_state_dim,
                hidden=enc_cfg.get("state_hidden", 256),
            )
        elif planner_input == "proprio_only":
            planner_encoder = StateContextEncoder(
                state_dim=proprio_dim - object_state_dim,
                context_dim=enc_cfg["context_dim"],
                future_state_dim=future_state_dim,
                hidden=enc_cfg.get("state_hidden", 256),
            )
        else:
            raise ValueError(f"Unknown planner_input: {planner_input}")
    else:
        context_encoder = ContextEncoder(
            proprio_dim=proprio_dim,
            context_dim=enc_cfg["context_dim"],
            siglip_model=enc_cfg["siglip_model"],
            freeze_encoder=enc_cfg["freeze_image_encoder"],
            proprio_hidden=enc_cfg["proprio_hidden"],
        )
        planner_encoder = context_encoder  # image mode: always shared

    model_type = mdl_cfg["type"]
    shared_kwargs = dict(
        context_dim=enc_cfg["context_dim"],
        action_dim=action_dim,
        action_horizon=data_cfg["action_horizon"],
        flow_hidden=mdl_cfg["flow_hidden"],
        flow_depth=mdl_cfg["flow_depth"],
        flow_steps=mdl_cfg["flow_steps"],
    )

    # future_feat_dim=0 when semantic loss is disabled, to keep posterior_enc shape consistent
    sem_weight = cfg.get("loss", {}).get("semantic_future_weight", 0.1)
    future_dim = context_encoder.image_lang_encoder.embed_dim if sem_weight > 0 else 0

    if model_type == "flat_flow":
        policy = FlatFlow(**shared_kwargs)

    elif model_type == "det_latent":
        policy = DetLatent(
            z_dim=lat_cfg["z_dim"],
            future_feat_dim=future_dim,
            **shared_kwargs,
        )

    elif model_type == "stoch_vae":
        policy = StochLatentVAE(
            z_dim=lat_cfg["z_dim"],
            future_feat_dim=future_dim,
            kl_beta=cfg["loss"]["kl_beta"],
            **shared_kwargs,
        )

    elif model_type == "stoch_flow_prior":
        policy = StochLatentFlowPrior(
            z_dim=lat_cfg["z_dim"],
            future_feat_dim=future_dim,
            prior_weight=cfg["loss"]["prior_weight"],
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return context_encoder, planner_encoder, policy


# ── VLM 모델 빌드 (LatentVLA) ──────────────────────────────────────────────────

def build_vlm_datasets(cfg: dict):
    """LatentVLA용 데이터셋 빌드 (기존 build_datasets와 동일하나 vlm_mode=True)."""
    data_cfg = cfg["data"]
    kwargs = dict(
        action_horizon=data_cfg["action_horizon"],
        image_size=data_cfg["image_size"],
        include_raw_image=True,   # PaliGemma용 uint8 이미지 추가
    )
    if data_cfg["dataset_type"] == "libero":
        train_ds = LiberoDataset(data_cfg["dataset_path"], split="train", **kwargs)
        val_ds   = LiberoDataset(data_cfg["dataset_path"], split="valid", **kwargs)
        if hasattr(train_ds, "action_mean"):
            val_ds.action_mean = train_ds.action_mean
            val_ds.action_std  = train_ds.action_std
    elif data_cfg["dataset_type"] == "robomimic":
        train_ds = RobomimicDataset(data_cfg["dataset_path"], split="train", **kwargs)
        val_ds   = RobomimicDataset(data_cfg["dataset_path"], split="valid", **kwargs)
        if hasattr(train_ds, "action_mean"):
            val_ds.action_mean = train_ds.action_mean
            val_ds.action_std  = train_ds.action_std
    else:
        raise ValueError(f"Unknown dataset_type: {data_cfg['dataset_type']}")
    return train_ds, val_ds


def build_dataloaders_vlm(train_ds, val_ds, cfg: dict, rank: int = 0, world_size: int = 1):
    from torch.utils.data import DistributedSampler
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    is_ddp = world_size > 1

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if is_ddp else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def build_vlm_model(cfg: dict, action_dim: int, proprio_dim: int = None):
    """LatentVLA (System2VLM + StochFlowPrior) 빌드.

    proprio_dim: 데이터셋 기준 값 (지정 시 cfg["system2"]["proprio_dim"] 무시)
    """
    from models.system2_vlm import System2VLM
    from models.latent_vla import LatentVLA

    s2_cfg  = cfg["system2"]
    enc_cfg = cfg["encoder"]
    lat_cfg = cfg["latent"]
    mdl_cfg = cfg["model"]

    _proprio_dim = proprio_dim if proprio_dim is not None else s2_cfg.get("proprio_dim", 9)

    system2 = System2VLM(
        model_name=s2_cfg["model_name"],
        context_dim=enc_cfg["context_dim"],
        z_form=s2_cfg["z_form"],
        pool_k=s2_cfg.get("pool_k", 8),
        freeze_backbone=True,           # 항상 Stage 1으로 시작
        lora_rank=s2_cfg.get("lora_rank", 16),
        lora_alpha=s2_cfg.get("lora_alpha", 32),
        lora_target_modules=s2_cfg.get("lora_target_modules", None),
        proprio_dim=_proprio_dim,
        proprio_hidden=s2_cfg.get("proprio_hidden", 128),
    )

    model = LatentVLA(
        system2=system2,
        action_dim=action_dim,
        action_horizon=cfg["data"]["action_horizon"],
        z_dim=lat_cfg["z_dim"],
        context_dim=enc_cfg["context_dim"],
        siglip_model=enc_cfg.get("siglip_model", "google/siglip-base-patch16-224"),
        prior_weight=cfg.get("loss", {}).get("prior_weight", 1.0),
        flow_hidden=mdl_cfg["flow_hidden"],
        flow_depth=mdl_cfg["flow_depth"],
        flow_steps=mdl_cfg["flow_steps"],
    )
    return model
