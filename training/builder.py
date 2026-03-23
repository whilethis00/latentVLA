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


def build_dataloaders(train_ds, val_ds, cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
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
