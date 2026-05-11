"""
Shared configuration helpers for training entry points.
"""

import random
from typing import Any, List

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_override_value(value: str, current: Any = None) -> Any:
    """Parse CLI override values while preserving existing config types."""
    if isinstance(current, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def apply_overrides(cfg: dict, overrides: List[str], create_missing: bool = False) -> dict:
    """Apply key=value CLI overrides to a nested config dict."""
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        target = cfg
        for part in keys[:-1]:
            if create_missing:
                target = target.setdefault(part, {})
            else:
                target = target[part]
        leaf = keys[-1]
        target[leaf] = parse_override_value(value, target.get(leaf))
    return cfg


def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
