"""Configuration loading, reproducibility and device selection.

Every experiment must run from a YAML configuration in ``configs/``; no
hyperparameters are hardcoded in training or model code. ``load_config``
returns a plain dict so downstream code stays framework-agnostic.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# src/afripara/config.py -> src/afripara -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base`` (update wins)."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(name: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load ``configs/<name>.yaml`` and apply optional nested overrides.

    Args:
        name: config file stem, e.g. ``"default"``.
        overrides: nested dict applied last (highest precedence).

    Returns:
        Merged configuration as a plain dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if overrides:
        cfg = deep_merge(cfg, overrides)
    logger.info("Loaded config '%s' from %s", name, path)
    return cfg


def load_model_config(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a model configuration from ``configs/models/<name>.yaml``.

    Args:
        name: model config stem, e.g. ``"afroxlmr_base"``.
        overrides: nested dict applied last (highest precedence).
    """
    return load_config(f"models/{name}", overrides=overrides)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path from ``cfg['paths'][key]`` relative to the project root."""
    try:
        raw = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"paths.{key} missing from configuration") from exc
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no CUDA locally
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning("PyTorch not installed; seeded Python/NumPy only")
    logger.debug("Set seed=%d", seed)


def get_device(prefer_mps: bool = True) -> str:
    """Return the best available torch device as a string.

    Selection order: cuda -> mps (Apple Silicon; not available on this
    project's Intel Mac) -> cpu. Returns ``"cpu"`` if torch is missing.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None:
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - defensive
            pass
    return "cpu"
