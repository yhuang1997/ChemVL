# SPDX-License-Identifier: MIT
"""ImageMol finetune hyperparameters (official defaults + issue #13 overrides)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.external.molmcl.molmcl_external_config import chemvl_repo_root, deep_merge

# MoleculeNet task overlay key -> params_imagemol.yaml datasets key
_HPARAM_DATASET_ALIASES = {
    "lipophilicity": "lipo",
}


def _hparams_dataset_key(dataset_name: str) -> str:
    key = str(dataset_name or "").lower()
    return _HPARAM_DATASET_ALIASES.get(key, key)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping yaml at {path}, got {type(data)}")
    return data


def load_merged_imagemol_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge optional ``model.imagemol.external_hparams_yaml`` + ``model.imagemol.imagemol_overrides``.

    Does **not** read ``training.lr`` / ``training.batch_size`` / ``training.epochs`` from ChemVL JSON —
    those live in yaml (aligned with MolMCL / MolCLR ``init_lr`` handling).

    Falls back to sensible ImageMol ``finetune.py`` defaults when yaml absent.
    """
    im = (cfg.get("model") or {}).get("imagemol") or {}
    repo = chemvl_repo_root()
    merged: Dict[str, Any] = {
        "optimizer": "SGD",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 100,
        "resume_key": "state_dict",
    }

    hparams = im.get("external_hparams_yaml") or "configs/external/imagemol/params_imagemol.yaml"
    p = Path(hparams) if Path(hparams).is_absolute() else (repo / hparams)
    ds_name = _hparams_dataset_key(str((cfg.get("dataset") or {}).get("dataset", "")))
    if p.is_file():
        raw = _load_yaml(p)
        defaults = raw.get("defaults") or {}
        merged = deep_merge(merged, defaults)
        per_ds = (raw.get("datasets") or {}).get(ds_name) or {}
        if per_ds:
            merged = deep_merge(merged, per_ds)

    overrides = im.get("imagemol_overrides")
    if overrides:
        merged = deep_merge(merged, overrides)

    merged.pop("datasets", None)
    merged.pop("defaults", None)

    merged["task_type"] = str((cfg.get("dataset") or {}).get("task_type", "classification")).lower()
    merged["num_tasks"] = int((cfg.get("dataset") or {}).get("num_tasks", 1))
    merged["resume"] = im.get("resume")
    merged["resume_key"] = im.get("resume_key") or merged.get("resume_key") or "state_dict"
    return merged


def preview_imagemol_epochs_batch(cfg: Dict[str, Any]) -> tuple[int | None, int | None]:
    try:
        m = load_merged_imagemol_config(cfg)
        return int(m.get("epochs", 0) or 0) or None, int(m.get("batch_size", 0) or 0) or None
    except (OSError, ValueError, TypeError, KeyError):
        return None, None
