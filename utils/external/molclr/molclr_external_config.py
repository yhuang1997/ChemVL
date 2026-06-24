# SPDX-License-Identifier: MIT
"""Load MolCLR ``config_finetune.yaml`` + ChemVL ``model.molclr`` overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, MutableMapping

import yaml

from utils.external.molmcl.molmcl_external_config import chemvl_repo_root, deep_merge


def molclr_root_from_cfg(cfg: Dict[str, Any]) -> str:
    mc = (cfg.get("model") or {}).get("molclr") or {}
    repo = chemvl_repo_root()
    return os.path.abspath(mc.get("molclr_root") or os.environ.get("MOLCLR_ROOT") or str(repo / "external" / "MolCLR"))


def _resolve_config_path(repo: Path, molclr_root: str, spec: str) -> str:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("model.molclr.external_config is empty.")
    if os.path.isabs(spec):
        p = Path(spec)
    else:
        cand_repo = (repo / spec).resolve()
        cand_ext = (Path(molclr_root) / spec).resolve()
        if cand_repo.is_file():
            p = cand_repo
        elif cand_ext.is_file():
            p = cand_ext
        else:
            raise FileNotFoundError(
                f"MolCLR external_config not found: tried {cand_repo} and {cand_ext} (spec={spec!r})."
            )
    if not p.is_file():
        raise FileNotFoundError(f"MolCLR config is not a file: {p}")
    return str(p)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping yaml at {path}, got {type(data)}")
    return data


def _resolve_resume(resume: Any, repo: Path) -> str | None:
    if not resume:
        return None
    path = os.path.expanduser(str(resume))
    if os.path.isfile(path):
        return os.path.abspath(path)
    cand = (repo / path).resolve()
    if cand.is_file():
        return str(cand)
    env_root = os.environ.get("CHEMVL_DATA_ROOT", "")
    if env_root:
        cand2 = Path(env_root) / path.lstrip("/")
        if cand2.is_file():
            return str(cand2.resolve())
    return path


def load_merged_molclr_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge MolCLR yaml + ``model.molclr.molclr_overrides`` + remaining ``model.molclr`` keys.

    Does **not** read ``training.epochs`` / ``training.batch_size`` from ChemVL JSON — those live in yaml
    (aligned with MolMCL).

    ChemVL shell fields (not in yaml):
    - ``task_type`` from ``cfg['dataset']['task_type']``
    - ``num_tasks`` from ``cfg['dataset']['num_tasks']``
    """
    mc = (cfg.get("model") or {}).get("molclr") or {}
    repo = chemvl_repo_root()
    molclr_root = molclr_root_from_cfg(cfg)

    ec = mc.get("external_config")
    if not ec:
        ec = "external/MolCLR/config_finetune.yaml"
    merged = _load_yaml(_resolve_config_path(repo, molclr_root, str(ec)))

    ds_name = str((cfg.get("dataset") or {}).get("dataset", "")).lower()
    model_type = str(mc.get("model_type") or merged.get("model_type") or "gin").lower()
    hparams = mc.get("external_hparams_yaml") or "scripts/external/molclr_under_chemvl/params_molclr.yaml"
    hp_path = Path(hparams) if Path(hparams).is_absolute() else (repo / hparams)
    if hp_path.is_file():
        raw_hp = _load_yaml(str(hp_path))
        defaults = raw_hp.get("defaults") or {}
        if defaults:
            merged = deep_merge(merged, defaults)
        per_ds = (raw_hp.get("datasets") or {}).get(ds_name) or {}
        if per_ds:
            if model_type in per_ds and isinstance(per_ds[model_type], dict):
                merged = deep_merge(merged, per_ds[model_type])
            elif not any(k in per_ds for k in ("gin", "gcn")):
                merged = deep_merge(merged, per_ds)

    overrides = mc.get("molclr_overrides")
    if overrides:
        merged = deep_merge(merged, overrides)

    reserved = {"molclr_root", "external_config", "external_hparams_yaml", "molclr_overrides", "resume", "model_type", "variant"}
    for k, v in mc.items():
        if k in reserved:
            continue
        if k not in merged:
            merged[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(merged[k], dict):
            merged[k] = deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = copy.deepcopy(v)

    merged["model_type"] = str(mc.get("model_type") or merged.get("model_type") or "gin").lower()
    merged["resume"] = _resolve_resume(mc.get("resume"), repo)
    merged["task_type"] = str((cfg.get("dataset") or {}).get("task_type", "classification")).lower()
    merged["num_tasks"] = int((cfg.get("dataset") or {}).get("num_tasks", 1))
    merged["dataset_name"] = str((cfg.get("dataset") or {}).get("dataset", "")).lower()

    return merged


def preview_molclr_epochs_batch(cfg: Dict[str, Any]) -> tuple[int | None, int | None]:
    try:
        m = load_merged_molclr_config(cfg)
        return int(m.get("epochs", 0) or 0) or None, int(m.get("batch_size", 0) or 0) or None
    except (OSError, ValueError, TypeError, KeyError):
        return None, None
