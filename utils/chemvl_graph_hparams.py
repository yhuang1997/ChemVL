"""Merge ``params_chemvl_graph.yaml`` into ChemVL graph finetune configs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.external.molmcl.molmcl_external_config import deep_merge


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping yaml at {path}, got {type(data)}")
    return data


def resolve_graph_hparams_yaml(cfg: Dict[str, Any]) -> str | None:
    model = cfg.get("model") or {}
    spec = model.get("graph_hparams_yaml")
    if not spec:
        return None
    spec = str(spec).strip()
    if os.path.isabs(spec):
        return spec if os.path.isfile(spec) else None
    cand = (_repo_root() / spec).resolve()
    return str(cand) if cand.is_file() else None


def apply_graph_hparams_yaml_file(cfg: Dict[str, Any], yaml_path: str | Path) -> Dict[str, Any] | None:
    """
    Merge dataset-specific graph hparams from an explicit yaml path (in-place).

    Used for ``params_chemvl_graph.yaml``, MolCLR reference yaml, etc.
    """
    path = Path(yaml_path)
    if not path.is_file():
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        if not path.is_file():
            return None

    params = _load_yaml(str(path))
    dataset_key = str((cfg.get("dataset") or {}).get("dataset", "")).lower()
    # MoleculeNet folder name vs yaml key (e.g. lipophilicity -> lipo).
    _YAML_KEY_ALIASES = {"lipophilicity": "lipo"}
    yaml_key = _YAML_KEY_ALIASES.get(dataset_key, dataset_key)
    defaults = params.get("defaults") or {}
    ds_block = (params.get("datasets") or {}).get(yaml_key) or {}

    effective: Dict[str, Any] = {}
    for src in (defaults, ds_block):
        for k, v in src.items():
            if k in ("note", "graph_encoder", "regression_scheduler", "graph_training_recipe"):
                continue
            effective[k] = v

    recipe = ds_block.get("graph_training_recipe")
    if recipe is not None:
        cfg.setdefault("model", {})["graph_training_recipe"] = str(recipe).strip().lower()

    training = cfg.setdefault("training", {})
    for k in ("batch_size", "optimizer", "lr", "weight_decay", "epochs", "encoder_lr", "use_patience"):
        if k in effective and effective[k] is not None:
            training[k] = effective[k]

    ge_defaults = defaults.get("graph_encoder") or {}
    ge_ds = ds_block.get("graph_encoder") or {}
    ge_merged = {**ge_defaults, **ge_ds}
    if ge_merged:
        model = cfg.setdefault("model", {})
        ge_cfg = model.setdefault("graph_encoder", {})
        ge_cfg.update(ge_merged)

    rs_defaults = defaults.get("regression_scheduler") or {}
    rs_ds = ds_block.get("regression_scheduler") or {}
    rs_merged = {**rs_defaults, **rs_ds}
    if rs_merged:
        rs_cfg = cfg.setdefault("regression_scheduler", {})
        rs_cfg.update(rs_merged)

    cfg.setdefault("basic", {})["_graph_hparams_yaml_resolved"] = str(path)
    return {"path": str(path), "defaults": defaults, "dataset": ds_block, "effective_training": effective}


def apply_chemvl_graph_hparams(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Merge hparams from ``model.graph_hparams_yaml`` in *cfg*."""
    path = resolve_graph_hparams_yaml(cfg)
    if not path:
        return None
    return apply_graph_hparams_yaml_file(cfg, path)


def apply_molclr_gin_reference_hparams(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Merge MolCLR GIN finetune hparams (mapped to ChemVL keys) for GRAPH-004 Phase0b."""
    return apply_graph_hparams_yaml_file(
        cfg, "configs/chemvl_baselines/params_molclr_gin_kgpt_reference.yaml"
    )
