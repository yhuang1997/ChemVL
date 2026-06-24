"""Load and apply per-dataset (and optional per-modality) training hparam overrides from YAML."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_MODEL_KEYS = ("dropout", "reduction_ratio")


def modality_from_group_id(group_id: str) -> Optional[str]:
    if group_id.startswith("image_"):
        return "image"
    if group_id.startswith("graph_"):
        return "graph"
    return None


def _merge_graph_encoder(model_cfg: Dict[str, Any], graph_encoder: Dict[str, Any]) -> None:
    ge_cfg = model_cfg.setdefault("graph_encoder", {})
    if not isinstance(ge_cfg, dict):
        model_cfg["graph_encoder"] = dict(graph_encoder)
        return
    ge_cfg.update(graph_encoder)


def _merge_section(cfg: Dict[str, Any], section: Dict[str, Any]) -> None:
    if not isinstance(section, dict):
        return
    training = section.get("training")
    if isinstance(training, dict):
        cfg.setdefault("training", {}).update(training)
    model = section.get("model")
    if isinstance(model, dict):
        model_cfg = cfg.setdefault("model", {})
        for key in _MODEL_KEYS:
            if key in model:
                model_cfg[key] = model[key]
        graph_encoder = model.get("graph_encoder")
        if isinstance(graph_encoder, dict):
            _merge_graph_encoder(model_cfg, graph_encoder)
    data_augmentation = section.get("data_augmentation")
    if isinstance(data_augmentation, dict):
        cfg.setdefault("data_augmentation", {}).update(data_augmentation)


def load_training_hparams_yaml(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"invalid yaml in {path}: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"training hparams yaml root must be a mapping: {path}")
    datasets = raw.get("datasets")
    if datasets is not None and not isinstance(datasets, dict):
        raise ValueError(f"'datasets' must be a mapping in {path}")
    defaults = raw.get("defaults")
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError(f"'defaults' must be a mapping in {path}")
    return raw


def apply_training_hparams_yaml(
    cfg: Dict[str, Any],
    yaml_root: Dict[str, Any],
    *,
    dataset: str,
    group_id: str,
) -> None:
    if not yaml_root:
        return

    defaults = yaml_root.get("defaults")
    if isinstance(defaults, dict):
        _merge_section(cfg, defaults)
        modality = modality_from_group_id(group_id)
        if modality is not None:
            mod_defaults = defaults.get(modality)
            if isinstance(mod_defaults, dict):
                _merge_section(cfg, mod_defaults)

    datasets = yaml_root.get("datasets") or {}
    if not isinstance(datasets, dict):
        return

    ds_block = datasets.get(dataset)
    if ds_block is None:
        return
    if not isinstance(ds_block, dict):
        raise ValueError(f"datasets.{dataset} must be a mapping")

    _merge_section(cfg, ds_block)

    modality = modality_from_group_id(group_id)
    if modality is not None:
        mod_block = ds_block.get(modality)
        if isinstance(mod_block, dict):
            _merge_section(cfg, mod_block)


def warn_unknown_datasets(yaml_root: Dict[str, Any], known_datasets: set[str]) -> None:
    datasets = yaml_root.get("datasets") or {}
    if not isinstance(datasets, dict):
        return
    for key in datasets:
        if key not in known_datasets:
            warnings.warn(
                f"training hparams yaml: unknown dataset {key!r} (not in --datasets); ignored at runtime",
                stacklevel=2,
            )
