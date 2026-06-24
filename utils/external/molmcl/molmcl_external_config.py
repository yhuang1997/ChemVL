# SPDX-License-Identifier: MIT
"""Load MolMCL finetune yaml(s) as single source of truth; optional ``molmcl_overrides`` for diffs."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

import yaml


def chemvl_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def molmcl_root_from_cfg(cfg: Dict[str, Any]) -> str:
    mc = (cfg.get("model") or {}).get("molmcl") or {}
    repo = chemvl_repo_root()
    return os.path.abspath(
        mc.get("molmcl_root") or os.environ.get("MOLMCL_ROOT") or str(repo / "external" / "MolMCL")
    )


def deep_merge(base: MutableMapping[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in (patch or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_config_path(repo: Path, molmcl_root: str, spec: str) -> str:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("external_config path is empty.")
    if os.path.isabs(spec):
        p = Path(spec)
    else:
        cand_repo = (repo / spec).resolve()
        cand_mol = (Path(molmcl_root) / spec).resolve()
        if cand_repo.is_file():
            p = cand_repo
        elif cand_mol.is_file():
            p = cand_mol
        else:
            raise FileNotFoundError(
                f"MolMCL external_config not found: tried {cand_repo} and {cand_mol} (spec={spec!r})."
            )
    if not p.is_file():
        raise FileNotFoundError(f"MolMCL config is not a file: {p}")
    return str(p)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping yaml at {path}, got {type(data)}")
    return data


def _resolve_checkpoint_in_model(model: MutableMapping[str, Any], molmcl_root: str, repo: Path) -> None:
    ck = model.get("checkpoint")
    if not ck:
        return
    if os.path.isfile(str(ck)):
        return
    rel = str(ck).lstrip("./")
    for base in (Path(molmcl_root), repo):
        cand = (base / rel).resolve()
        if cand.is_file():
            model["checkpoint"] = str(cand)
            return
    try:
        from utils.path_utils import get_data_root

        hf_cand = get_data_root() / "checkpoints" / "external" / Path(rel).name
        if hf_cand.is_file():
            model["checkpoint"] = str(hf_cand.resolve())
            return
    except RuntimeError:
        pass


def _format_external_config_spec(spec: str, cfg: Dict[str, Any]) -> str:
    """Allow ``{dataset}`` / ``{DATASET}`` placeholders (MoleculeNet lower/upper stems)."""
    if "{dataset}" not in spec and "{DATASET}" not in spec:
        return spec
    stem = str((cfg.get("dataset") or {}).get("dataset", "")).strip()
    return spec.replace("{dataset}", stem.lower()).replace("{DATASET}", stem)


def _inject_moleculenet_data_dir(cfg: Dict[str, Any], merged: Dict[str, Any], repo: Path) -> None:
    """Point MolMCL ``dataset.data_dir`` at the directory that contains ``<name>.csv`` (ChemVL dataroot layout)."""
    from utils.external.molmcl.moleculenet_io import resolve_moleculenet_csv

    csv_path = resolve_moleculenet_csv(cfg, repo)
    merged.setdefault("dataset", {})
    merged["dataset"]["data_dir"] = os.path.dirname(os.path.abspath(csv_path))


def load_merged_molmcl_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge MolMCL yaml(s) + ``model.molmcl.molmcl_overrides`` + remaining ``model.molmcl`` keys.

    Does **not** read ``training.epochs`` / ``training.batch_size`` from ChemVL — those live in yaml.

    ChemVL shell fields applied onto merged dict:

    - ``dataset.data_name`` ← ``cfg["dataset"]["dataset"]``
    - ``dataset.num_workers`` ← ``cfg["basic"]["num_workers"]``
    - ``verbose`` ← ``cfg["basic"]["verbose"]``

    When ``dataset.benchmark == "moleculenet"``, also sets ``dataset.data_dir`` from ``dataset.dataroot``
    and the dataset stem so MolMCL's ``MoleculeDataset``-style paths resolve under ChemVL data roots.
    """
    mc = (cfg.get("model") or {}).get("molmcl") or {}
    repo = chemvl_repo_root()
    molmcl_root = mc.get("molmcl_root") or os.environ.get("MOLMCL_ROOT") or str(repo / "external" / "MolMCL")
    molmcl_root = os.path.abspath(molmcl_root)

    paths: List[str] = []
    ec = mc.get("external_config")
    yc = mc.get("yaml_config")
    if ec:
        paths.append(_resolve_config_path(repo, molmcl_root, _format_external_config_spec(str(ec), cfg)))
    elif yc:
        rel = str(yc).strip().lstrip("/")
        paths.append(str(Path(molmcl_root) / rel))
        if not os.path.isfile(paths[0]):
            raise FileNotFoundError(f"yaml_config not found: {paths[0]}")
    else:
        raise ValueError(
            'Set model.molmcl.external_config to a yaml path (e.g. '
            '"configs/external/molmcl/base_config/moleculeace/chembl_gps.yaml" '
            'or "external/MolMCL/config/moleculeace/chembl.yaml") '
            'or model.molmcl.yaml_config relative to molmcl_root (deprecated).'
        )

    extra = mc.get("external_config_extra")
    if extra:
        paths.append(_resolve_config_path(repo, molmcl_root, _format_external_config_spec(str(extra), cfg)))

    merged: Dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, _load_yaml(p))

    overrides = mc.get("molmcl_overrides")
    if overrides:
        merged = deep_merge(merged, overrides)

    reserved = {
        "molmcl_root",
        "external_config",
        "external_config_extra",
        "molmcl_overrides",
        "yaml_config",
    }
    for k, v in mc.items():
        if k in reserved:
            continue
        if k not in merged:
            merged[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(merged[k], dict):
            merged[k] = deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = copy.deepcopy(v)

    merged.setdefault("dataset", {})
    dn = cfg["dataset"]["dataset"]
    if str((cfg.get("dataset") or {}).get("benchmark", "")).lower() == "moleculenet":
        merged["dataset"]["data_name"] = str(dn).lower()
    else:
        merged["dataset"]["data_name"] = dn
    merged["dataset"]["num_workers"] = int((cfg.get("basic") or {}).get("num_workers", 0))
    merged["verbose"] = bool((cfg.get("basic") or {}).get("verbose", True))

    if str((cfg.get("dataset") or {}).get("benchmark", "")).lower() == "moleculenet":
        _inject_moleculenet_data_dir(cfg, merged, repo)

    mdl = merged.setdefault("model", {})
    _resolve_checkpoint_in_model(mdl, molmcl_root, repo)

    if not mdl.get("checkpoint"):
        mdl["use_prompt"] = False

    return merged


def sync_chemvl_task_type_from_molmcl_yaml(cfg: Dict[str, Any]) -> None:
    """Align ``cfg['dataset']['task_type']`` with merged MolMCL yaml ``dataset.task``."""
    m = load_merged_molmcl_config(cfg)
    raw = str((m.get("dataset") or {}).get("task", "regression")).lower()
    if raw not in ("classification", "regression"):
        raw = "regression"
    cfg.setdefault("dataset", {})["task_type"] = raw


def preview_molmcl_epochs_batch(cfg: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Best-effort epochs/batch for logging (e.g. batch driver dry-run)."""
    try:
        m = load_merged_molmcl_config(cfg)
        return int(m.get("epochs", 0) or 0) or None, int(m.get("batch_size", 0) or 0) or None
    except (OSError, ValueError, TypeError, KeyError):
        return None, None
