"""Resolve finetuned log_dir/ckpt for MoleculeACE datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

CKPT_CANDIDATES = ("best.pth", "valid_best.pth", "train_best.pth", "ckpt.pth")


def _pick_ckpt(log_dir: Path, ckpt_hint: Optional[str] = None) -> Path:
    if ckpt_hint:
        ckpt = Path(str(ckpt_hint)).expanduser()
        if ckpt.is_file():
            return ckpt.resolve()
    for name in CKPT_CANDIDATES:
        candidate = log_dir / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no checkpoint found under {log_dir}")


def _latest_log_dir(checkpoints_root: Path, dataset_id: str) -> Optional[Path]:
    ds_root = checkpoints_root / dataset_id
    if not ds_root.is_dir():
        return None
    runs = sorted(
        [p for p in ds_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0].resolve() if runs else None


def _manifest_for_dataset(results_root: Path, dataset_id: str) -> Optional[Dict[str, str]]:
    preferred = results_root / f"{dataset_id}_all" / "run_manifest.json"
    if preferred.is_file():
        manifest_path = preferred
    else:
        matches = sorted(results_root.glob(f"{dataset_id}_*/run_manifest.json"))
        if not matches:
            return None
        manifest_path = matches[0]
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = data.get("resolved") or {}
    log_dir = resolved.get("log_dir")
    ckpt = resolved.get("ckpt")
    if not log_dir:
        return None
    log_dir_path = Path(str(log_dir)).expanduser().resolve()
    ckpt_path = _pick_ckpt(log_dir_path, ckpt)
    return {
        "log_dir": str(log_dir_path),
        "ckpt": str(ckpt_path),
        "source": f"manifest:{manifest_path}",
    }


def _runs_from_gradcam_preset(preset_path: Path, dataset_id: str) -> Optional[Dict[str, str]]:
    if not preset_path.is_file():
        return None
    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    runs = preset.get("runs") or {}
    entry = runs.get(dataset_id)
    if not isinstance(entry, dict):
        return None
    log_dir_raw = entry.get("log_dir")
    if not log_dir_raw:
        return None
    log_dir = Path(str(log_dir_raw)).expanduser().resolve()
    ckpt = _pick_ckpt(log_dir, entry.get("ckpt"))
    return {
        "log_dir": str(log_dir),
        "ckpt": str(ckpt),
        "source": f"preset:{preset_path}",
    }


def resolve_ckpt_from_log_dir(log_dir: str | Path, ckpt_hint: Optional[str] = None) -> str:
    """Resolve checkpoint file path under a finetuning log_dir."""
    log_dir_path = Path(log_dir).expanduser().resolve()
    return str(_pick_ckpt(log_dir_path, ckpt_hint))


def resolve_dataset_checkpoint(
    dataset_id: str,
    *,
    results_roots: Iterable[Path],
    gradcam_preset_path: Optional[Path] = None,
    checkpoints_root: Optional[Path] = None,
) -> Tuple[Path, Path, str]:
    """
    Resolve (log_dir, ckpt, source_label) for a dataset.

    Priority:
      1. run_manifest.json under results_roots
      2. presets/gradcam runs entry
      3. latest checkpoint run dir under checkpoints_root
    """
    for root in results_roots:
        hit = _manifest_for_dataset(root.expanduser().resolve(), dataset_id)
        if hit:
            return Path(hit["log_dir"]), Path(hit["ckpt"]), hit["source"]

    if gradcam_preset_path is not None:
        hit = _runs_from_gradcam_preset(gradcam_preset_path.expanduser().resolve(), dataset_id)
        if hit:
            return Path(hit["log_dir"]), Path(hit["ckpt"]), hit["source"]

    if checkpoints_root is not None:
        log_dir = _latest_log_dir(checkpoints_root.expanduser().resolve(), dataset_id)
        if log_dir is not None:
            ckpt = _pick_ckpt(log_dir)
            return log_dir, ckpt, f"checkpoints:{log_dir}"

    raise FileNotFoundError(
        f"could not resolve checkpoint for {dataset_id}; "
        "checked manifests, gradcam preset, and checkpoints root"
    )


def resolve_many_dataset_checkpoints(
    dataset_ids: Iterable[str],
    *,
    results_roots: Iterable[Path],
    gradcam_preset_path: Optional[Path] = None,
    checkpoints_root: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for dataset_id in sorted(set(dataset_ids)):
        log_dir, ckpt, source = resolve_dataset_checkpoint(
            dataset_id,
            results_roots=results_roots,
            gradcam_preset_path=gradcam_preset_path,
            checkpoints_root=checkpoints_root,
        )
        out[dataset_id] = {
            "log_dir": str(log_dir),
            "ckpt": str(ckpt),
            "source": source,
        }
    return out
