"""Load and list interpret showcase presets under ``analysis/interpret/presets/``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.path_utils import get_data_root

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_path(value: str | None, *, repo_root: Optional[Path] = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    text = os.path.expanduser(str(value))
    data_root = str(get_data_root())
    text = text.replace("{CHEMVL_DATA_ROOT}", data_root)
    root = repo_root or _repo_root()
    text = text.replace("{REPO_ROOT}", str(root))
    path = Path(text)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def resolve_data_path_str(value: str | None, *, repo_root: Optional[Path] = None) -> str | None:
    path = resolve_data_path(value, repo_root=repo_root)
    return str(path) if path is not None else None


def list_presets() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not PRESETS_DIR.is_dir():
        return rows
    for path in sorted(PRESETS_DIR.glob("*.yaml")):
        preset = load_preset(path)
        rows.append(
            {
                "file": str(path.relative_to(_repo_root())),
                "id": str(preset.get("id", path.stem)),
                "title": str(preset.get("title", path.stem)),
                "mode": str(preset.get("mode", "")),
            }
        )
    return rows


def load_preset(path: Path | str) -> Dict[str, Any]:
    preset_path = Path(path)
    if not preset_path.is_file():
        raise FileNotFoundError(f"Preset not found: {preset_path}")
    with preset_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a YAML mapping: {preset_path}")
    data["_path"] = str(preset_path.resolve())
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("preset.defaults must be a mapping")
    merged = {**defaults, **{k: v for k, v in data.items() if k != "defaults"}}
    merged["id"] = merged.get("id") or preset_path.stem
    merged["defaults"] = defaults
    return merged


def checkpoint_paths_for_dataset(
    preset: Dict[str, Any],
    dataset: str,
    *,
    ckpt_basename: str | None = None,
) -> tuple[Path, Path]:
    root = resolve_data_path(preset.get("checkpoint_root"))
    if root is None:
        raise ValueError("preset.checkpoint_root is required")
    basename = ckpt_basename or preset.get("checkpoint_basename") or "ckpt.pth"
    cfg_path = root / dataset / "config.json"
    ckpt_path = root / dataset / basename
    for path in (cfg_path, ckpt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    return cfg_path, ckpt_path
