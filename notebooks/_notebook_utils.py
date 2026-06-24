"""Shared helpers for ChemVL demo notebooks (a/b/c)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

_NOTEBOOK_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _NOTEBOOK_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.interpret.presets_loader import resolve_data_path, resolve_data_path_str
from utils.path_utils import get_data_root, get_project_root

_NOTEBOOK_DATA_ROOT_FILE = _NOTEBOOK_DIR / ".chemvl_data_root"
_CKPT_PROBE = Path("checkpoints/pretraining/RN50px224.ckpt")


def configure_chemvl_data_root(
    *,
    candidates: Optional[Sequence[Path]] = None,
) -> Path:
    """
    Ensure ``CHEMVL_DATA_ROOT`` is set before any ``data_root()`` / ``get_data_root()`` call.

    Resolution order:
    1. Existing ``CHEMVL_DATA_ROOT`` environment variable
    2. One-line path in ``notebooks/.chemvl_data_root`` (optional local override)
    3. Auto-detect among common roots that contain the pretraining RN50 checkpoint
    """
    raw = os.environ.get("CHEMVL_DATA_ROOT", "").strip()
    if raw:
        resolved = Path(os.path.abspath(os.path.expanduser(raw)))
        os.environ["CHEMVL_DATA_ROOT"] = str(resolved)
        return resolved

    if _NOTEBOOK_DATA_ROOT_FILE.is_file():
        hint = _NOTEBOOK_DATA_ROOT_FILE.read_text(encoding="utf-8").strip()
        if hint:
            resolved = Path(os.path.abspath(os.path.expanduser(hint)))
            os.environ["CHEMVL_DATA_ROOT"] = str(resolved)
            return resolved

    for cand in candidates or (
        Path("/mnt/d/wsl-data/chemvl"),
        Path.home() / "chemvl-data",
    ):
        root = Path(cand).expanduser()
        if (root / _CKPT_PROBE).is_file():
            resolved = root.resolve()
            os.environ["CHEMVL_DATA_ROOT"] = str(resolved)
            return resolved

    raise EnvironmentError(
        "CHEMVL_DATA_ROOT is not set and no data root with "
        f"{_CKPT_PROBE.as_posix()} was found.\n"
        "Set export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data, or write that path to "
        "notebooks/.chemvl_data_root (one line), then run tools/hf_download.py."
    )


def repo_root() -> Path:
    return get_project_root()


def data_root() -> Path:
    return get_data_root()


def configure_quiet_demo_logging() -> None:
    """Enable quiet third-party logging in-process and for subsequent ``run_cmd`` calls."""
    os.environ.setdefault("CHEMVL_NOTEBOOK_QUIET", "1")
    try:
        from utils.notebook_quiet_logging import apply_notebook_quiet_logging

        apply_notebook_quiet_logging()
    except Exception:
        pass


def ensure_repo_on_path() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_finetune_ckpt(run_dir: Union[str, Path]) -> Path:
    """Prefer Hub canonical ``best.pth``; fall back to ``valid_best.pth``."""
    d = Path(run_dir)
    for name in ("best.pth", "valid_best.pth", "ckpt.pth"):
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No finetune checkpoint under {d} (tried best.pth, valid_best.pth, ckpt.pth)"
    )


# JSON preset fields whose string values are filesystem paths (not mode names / labels).
_PRESET_PATH_KEYS = frozenset({
    "model_ckpt",
    "pretraining_resume",
    "cache_base",
    "validation_csv_file",
    "descriptor_info_path",
    "descriptor_cache_file",
    "downstream_csv_file",
    "finetune_cfg_path",
    "out_dir",
    "csv_file",
    "regression_labels_csv",
})


def _looks_like_path(value: str) -> bool:
    """True for absolute paths or strings with path separators (not bare tokens like ``downstream``)."""
    text = str(value).strip()
    if not text:
        return False
    if text.startswith(("/", "~")):
        return True
    return "/" in text or "\\" in text


def _resolve_preset_value(key: str | None, value: str, *, root: Path) -> str:
    if key not in _PRESET_PATH_KEYS and not _looks_like_path(value):
        return value
    resolved = resolve_data_path_str(value, repo_root=root)
    return resolved if resolved is not None else value


def render_preset_template(
    template_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Load JSON preset template, substitute path placeholders, optional deep overrides."""
    template_path = Path(template_path)
    text = template_path.read_text(encoding="utf-8")
    root = repo_root()
    dr = str(data_root())
    text = text.replace("{CHEMVL_DATA_ROOT}", dr).replace("{REPO_ROOT}", str(root))
    preset = json.loads(text)

    if overrides:
        _deep_update(preset, overrides)

    def _resolve_obj(obj: Any, key: str | None = None) -> Any:
        if isinstance(obj, str):
            return _resolve_preset_value(key, obj, root=root)
        if isinstance(obj, dict):
            return {k: _resolve_obj(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_obj(v, key) for v in obj]
        return obj

    preset = _resolve_obj(preset, None)

    ckpt = preset.get("common", {}).get("model_ckpt")
    if ckpt and not Path(str(ckpt)).is_file():
        try:
            preset["common"]["model_ckpt"] = str(resolve_finetune_ckpt(Path(str(ckpt)).parent))
        except FileNotFoundError:
            pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    return out_path


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    quiet_logging: bool = True,
) -> subprocess.CompletedProcess:
    cwd = cwd or repo_root()
    if quiet_logging:
        from utils.notebook_quiet_logging import notebook_quiet_env

        merged_env = notebook_quiet_env(env)
    else:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
    printable = " ".join(cmd)
    print(f"$ {printable}")
    return subprocess.run(list(cmd), cwd=str(cwd), env=merged_env, check=check)


def assert_paths_exist(paths: Iterable[Union[str, Path]], *, label: str = "Required asset") -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{label} missing:\n" + "\n".join(f"  - {m}" for m in missing)
            + "\nSet CHEMVL_DATA_ROOT and run tools/hf_download.py if needed."
        )


def display_pngs(directory: Union[str, Path], pattern: str = "*.png") -> None:
    from IPython.display import Image, display

    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if not paths:
        print(f"No PNG files matching {pattern!r} under {directory}")
        return
    for path in paths:
        print(path.name)
        display(Image(filename=str(path)))


def subsample_csv(
    src_csv: Union[str, Path],
    dst_csv: Union[str, Path],
    *,
    n: int,
    seed: int = 2024,
) -> Path:
    import pandas as pd

    df = pd.read_csv(src_csv)
    n = min(int(n), len(df))
    out = df.sample(n=n, random_state=seed).reset_index(drop=True)
    dst = Path(dst_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote subsample ({n} rows) -> {dst}")
    return dst
