import os
from pathlib import Path
from typing import Any, Optional, Union

# Fallback when CHEMVL_DATA_ROOT is unset. Set CHEMVL_DATA_ROOT on any real machine.
DEFAULT_CHEMVL_DATA_ROOT = Path.home() / "chemvl-data"

# Legacy absolute paths rewritten to get_data_root() when loading configs.
LEGACY_DATA_ROOT_PATHS = (
    Path("/mnt/d/wsl-data/chemvl"),
)

DATA_ROOT_PLACEHOLDER = "{CHEMVL_DATA_ROOT}"


def _replace_legacy_data_root(value: str, legacy: str, root: str) -> str:
    if root == legacy or not value.startswith(legacy):
        return value
    if len(value) == len(legacy):
        return root
    if value[len(legacy)] in ("/", os.sep):
        return root + value[len(legacy):]
    return value


def expand_data_root_string(value: str) -> str:
    """Replace ``{CHEMVL_DATA_ROOT}`` and legacy author absolute paths with ``get_data_root()``."""
    if not isinstance(value, str):
        return value
    root = str(get_data_root())
    if DATA_ROOT_PLACEHOLDER in value:
        value = value.replace(DATA_ROOT_PLACEHOLDER, root)
    for legacy_path in LEGACY_DATA_ROOT_PATHS:
        value = _replace_legacy_data_root(value, str(legacy_path), root)
    value = _replace_legacy_data_root(value, str(DEFAULT_CHEMVL_DATA_ROOT), root)
    return value


def expand_data_root_strings(obj: Any) -> Any:
    """Recursively expand ``{CHEMVL_DATA_ROOT}`` in dict/list/str configs."""
    if isinstance(obj, dict):
        return {k: expand_data_root_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_data_root_strings(v) for v in obj]
    if isinstance(obj, str):
        return expand_data_root_string(obj)
    return obj


def get_project_root() -> Path:
    """
    Resolve project root directory robustly.
    Assumes project root contains one of:
    - .git
    - setup.py
    - utils/
    """
    cur = Path(__file__).resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists():
            return p
        if (p / "setup.py").exists():
            return p
        if (p / "utils").exists():
            return p
    raise RuntimeError("Cannot locate project root.")


def get_data_root() -> Path:
    """
    Root directory for large artifacts (datasets, checkpoints, logs, prior caches).

    Resolution:
    1. If environment variable ``CHEMVL_DATA_ROOT`` is set and non-empty after strip,
       return that path (``expanduser`` + absolute, not required to exist).
    2. Otherwise return ``DEFAULT_CHEMVL_DATA_ROOT`` (``~/chemvl-data``).

    Set ``CHEMVL_DATA_ROOT`` to a writable location before running experiments.
    """
    raw = os.environ.get("CHEMVL_DATA_ROOT", "")
    if raw and raw.strip():
        return Path(os.path.abspath(os.path.expanduser(raw.strip())))
    return Path(DEFAULT_CHEMVL_DATA_ROOT)


def get_knowledge_cache_dir() -> Path:
    """Directory for prior-knowledge memory pickles: ``{get_data_root()}/cache_for_knowledge``."""
    return get_data_root() / "cache_for_knowledge"


def get_descriptor_only_text_cache_dir() -> Path:
    """
    Descriptor-only @text embedding cache (separate from prior ``cache_for_knowledge``).

    Resolution:
    1. If ``CHEMVL_DESCRIPTOR_ONLY_TEXT_CACHE`` is set and non-empty, return that absolute path.
    2. Otherwise ``{get_data_root()}/cache_for_descriptor-only_knowledge``.
    """
    raw = os.environ.get("CHEMVL_DESCRIPTOR_ONLY_TEXT_CACHE", "")
    if raw and str(raw).strip():
        return Path(os.path.abspath(os.path.expanduser(str(raw).strip())))
    return get_data_root() / "cache_for_descriptor-only_knowledge"


def resolve_optional_dir_under_project(raw: Optional[str]) -> Optional[Path]:
    """
    If *raw* is None or blank, return None.
    If *raw* is an absolute path, return Path(abspath(expanduser(raw))).
    Otherwise return ``get_project_root() / raw`` (relative paths are repo-relative).
    """
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if os.path.isabs(s):
        return Path(os.path.abspath(os.path.expanduser(s)))
    return get_project_root() / s
