from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_results_root() -> str:
    try:
        from utils.path_utils import get_data_root

        return str(get_data_root() / "results")
    except Exception:
        return str(Path.home() / "chemvl-data" / "results")


def git_commit(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def preset_sha256(data: Dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_results_base(preset: Dict[str, Any]) -> Path:
    outs = preset.get("outputs") or {}
    base = outs.get("results_base") or os.environ.get("CHEMVL_RESULTS_ROOT") or _default_results_root()
    return Path(str(base)).expanduser().resolve()


def resolve_output_dir(preset: Dict[str, Any]) -> Path:
    outs = preset.get("outputs") or {}
    base = resolve_results_base(preset)
    if outs.get("out_dir"):
        p = Path(str(outs["out_dir"]))
        return p if p.is_absolute() else (base / p)
    slug = outs.get("slug") or preset.get("name") or "representation_analysis"
    rel = Path(str(slug))
    if rel.parts and rel.parts[0] == "aggregates":
        return base / rel
    return base / "aggregates" / rel


def write_manifest(
    manifest_path: Path,
    *,
    preset_path: Path,
    preset: Dict[str, Any],
    source_csv: Path,
    plot_paths: List[Path],
    extra_artifacts: Dict[str, Any],
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(REPO_ROOT),
        "preset_path": str(preset_path.resolve()),
        "preset_sha256": preset_sha256(preset),
        "preset": preset,
        "results_base": str(resolve_results_base(preset)),
        "source_csv": str(source_csv.resolve()),
        "plot_paths": [str(x.resolve()) for x in plot_paths],
        **extra_artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

