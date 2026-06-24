"""
Write reproducibility metadata under ``<log_dir_base_after_exp_name>/_repro/``.

Used by ``ablation_study_run.py``: git revision, working tree status, optional
diff when dirty, and an append-only invocation log.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_git(args: List[str], cwd: str) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


def write_repro_bundle(
    repro_dir: Path,
    argv: List[str],
    repo_root: str,
    *,
    save_git_diff: bool,
) -> None:
    """
    Create ``repro_dir`` and write ``git_head.txt``, ``git_status.txt``;
    optionally ``git_diff.patch`` when the tree is dirty and ``save_git_diff``.
    Appends one block to ``invocations.txt`` (timestamp + argv).
    """
    repro_dir.mkdir(parents=True, exist_ok=True)
    head = _run_git(["rev-parse", "HEAD"], repo_root)
    (repro_dir / "git_head.txt").write_text(head + "\n" if head else "(git rev-parse failed)\n", encoding="utf-8")

    status = _run_git(["status", "--porcelain"], repo_root)
    (repro_dir / "git_status.txt").write_text(status + ("\n" if status else ""), encoding="utf-8")

    if save_git_diff and status:
        diff = _run_git(["diff"], repo_root)
        if diff:
            (repro_dir / "git_diff.patch").write_text(diff + "\n", encoding="utf-8")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    inv_line = f"--- {ts} ---\n{' '.join(argv)}\n\n"
    with open(repro_dir / "invocations.txt", "a", encoding="utf-8") as f:
        f.write(inv_line)


def append_run_jsonl(repro_dir: Path, record: Dict[str, Any]) -> None:
    repro_dir.mkdir(parents=True, exist_ok=True)
    path = repro_dir / "runs.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
