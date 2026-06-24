#!/usr/bin/env python3
"""Scan public-export tree for hardcoded data paths and Tier-1 path hygiene."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[2]

HARDcoded = re.compile(r"/mnt/d/wsl-data/chemvl")
TRAIN_VALID_BEST = re.compile(r"\b(train_best|valid_best)\.pth\b")

CKPT_SCAN_GLOBS = [
    "configs/tutorials/**",
    "analysis/interpret/presets/**",
]

SCAN_GLOBS = [
    "configs/tutorials/**",
    "analysis/interpret/**",
    "ordinalclip/configs/**/local*.yaml",
    "ordinalclip/configs/**/graph_*_local.yaml",
    "ordinalclip/configs/default.yaml",
    "finetune_*.py",
    "pretrain*.py",
    "interpret.py",
    "tools/hf_download.py",
    "README.md",
    "docs/data/HF_DATASET_CARD.md",
]

SKIP_PARTS = (
    "configs/external/",
    "configs/ablation_",
    "analysis/representation_analysis/",
    "docs/public_export/PATH_VALIDATION.md",
)


def iter_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    unique = sorted({p for p in out if p.is_file()})
    filtered: List[Path] = []
    for p in unique:
        rel = p.relative_to(root).as_posix()
        if any(skip in rel for skip in SKIP_PARTS):
            continue
        filtered.append(p)
    return filtered


def scan_file(path: Path) -> Tuple[List[int], List[int]]:
    hard_lines: List[int] = []
    ckpt_lines: List[int] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return hard_lines, ckpt_lines
    for i, line in enumerate(text.splitlines(), start=1):
        if HARDcoded.search(line):
            hard_lines.append(i)
        if TRAIN_VALID_BEST.search(line):
            ckpt_lines.append(i)
    return hard_lines, ckpt_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public-facing path strings.")
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    root = args.root.resolve()

    hard_hits: List[str] = []
    ckpt_hits: List[str] = []
    for path in iter_files(root, SCAN_GLOBS):
        rel = path.relative_to(root).as_posix()
        hard_lines, _ = scan_file(path)
        if hard_lines:
            hard_hits.append(f"{rel}: lines {hard_lines}")
    for path in iter_files(root, CKPT_SCAN_GLOBS):
        rel = path.relative_to(root).as_posix()
        _, ckpt_lines = scan_file(path)
        if ckpt_lines:
            ckpt_hits.append(f"{rel}: lines {ckpt_lines}")

    print(f"root: {root}")
    print(f"files scanned: {len(iter_files(root, SCAN_GLOBS))}")
    print(f"hardcoded /mnt/d/wsl-data/chemvl: {len(hard_hits)}")
    for row in hard_hits:
        print(f"  {row}")
    print(f"train_best|valid_best in public scan set: {len(ckpt_hits)}")
    for row in ckpt_hits:
        print(f"  {row}")

    return 1 if (hard_hits or ckpt_hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
