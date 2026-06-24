"""Representation analysis CLI (preset-driven, traceable outputs)."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.representation_analysis.core.pipeline import run_preset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", type=Path, required=True, help="Path to representation-analysis preset JSON")
    args = p.parse_args()

    return run_preset(args.preset.resolve())

