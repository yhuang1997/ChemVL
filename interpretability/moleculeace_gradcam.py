#!/usr/bin/env python3
"""
Demo CLI — 224px MoleculeACE Grad-CAM (single checkpoint, quick try)

For full-dataset batch runs with series mining, use::

    scripts/experiments/moleculeace_interpretability/run_from_preset.py

Prerequisites
-------------
A finetuned ``.pth`` checkpoint whose **parent directory** also contains
``config.json`` from ``finetune_moleculeace.py``::

    .../CHEMBL204_Ki/20250419_020900/
    ├── config.json
    └── valid_best.pth

Quick start
-----------
Visualize six validation molecules::

    python interpret.py visual \\
        --ckpt /path/to/CHEMBL204_Ki/20250419_020900/valid_best.pth \\
        --split val \\
        --max-molecules 6

Outputs land in ``{output_dir}/{dataset}_{split}/``::

    CHEMBL204_Ki_val/
    ├── gradcam_gallery.png              # mosaic (structure | finetuned CAM)
    ├── CHEMBL204_Ki_val_Grad-CAM.png    # legacy mosaic name
    ├── CHEMBL204_Ki_val_interpretability.csv
    ├── plots/mol_0000.png …
    └── run_manifest.json

Optional — add pretrained descriptor panels (slower)::

    python interpret.py visual \\
        --ckpt /path/to/.../valid_best.pth \\
        --split val --max-molecules 4 --upstream
"""
from __future__ import annotations

import argparse
import os
import sys

from utils.interpretability_support.moleculeace_gradcam_batch import (
    MoleculeAceGradcamConfig,
    run_moleculeace_gradcam,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Finetuned checkpoint (.pth); log_dir is inferred as its parent directory",
    )
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    p.add_argument(
        "--max-molecules",
        type=int,
        default=6,
        help="Max samples drawn from pool (without replacement). Use -1 for the entire pool.",
    )
    p.add_argument(
        "--gradcam-batch-size",
        type=int,
        default=8,
        help="Mini-batch size for Grad-CAM forward (lower if GPU OOM on full datasets).",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-dir", type=str, default="results/moleculeace_interpretability")
    p.add_argument("--upstream", action="store_true", help="Also run pretrained descriptor Grad-CAM (slower)")
    p.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Pretrained RN50px224 path for --upstream (default: data root checkpoint)",
    )
    p.add_argument(
        "--descriptors",
        type=str,
        nargs="*",
        default=None,
        help="Descriptor names for --upstream (defaults to a small built-in list)",
    )
    return p


def main() -> int:
    print(
        "Note: random-split MoleculeACE Grad-CAM batch demo is a maintainer library entry.\n"
        "Readers should use: python interpret.py visual run --preset analysis/interpret/presets/...\n",
        file=sys.stderr,
    )
    args = build_arg_parser().parse_args()
    try:
        config = MoleculeAceGradcamConfig.from_demo_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    config.reproduce_script_path = os.path.abspath(__file__)
    return run_moleculeace_gradcam(config)


if __name__ == "__main__":
    sys.exit(main())
