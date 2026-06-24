#!/usr/bin/env python3
"""
Offline-render molecular PNGs for non-default depiction presets (layout / style / zoom).

Writes images under ``{dataroot}/{dataset}/processed/{subdir}/`` where ``subdir`` matches
``dataloader.image_dataloader.depiction_processed_subdir`` for the given
``--render-canvas-px`` and ``--render-preset`` (e.g. ``224_layout_var``, ``224_zoom_50``).

Example (from repo root)::

    python tools/datasets/render_depiction_dataset.py \\
        --dataroot /path/to/MPP/classification \\
        --dataset bbbp \\
        --render-canvas-px 224 \\
        --render-preset zoom_50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from dataloader.image_dataloader import depiction_processed_subdir
from utils.depiction_constants import VALID_RENDER_PRESETS
from utils.rdkit_depiction_utils import smiles_to_pil


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", required=True, type=str, help="Dataset root (…/classification)")
    p.add_argument("--dataset", required=True, type=str, help="Dataset key, e.g. bbbp")
    p.add_argument(
        "--render-canvas-px",
        type=int,
        default=224,
        help="Square RDKit canvas side length in pixels (typically 224)",
    )
    p.add_argument(
        "--render-preset",
        type=str,
        default="default",
        choices=list(VALID_RENDER_PRESETS),
        help="Depiction preset",
    )
    p.add_argument("--max-mols", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--dry-run", action="store_true", help="Print paths only")
    args = p.parse_args()

    depiction = {
        "render_canvas_px": int(args.render_canvas_px),
        "render_preset": args.render_preset,
    }
    subdir = depiction_processed_subdir(depiction)
    proc_dir = os.path.join(args.dataroot, args.dataset, "processed")
    out_dir = os.path.join(proc_dir, subdir)
    csv_path = os.path.join(proc_dir, f"{args.dataset}_processed_ac.csv")

    if not os.path.isfile(csv_path):
        print(f"Missing CSV: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    if args.max_mols is not None:
        df = df.head(int(args.max_mols))

    print(f"Output dir: {out_dir}  (subdir={subdir!r})")
    if args.dry_run:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    canvas = int(args.render_canvas_px)
    preset = args.render_preset

    for _, row in df.iterrows():
        idx = int(row["index"])
        smi = str(row["smiles"])
        path = os.path.join(out_dir, f"{idx}.png")
        img = smiles_to_pil(smi, canvas_px=canvas, preset=preset)
        img.save(path)

    print(f"Wrote {len(df)} PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
