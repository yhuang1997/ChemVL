#!/usr/bin/env python3
"""
Build ChemVL-style processed files for MoleculeACE targets (MolMCL CSV convention).

MolMCL ``MoleculeDataset`` for CHEMBL* expects a CSV with columns ``smiles`` and ``y``
(see ``external/MolMCL/molmcl/finetune/loader.py``). ``y`` is used as-is for regression;
``moleculeace_split(..., in_log10=True)`` assumes log10(nM) style values unless you pass
``in_log10=False`` (then values are converted with ``-log10`` inside the splitter).

This script writes:
  ``{dataroot}/{dataset}/processed/{dataset}_processed_ac.csv``
with columns ``index``, ``smiles``, ``label`` (single-task: one float per row as string),
and optionally renders ``{dataroot}/{dataset}/processed/{subdir}/{index}.png``.

Example:
  python tools/datasets/prepare_moleculeace_chemvl.py \\
    --input-csv /path/to/CHEMBL2047.csv \\
    --dataroot datasets/downstream \\
    --dataset CHEMBL2047 \\
    --render-images
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataloader.image_dataloader import Smiles2Img  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Prepare MoleculeACE CSV + optional PNGs for ChemVL.")
    p.add_argument("--input-csv", required=True, help="MolMCL-style CSV with smiles + y columns.")
    p.add_argument("--dataroot", required=True, help="ChemVL dataroot (parent of dataset folder).")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. CHEMBL2047_EC50.")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name.")
    p.add_argument("--y-col", default="y", help="Regression label column (MolMCL uses 'y').")
    p.add_argument("--render-images", action="store_true", help="If set, render PNGs with RDKit.")
    p.add_argument("--canvas", type=int, default=224, help="Image size when --render-images.")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns or args.y_col not in df.columns:
        raise SystemExit(f"CSV must contain columns {args.smiles_col!r} and {args.y_col!r}; got {list(df.columns)}")

    processed = os.path.join(args.dataroot, args.dataset, "processed")
    os.makedirs(processed, exist_ok=True)

    subdir = str(args.canvas)
    img_dir = os.path.join(processed, subdir)
    if args.render_images:
        os.makedirs(img_dir, exist_ok=True)

    rows = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        smi = row[args.smiles_col]
        y = row[args.y_col]
        rows.append({"index": idx, "smiles": smi, "label": str(float(y))})
        if args.render_images:
            out_png = os.path.join(img_dir, f"{idx}.png")
            if not os.path.isfile(out_png):
                img = Smiles2Img(smi, size=args.canvas, savePath=out_png)
                if img is None:
                    print(f"Warning: could not render index {idx} smiles={smi!r}")

    out_csv = os.path.join(processed, f"{args.dataset}_processed_ac.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} rows).")
    if args.render_images:
        print(f"Images under {img_dir}")


if __name__ == "__main__":
    main()
