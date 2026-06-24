#!/usr/bin/env python3
"""Render or verify ChemVL image-pretraining PNGs from 10M-106mds metadata.

Layout matches ``ordinalclip/configs/.../mol-10M-106mds/local.yaml``::

    pretraining_datasets/10M-106mds/mds.csv
    pretraining_datasets/10M-106mds/{train,test}_data/data_list/{train,test}.txt
    pretraining_datasets/images-10M@224px/{train,test}_data/*.png

Examples (from repo root)::

    export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data

    python tools/datasets/render_pretrain_images.py render --split train --skip-existing
    python tools/datasets/render_pretrain_images.py render --split test --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.pretrain_image_render import DEFAULT_CANVAS_PX, smiles_to_pretrain_pil

DEFAULT_PRETRAIN_REL = "pretraining_datasets"
SPLIT_PRESETS = {
    "train": {
        "list_file": "10M-106mds/train_data/data_list/train.txt",
        "images_root": "images-10M@224px/train_data",
    },
    "test": {
        "list_file": "10M-106mds/test_data/data_list/test.txt",
        "images_root": "images-10M@224px/test_data",
    },
}


def _pretraining_root(data_root: Path) -> Path:
    return data_root / DEFAULT_PRETRAIN_REL


def _resolve_data_root(raw: str | None) -> Path:
    if raw:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    env = os.environ.get("CHEMVL_DATA_ROOT", "").strip()
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    try:
        from utils.path_utils import get_data_root

        return Path(get_data_root())
    except Exception:
        return Path.home() / "chemvl-data"


def _normalize_sample_id(name: str) -> str:
    return Path(name).stem


def load_smiles_table(metadata_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "index" not in reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError(f"{metadata_path} must contain 'index' and 'smiles' columns.")
        for row in reader:
            key = str(row["index"]).strip()
            lookup[key] = str(row["smiles"]).strip()
    return lookup


def lookup_smiles(smiles_map: dict[str, str], sample_id: str) -> str | None:
    key = str(sample_id).strip()
    if key in smiles_map:
        return smiles_map[key]
    if key.isdigit():
        alt = str(int(key))
        if alt in smiles_map:
            return smiles_map[alt]
    return None


def parse_list_file(list_path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with list_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            rows.append((parts[0], parts[1:]))
    return rows


def _rgb_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _generated_rgb(smiles: str, canvas_px: int) -> np.ndarray:
    return np.asarray(smiles_to_pretrain_pil(smiles, canvas_px=canvas_px))


def cmd_verify(args: argparse.Namespace) -> int:
    pretrain = _pretraining_root(args.data_root)
    list_path = pretrain / args.list_file
    images_root = pretrain / args.images_root
    metadata_path = pretrain / args.metadata

    if not list_path.is_file():
        print(f"Missing list file: {list_path}", file=sys.stderr)
        return 1
    if not metadata_path.is_file():
        print(f"Missing metadata: {metadata_path}", file=sys.stderr)
        return 1

    print(f"Loading SMILES table from {metadata_path} ...", flush=True)
    smiles_map = load_smiles_table(metadata_path)
    entries = parse_list_file(list_path)
    if not entries:
        print("List file is empty.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    if args.sample_n is not None and args.sample_n < len(entries):
        entries = rng.sample(entries, args.sample_n)

    report_dir = args.report_dir
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

    matched = 0
    mismatched = 0
    missing_ref = 0
    missing_smiles = 0

    for rel_path, _labels in entries:
        ref_path = images_root / rel_path
        sample_id = _normalize_sample_id(rel_path)
        smiles = lookup_smiles(smiles_map, sample_id)
        if smiles is None:
            missing_smiles += 1
            print(f"[missing-smiles] {rel_path} id={sample_id}", file=sys.stderr)
            continue
        if not ref_path.is_file():
            missing_ref += 1
            print(f"[missing-ref] {ref_path}", file=sys.stderr)
            continue

        ref = _rgb_array(ref_path)
        gen = _generated_rgb(smiles, args.canvas_px)
        if ref.shape != gen.shape:
            mismatched += 1
            print(f"[shape-mismatch] {rel_path} ref={ref.shape} gen={gen.shape}", file=sys.stderr)
            continue
        if np.array_equal(ref, gen):
            matched += 1
        else:
            mismatched += 1
            print(
                f"[pixel-mismatch] {rel_path} id={sample_id} smiles={smiles[:80]!r}",
                file=sys.stderr,
            )
            if report_dir is not None:
                diff = np.abs(ref.astype(np.int16) - gen.astype(np.int16)).astype(np.uint8)
                stem = Path(rel_path).stem
                Image.fromarray(ref).save(report_dir / f"{stem}_ref.png")
                Image.fromarray(gen).save(report_dir / f"{stem}_gen.png")
                Image.fromarray(diff).save(report_dir / f"{stem}_diff.png")

    checked = matched + mismatched
    print(f"\nVerify summary: matched={matched} mismatched={mismatched} checked={checked}")
    if missing_ref:
        print(f"  missing reference PNGs: {missing_ref}")
    if missing_smiles:
        print(f"  missing SMILES in metadata: {missing_smiles}")
    if mismatched or missing_ref or missing_smiles:
        return 1
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    pretrain = _pretraining_root(args.data_root)
    list_path = pretrain / args.list_file
    metadata_path = pretrain / args.metadata
    out_root = Path(args.out_images_root)
    if not out_root.is_absolute():
        out_root = pretrain / out_root

    if not list_path.is_file():
        print(f"Missing list file: {list_path}", file=sys.stderr)
        return 1
    if not metadata_path.is_file():
        print(f"Missing metadata: {metadata_path}", file=sys.stderr)
        return 1

    print(f"Loading SMILES table from {metadata_path} ...", flush=True)
    smiles_map = load_smiles_table(metadata_path)
    entries = parse_list_file(list_path)
    if args.max_lines is not None:
        entries = entries[: int(args.max_lines)]

    written = 0
    skipped = 0
    errors = 0

    for rel_path, _labels in entries:
        out_path = out_root / rel_path
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        sample_id = _normalize_sample_id(rel_path)
        smiles = lookup_smiles(smiles_map, sample_id)
        if smiles is None:
            errors += 1
            print(f"[missing-smiles] {rel_path} id={sample_id}", file=sys.stderr)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = smiles_to_pretrain_pil(smiles, canvas_px=args.canvas_px)
        img.save(out_path)
        written += 1

    print(f"Render summary: written={written} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


def _apply_split_preset(args: argparse.Namespace) -> None:
    if not args.split:
        return
    preset = SPLIT_PRESETS[args.split]
    if args.list_file is None:
        args.list_file = preset["list_file"]
    if args.images_root is None:
        args.images_root = preset["images_root"]
    if hasattr(args, "out_images_root") and args.out_images_root is None:
        args.out_images_root = preset["images_root"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="CHEMVL_DATA_ROOT (parent of pretraining_datasets/)",
    )
    common.add_argument(
        "--metadata",
        type=str,
        default="10M-106mds/mds.csv",
        help="Relative to pretraining_datasets/",
    )
    common.add_argument("--list-file", type=str, default=None, help="Relative to pretraining_datasets/")
    common.add_argument("--images-root", type=str, default=None, help="Relative to pretraining_datasets/")
    common.add_argument("--split", choices=("train", "test"), default=None)
    common.add_argument("--canvas-px", type=int, default=DEFAULT_CANVAS_PX)

    v = sub.add_parser("verify", parents=[common], help="Pixel-compare regenerated vs on-disk PNGs")
    v.add_argument("--sample-n", type=int, default=200, help="Random sample size (default: 200)")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--report-dir", type=Path, default=None, help="Save ref/gen/diff for mismatches")

    r = sub.add_parser("render", parents=[common], help="Write PNGs from metadata list")
    r.add_argument(
        "--out-images-root",
        type=str,
        default=None,
        help="Output root relative to pretraining_datasets/ (default: same as --images-root)",
    )
    r.add_argument("--max-lines", type=int, default=None)
    r.add_argument("--skip-existing", action="store_true")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.data_root = _resolve_data_root(str(args.data_root) if args.data_root else None)
    _apply_split_preset(args)

    if args.list_file is None:
        print("Provide --list-file or --split", file=sys.stderr)
        return 2
    if args.command == "verify" and args.images_root is None:
        print("Provide --images-root or --split", file=sys.stderr)
        return 2

    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "render":
        return cmd_render(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
