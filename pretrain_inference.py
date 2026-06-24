"""
Upstream (pretraining) descriptor-conditioned inference for small molecule batches.

Public release entry replacing ``01_upstream_inference.py``. Runs OrdinalCLIP-style
descriptor logits for selected molecular descriptors and writes CSV under ``--output_dir``.

Set ``CHEMVL_DATA_ROOT`` (see ``utils.path_utils.get_data_root``) before running.
Default checkpoint when ``--ckpt`` is omitted::

    {CHEMVL_DATA_ROOT}/checkpoints/pretraining/RN50px224.ckpt

Example::

    python pretrain_inference.py \\
        --config ordinalclip/configs/default.yaml \\
        --config ordinalclip/configs/base_cfgs/data_cfg/datasets/mol-10M-106mds/local.yaml

    python pretrain_inference.py ... \\
        --mds fr_benzene --mds NOCount \\
        --smiles "CCO" --smiles "CC(=O)O" \\
        --output_dir results/notebooks/a_md_inference

    python pretrain_inference.py ... \\
        --validation-csv "$CHEMVL_DATA_ROOT/pretraining_datasets/10M-106mds/mds-validation.csv" \\
        --num-samples 10 --seed 0 \\
        --output_dir results/notebooks/a_md_validation
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd

from utils.path_utils import get_data_root
from utils.upstream_infer_utils import *


DEFAULT_MDS = [
    "fr_benzene",
    "fr_halogen",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
]

DEFAULT_SMILES = [
    "CCC(C)CCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](COP(=O)(O)OC[C@@H](COC(=O)CCCCCCCCC(C)C)OC(=O)CCCCCCCCCCCCCCCCCC(C)C)O)OC(=O)CCCCCCCCCCCCCCCCCCC(C)C",
    "C1CN(CCN1C(=O)C(CC2=C(C=C(C=C2)Cl)Cl)N)C(=O)C3(C=CC=CC3C(F)(F)F)Cl",
    "C1=CC=C(C=C1)C2=CC=CC=C2NC3=C4C(=CC=C3)OC5=C4C=C(C=C5)C6=CC7=C(C=C6)OC8=C7C=C(C=C8)N9C1=CC=CC=C1N1C9=NC2=CC=CC=C21",
    "C1C(CN1C2=NC=C(C=N2)Cl)CN3C(=O)C=CC(=N3)C4=CN=CC=C4",
]


def _default_pretrain_ckpt() -> str:
    return os.path.join(get_data_root(), "checkpoints/pretraining/RN50px224.ckpt")


def _resolve_inputs(args) -> Tuple[List[str], List[str], List[str], List[str]]:
    mds = list(args.mds) if args.mds else list(DEFAULT_MDS)

    if args.validation_csv:
        if args.smiles:
            raise SystemExit("Use either --validation-csv or --smiles, not both.")
        csv_path = args.validation_csv
        if not os.path.isfile(csv_path):
            raise SystemExit(f"Missing validation CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if "smiles" not in df.columns:
            raise SystemExit(f"Column 'smiles' not found in {csv_path}")
        n = int(args.num_samples)
        if n > 0 and len(df) > n:
            df = df.sample(n=n, random_state=int(args.seed))
        smiles = df["smiles"].astype(str).tolist()
        names_col = str(args.names_column)
        if names_col in df.columns:
            names = df[names_col].astype(str).tolist()
        else:
            names = [f"Mol {i}" for i in range(len(smiles))]
        return mds, smiles, names, mds

    smiles = list(args.smiles) if args.smiles else list(DEFAULT_SMILES)
    if args.names:
        names = list(args.names)
        if len(names) != len(smiles):
            raise SystemExit("--names count must match --smiles count.")
    else:
        names = [f"Mol {i}" for i in range(len(smiles))]
    return mds, smiles, names, mds


if __name__ == "__main__":
    from utils.notebook_quiet_logging import apply_notebook_quiet_logging

    apply_notebook_quiet_logging()
    default_ckpt = _default_pretrain_ckpt()
    parser = argparse.ArgumentParser(
        description="Pretrained descriptor inference (public Tier-1 entry).",
    )
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=f"Pretrained RN50px224 checkpoint (default: {default_ckpt})",
    )
    parser.add_argument("--output_dir", type=str, default="md_inference_results/")
    parser.add_argument("--mds", action="append", type=str, default=[], help="Descriptor task(s) to predict")
    parser.add_argument("--smiles", action="append", type=str, default=[], help="SMILES string(s)")
    parser.add_argument("--names", action="append", type=str, default=[], help="Optional molecule name(s)")
    parser.add_argument(
        "--validation-csv",
        type=str,
        default=None,
        help="Sample molecules from a validation CSV (requires 'smiles' column)",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Rows to sample from --validation-csv")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --validation-csv sampling")
    parser.add_argument(
        "--names-column",
        type=str,
        default="index",
        help="Column used as molecule names when --validation-csv is set",
    )
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = default_ckpt

    mds, smiles, molecule_names, keys = _resolve_inputs(args)

    cfg = parse_cfg(args)
    runner = load_checkpoint(cfg)

    images, logits, gts = inference(runner, smiles, keys=keys)

    logits_per_descriptor = []
    md_gts = []
    for i in range(len(mds)):
        logits_per_descriptor.append([logit[i] for logit in logits])
        md_gts.append(gts[:, i])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_as_csv(molecule_names, smiles, logits_per_descriptor, md_gts, mds, args.output_dir)

    # Save rendered input images for notebook visualization (optional sidecar).
    if images is not None:
        try:
            import numpy as np
            from PIL import Image

            img_dir = os.path.join(args.output_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            for i, (name, img) in enumerate(zip(molecule_names, images)):
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
                out = os.path.join(img_dir, f"{i:03d}_{safe}.png")
                Image.fromarray(np.asarray(img)).save(out)
        except Exception as exc:
            print(f"Warning: could not save sidecar images: {exc}")

    print(f"Wrote {os.path.join(args.output_dir, 'results.csv')}")
