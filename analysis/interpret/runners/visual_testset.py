"""Test-split Grad-CAM gallery for MoleculeNet finetune checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analysis.interpret.presets_loader import resolve_data_path
from analysis.interpret.runners.render_gallery import write_manifest
from analysis.interpret.runners.visual_case import _run_panel_for_cases, load_shared_panel_models
from utils.argparser import load_config
from utils.finetune_utils import get_datafile, get_split
from utils.train_utils import load_smiles


def _sample_test_cases(
    preset: Dict[str, Any],
    *,
    max_molecules: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    ckpt_dir = resolve_data_path(preset.get("ckpt_dir"))
    if ckpt_dir is None or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing ckpt_dir: {ckpt_dir}")
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json under {ckpt_dir}")

    cfg = load_config([str(cfg_path)])
    dataset = str(preset.get("dataset") or cfg["dataset"]["dataset"])
    split_name = str(preset.get("split", "test"))
    max_n = int(max_molecules if max_molecules is not None else preset.get("max_molecules", 12))
    rng_seed = int(seed if seed is not None else preset.get("seed", 123))

    image_folder, txt_file = get_datafile(cfg)
    smiles = np.array(load_smiles(txt_file))
    from dataloader.image_dataloader import load_filenames_and_labels_multitask

    names, labels = load_filenames_and_labels_multitask(image_folder, txt_file, task_type="classification")
    names = np.array(names)
    labels = np.array(labels)

    train_idx, val_idx, test_idx = get_split(cfg, names, labels, smiles)
    split_map = {"train": train_idx, "valid": val_idx, "val": val_idx, "test": test_idx}
    if split_name not in split_map:
        raise ValueError(f"Unknown split {split_name!r}; expected train, valid, or test")
    indices = list(split_map[split_name])

    molecule_ids = preset.get("molecule_ids")
    if molecule_ids:
        selected = [int(i) for i in molecule_ids if int(i) in indices]
    else:
        rng = random.Random(rng_seed)
        pool = list(indices)
        rng.shuffle(pool)
        selected = pool[:max_n]

    cases: List[Dict[str, object]] = []
    for idx in selected:
        cases.append(
            {
                "name": f"mol_{idx:05d}",
                "dataset": dataset,
                "smiles": str(smiles[idx]),
                "label": int(labels[idx][0] if labels[idx].ndim else labels[idx]),
            }
        )
    return cases


def run_testset_gallery(
    preset: Dict[str, Any],
    output_dir: Path,
    *,
    max_molecules: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Path]:
    from PIL import Image

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    benchmark = str(preset.get("benchmark", "moleculenet"))
    if benchmark != "moleculenet":
        raise ValueError(f"testset_gallery preset must set benchmark: moleculenet (got {benchmark!r})")

    ckpt_dir = resolve_data_path(preset.get("ckpt_dir"))
    if ckpt_dir is None:
        raise ValueError("preset.ckpt_dir is required")
    cfg_path = ckpt_dir / "config.json"
    ckpt_path = ckpt_dir / "ckpt.pth"
    for path in (cfg_path, ckpt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    cases = _sample_test_cases(preset, max_molecules=max_molecules, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_models, finetuned_model = load_shared_panel_models(
        preset,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        device=device,
    )

    saved: List[Path] = []
    manifest: List[Dict[str, object]] = []
    try:
        for case in cases:
            panel = _run_panel_for_cases(
                [case],
                cfg_path=cfg_path,
                ckpt_path=ckpt_path,
                preset=preset,
                device=device,
                pretrained_models=pretrained_models,
                finetuned_model=finetuned_model,
            )
            out_path = output_dir / f"{case['name']}_gradcam_panel.png"
            Image.fromarray(panel).save(out_path)
            saved.append(out_path)
            manifest.append({"case": case["name"], "smiles": case["smiles"], "image": out_path.name})
            del panel
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        pretrained_models.clear()
        finetuned_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(output_dir, manifest)
    return saved
