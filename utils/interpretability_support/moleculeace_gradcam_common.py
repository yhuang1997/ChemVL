"""Shared helpers for MoleculeACE 224px Grad-CAM (batch + highres)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.clip_model_utils import AdaptedCLIP, ExtendedCLIPVisual
from utils.argparser import load_config
from utils.finetune_utils import get_datafile
from utils.interpretability_support.highres_checkpoint_resolver import resolve_ckpt_from_log_dir
from utils.interpretability_support.visual_utils import load_finetuned_model

DEFAULT_DESCRIPTORS = [
    "NumAromaticRings",
    "fr_benzene",
    "NOCount",
    "fr_halogen",
]

RESOLUTION = 224


def validate_moleculeace_cfg(cfg: Dict[str, Any]) -> None:
    ds = cfg.get("dataset") or {}
    if ds.get("benchmark") != "moleculeace":
        raise ValueError("dataset.benchmark must be 'moleculeace'.")
    if (ds.get("protocol") or "").strip() != "MolMCL":
        raise ValueError('dataset.protocol must be "MolMCL".')
    if ds.get("task_type") != "regression":
        raise ValueError("Only dataset.task_type == 'regression' is supported.")


def membership_split(
    pool_index: int,
    train_set: set,
    val_set: set,
    test_set: set,
) -> str:
    if pool_index in train_set:
        return "train"
    if pool_index in val_set:
        return "val"
    if pool_index in test_set:
        return "test"
    raise ValueError(f"pool_index {pool_index} not in train/val/test splits")


def labels_for_split_indices(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """1D float targets for Grad-CAM RMSE targets (task 0 if multitask)."""
    y = np.asarray(labels[indices], dtype=np.float64)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y[:, 0]
    raise ValueError(f"Unexpected labels shape {labels.shape}")


def prepare_dataset_images(
    names: np.ndarray,
    indices: List[int],
) -> Tuple[List[np.ndarray], List[Image.Image], torch.Tensor]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Resize(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    pils: List[Image.Image] = []
    rgb_images: List[np.ndarray] = []
    for i in indices:
        path = names[i]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing image: {path}")
        im = Image.open(path).convert("RGB")
        pils.append(im)
        rgb_images.append((np.asarray(im).astype(np.float32) / 255.0))
    input_tensor = torch.stack([transform(im) for im in pils], dim=0)
    return rgb_images, pils, input_tensor


def load_finetuned_from_log_dir(
    log_dir: Path,
    ckpt: Path,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    cfg_path = log_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json under log_dir: {log_dir}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    cfg = load_config([str(cfg_path)])
    validate_moleculeace_cfg(cfg)
    _image_folder, txt_file = get_datafile(cfg)
    cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = txt_file

    fd, tmp_cfg_path = tempfile.mkstemp(suffix="_moleculeace_gradcam.json")
    os.close(fd)
    try:
        Path(tmp_cfg_path).write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        model = load_finetuned_model(tmp_cfg_path, str(ckpt), device=device, verbose=False)
    finally:
        try:
            Path(tmp_cfg_path).unlink(missing_ok=True)
        except OSError:
            pass

    if not isinstance(model, (AdaptedCLIP, ExtendedCLIPVisual)):
        raise TypeError(
            f"Unsupported model type {type(model).__name__}; "
            "expected AdaptedCLIP or ExtendedCLIPVisual."
        )
    if not hasattr(model, "image_encoder") or not hasattr(model.image_encoder, "layer4"):
        raise TypeError(f"Model has no image_encoder.layer4: {type(model).__name__}")

    return model, cfg, str(cfg_path.resolve())


def resolve_ckpt(log_dir: str, ckpt: Optional[str] = None) -> str:
    """Resolve checkpoint path under a finetuning log_dir."""
    return resolve_ckpt_from_log_dir(log_dir, ckpt)
