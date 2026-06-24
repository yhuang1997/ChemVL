# SPDX-License-Identifier: MIT
"""Build ImageMol dataloaders with merged hyperparameters."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.image_dataloader import ImageDataset
from utils.external.imagemol.imagemol_transforms import build_imagemol_transform_triplet


def build_imagemol_dataloaders(
    cfg: Dict[str, Any],
    names: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    im_cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = int(im_cfg.get("batch_size", cfg["training"].get("batch_size", 128)))

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int((cfg.get("basic") or {}).get("num_workers", 0))

    name_train, name_val, name_test = names[train_idx], names[val_idx], names[test_idx]
    labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]

    train_t, val_t, test_t = build_imagemol_transform_triplet(cfg)
    train_ds = ImageDataset(name_train, labels_train, img_transformer=train_t, normalize=None)
    val_ds = ImageDataset(name_val, labels_val, img_transformer=val_t, normalize=None)
    test_ds = ImageDataset(name_test, labels_test, img_transformer=test_t, normalize=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
