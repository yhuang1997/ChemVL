"""Torch datasets / loaders for descriptor-only finetuning (no images / graphs)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DescriptorFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        if task_type == "classification":
            self.y = torch.as_tensor(y, dtype=torch.long)
        else:
            self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DescriptorSmilesDataset(Dataset):
    def __init__(self, smiles: Sequence[str], y: np.ndarray, task_type: str):
        self.smiles = list(smiles)
        if task_type == "classification":
            self.y = torch.as_tensor(y, dtype=torch.long)
        else:
            self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.smiles[idx], self.y[idx]


def _collate_smiles(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    sms = [b[0] for b in batch]
    y = torch.stack([b[1] for b in batch], dim=0)
    return sms, y


def build_descriptor_only_dataloaders(
    cfg: Dict[str, Any],
    mode: str,
    X_scaled: np.ndarray | None,
    smiles: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["basic"]["num_workers"])
    task_type = cfg["dataset"]["task_type"]
    mode = mode.lower()

    if mode == "feature":
        if X_scaled is None:
            raise ValueError("feature mode requires X_scaled")
        X_tr, y_tr = X_scaled[train_idx], labels[train_idx]
        X_va, y_va = X_scaled[val_idx], labels[val_idx]
        X_te, y_te = X_scaled[test_idx], labels[test_idx]
        train_ds = DescriptorFeatureDataset(X_tr, y_tr, task_type)
        val_ds = DescriptorFeatureDataset(X_va, y_va, task_type)
        test_ds = DescriptorFeatureDataset(X_te, y_te, task_type)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        )

    if mode == "text":
        sm = np.asarray(smiles, dtype=object)
        y_tr, y_va, y_te = labels[train_idx], labels[val_idx], labels[test_idx]
        train_ds = DescriptorSmilesDataset(sm[train_idx], y_tr, task_type)
        val_ds = DescriptorSmilesDataset(sm[val_idx], y_va, task_type)
        test_ds = DescriptorSmilesDataset(sm[test_idx], y_te, task_type)
        kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=_collate_smiles)
        return (
            DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
        )

    raise ValueError(f"Unknown descriptor_only_mode: {mode!r}")
