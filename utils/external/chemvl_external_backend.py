# SPDX-License-Identifier: MIT
"""ChemVL external fine-tune backends: protocol + factory."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class FinetuneBackend(Protocol):
    """Swappable train/eval for ``utils/external/finetune_external.py``."""

    def train_epoch(self, epoch: int) -> float:
        """Return scalar training loss for logging (e.g. mean batch MSE)."""

    def evaluate(self, split: str) -> Dict[str, float]:
        """Return metrics compatible with ``models.evaluate`` keys (e.g. ``ROCAUC``, ``RMSE``)."""

    @property
    def model(self) -> torch.nn.Module:
        ...

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        ...

    @property
    def scheduler(self) -> Any:
        """LR scheduler or ``None``."""

    def scheduler_step_after_epoch(self, epoch: int) -> None:
        """Match MolMCL ``finetune.py`` post-epoch scheduler stepping."""


def build_finetune_backend(
    name: str,
    cfg: Dict[str, Any],
    device: torch.device,
    smiles: list[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    names: Optional[np.ndarray] = None,
) -> FinetuneBackend:
    if name == "molmcl_moleculeace":
        from utils.external.molmcl.molmcl_moleculeace_backend import MolMCLMoleculeACEBackend

        return MolMCLMoleculeACEBackend(cfg, device, smiles, labels, train_idx, val_idx, test_idx)

    if name == "molmcl_moleculenet":
        from utils.external.molmcl.molmcl_moleculenet_backend import MolMCLMoleculeNetBackend

        return MolMCLMoleculeNetBackend(cfg, device, smiles, labels, train_idx, val_idx, test_idx)

    if name == "molclr_moleculenet":
        from utils.external.molclr.molclr_moleculenet_backend import MolCLRMoleculeNetBackend

        return MolCLRMoleculeNetBackend(cfg, device, smiles, labels, train_idx, val_idx, test_idx)

    if name == "imagemol_moleculenet":
        from utils.external.imagemol.imagemol_moleculenet_backend import ImageMolMoleculeNetBackend

        if names is None:
            raise ValueError("imagemol_moleculenet backend requires ``names`` (image indices).")
        return ImageMolMoleculeNetBackend(
            cfg, device, names, labels, train_idx, val_idx, test_idx, smiles=smiles
        )

    raise ValueError(f"Unknown finetune backend: {name!r}")
