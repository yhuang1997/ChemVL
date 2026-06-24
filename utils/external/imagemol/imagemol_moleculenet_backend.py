# SPDX-License-Identifier: MIT
"""
ImageMol ResNet18 backend for MoleculeNet inside ChemVL (M3).

- RN18* backbone load (first 120 keys)
- Single ``fc(num_tasks)`` + BCEWithLogitsLoss (classification)
- Official train aug: CenterCrop + Grayscale / Rotation / Flip
- Eval: CenterCrop only, no TTA
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from utils.external.chemvl_external_backend import FinetuneBackend
from utils.external.imagemol.imagemol_checkpoint import load_imagemol_pretrained_backbone, resolve_imagemol_checkpoint
from utils.external.imagemol.imagemol_dataloader import build_imagemol_dataloaders
from utils.external.imagemol.imagemol_eval import evaluate_imagemol_predictions
from utils.external.imagemol.imagemol_external_config import load_merged_imagemol_config
from utils.external.imagemol.imagemol_model import ImageMolFinetuneModel


class ImageMolMoleculeNetBackend(FinetuneBackend):
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        names: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        smiles: list[str] | None = None,
    ):
        del smiles
        self._cfg = cfg
        self._device = device
        self._im_cfg = load_merged_imagemol_config(cfg)
        self._task_type = self._im_cfg["task_type"]
        self._classification = self._task_type == "classification"
        self._num_tasks = int(self._im_cfg["num_tasks"])

        labels = np.asarray(labels, dtype=np.float32)
        self._train_loader, self._val_loader, self._test_loader = build_imagemol_dataloaders(
            cfg, names, labels, train_idx, val_idx, test_idx, self._im_cfg
        )

        self._model = ImageMolFinetuneModel(self._num_tasks, self._task_type).to(device)

        resume = resolve_imagemol_checkpoint(self._im_cfg.get("resume"))
        if resume:
            load_imagemol_pretrained_backbone(
                self._model, resume, resume_key=str(self._im_cfg.get("resume_key", "state_dict"))
            )

        lr = float(self._im_cfg.get("lr", 0.01))
        momentum = float(self._im_cfg.get("momentum", 0.9))
        weight_decay = float(self._im_cfg.get("weight_decay", 1e-5))
        self._optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self._scheduler = None

        if self._classification:
            self._criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self._criterion = nn.MSELoss()

    @property
    def num_training_epochs(self) -> int:
        return int(self._im_cfg.get("epochs", 100))

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    def scheduler_step_after_epoch(self, epoch: int) -> None:
        del epoch

    def _compute_loss(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(self._device)
        if self._classification:
            target = labels.float()
            if target.dim() == 1:
                target = target.view(-1, 1)
            if pred.dim() == 1:
                pred = pred.view(-1, 1)
            valid = target != -1
            if valid.sum() == 0:
                return pred.sum() * 0.0
            loss_mat = self._criterion(pred, target)
            loss_mat = torch.where(valid, loss_mat, torch.zeros_like(loss_mat))
            return loss_mat.sum() / valid.sum().clamp_min(1.0)

        if pred.dim() == 1:
            pred = pred.view(-1, 1)
        return self._criterion(pred, labels.float())

    def train_epoch(self, epoch: int) -> float:
        del epoch
        self._model.train()
        losses = []
        for batch in self._train_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch
            images = images.to(self._device)
            self._optimizer.zero_grad()
            pred = self._model(images)
            loss = self._compute_loss(pred, labels)
            loss.backward()
            self._optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def _collect(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self._model.eval()
        ys, ps = [], []
        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch
            images = images.to(self._device)
            logits = self._model(images)
            ys.append(labels.numpy())
            ps.append(logits.cpu().numpy())
        y_true = np.concatenate(ys, axis=0)
        y_logits = np.concatenate(ps, axis=0)
        return y_true, y_logits

    def evaluate(self, split: str) -> Dict[str, float]:
        loader = {"train": self._train_loader, "val": self._val_loader, "test": self._test_loader}.get(split)
        if loader is None:
            raise ValueError(split)
        y_true, y_logits = self._collect(loader)
        return evaluate_imagemol_predictions(
            y_true,
            y_logits,
            task_type=self._task_type,
            num_tasks=self._num_tasks,
        )
