# SPDX-License-Identifier: MIT
"""Single-target MolCLR GIN/GCN finetune loop (one column / one regression task)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from models.evaluate import metric, metric_reg
from utils.external.molclr.molclr_external_config import load_merged_molclr_config, molclr_root_from_cfg
from utils.external.molclr.molclr_graph import attach_labels_to_graph, build_molclr_graph_from_smiles, molclr_graph_collate
from utils.external.molclr.molclr_model_loader import get_molclr_gcn_class, get_molclr_ginet_class
from utils.external.molclr.molclr_normalizer import MolCLRLabelNormalizer


class _MolCLRGraphDataset(Dataset):
    def __init__(self, smiles: list[str], labels: np.ndarray):
        self.smiles = smiles
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        graph = build_molclr_graph_from_smiles(self.smiles[idx])
        return attach_labels_to_graph(graph, float(self.labels[idx]))


def _load_molclr_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    if hasattr(model, "load_my_state_dict"):
        model.load_my_state_dict(state)
    else:
        model.load_state_dict(state, strict=False)
    print(f"Loaded MolCLR checkpoint from {path}")


def _build_molclr_model(cfg: Dict[str, Any], mol_cfg: Dict[str, Any], classification: bool) -> nn.Module:
    molclr_root = molclr_root_from_cfg(cfg)
    model_cfg = dict(mol_cfg.get("model") or {})
    task = "classification" if classification else "regression"
    if str(mol_cfg.get("model_type", "gin")) == "gcn":
        GCN = get_molclr_gcn_class(molclr_root)
        return GCN(task, **model_cfg)
    GINet = get_molclr_ginet_class(molclr_root)
    return GINet(task, **model_cfg)


def _molclr_adam(model: nn.Module, mol_cfg: Dict[str, Any]) -> torch.optim.Adam:
    pred_params, base_params = [], []
    for name, param in model.named_parameters():
        if "pred_head" in name:
            pred_params.append(param)
        else:
            base_params.append(param)
    init_lr = float(mol_cfg.get("init_lr", 5e-4))
    init_base_lr = float(mol_cfg.get("init_base_lr", 1e-4))
    weight_decay = float(mol_cfg.get("weight_decay", 1e-6))
    return torch.optim.Adam(
        [
            {"params": base_params, "lr": init_base_lr},
            {"params": pred_params, "lr": init_lr},
        ],
        weight_decay=weight_decay,
    )


class MolCLRTaskRunner:
    """One MolCLR model for a single classification/regression target."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        smiles: list[str],
        labels_1d: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        task_name: str = "",
    ):
        self._cfg = cfg
        self._device = device
        self._mol_cfg = load_merged_molclr_config(cfg)
        self._classification = self._mol_cfg["task_type"] == "classification"
        self._dataset_name = str(self._mol_cfg.get("dataset_name", "")).lower()
        self._task_name = task_name or self._dataset_name

        labels_1d = np.asarray(labels_1d, dtype=np.float32).reshape(-1)
        full_ds = _MolCLRGraphDataset(smiles, labels_1d)
        batch_size = int(self._mol_cfg.get("batch_size", 32))
        nw = int((cfg.get("basic") or {}).get("num_workers", 0))

        self._train_loader = PyGDataLoader(
            Subset(full_ds, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=molclr_graph_collate,
        )
        self._val_loader = PyGDataLoader(
            Subset(full_ds, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=molclr_graph_collate,
        )
        self._test_loader = PyGDataLoader(
            Subset(full_ds, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=molclr_graph_collate,
        )

        self._model = _build_molclr_model(cfg, self._mol_cfg, self._classification).to(device)
        resume = self._mol_cfg.get("resume")
        if resume:
            _load_molclr_checkpoint(self._model, str(resume), device)

        self._optimizer = _molclr_adam(self._model, self._mol_cfg)
        self._scheduler = None
        self._normalizer: Optional[MolCLRLabelNormalizer] = None

        if not self._classification and self._dataset_name in ("qm7", "qm9"):
            train_vals = labels_1d[train_idx]
            self._normalizer = MolCLRLabelNormalizer(torch.tensor(train_vals, dtype=torch.float32))

        if self._classification:
            self._criterion = nn.CrossEntropyLoss()
        elif self._dataset_name in ("qm7", "qm8", "qm9"):
            self._criterion = nn.L1Loss()
        else:
            self._criterion = nn.MSELoss()

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    def _step_loss(self, batch) -> torch.Tensor:
        batch = batch.to(self._device)
        _, pred = self._model(batch)
        y = batch.y.view(-1)

        if self._classification:
            target = y.long()
            mask = target >= 0
            if mask.sum() == 0:
                return pred.sum() * 0.0
            return self._criterion(pred[mask], target[mask])

        pred = pred.view(-1)
        target = y.float()
        if self._normalizer is not None:
            target = self._normalizer.norm(target)
        return self._criterion(pred, target)

    def train_epoch(self, epoch: int) -> float:
        del epoch
        self._model.train()
        losses = []
        for batch in self._train_loader:
            self._optimizer.zero_grad()
            loss = self._step_loss(batch)
            loss.backward()
            self._optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _collect_preds(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self._model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                _, pred = self._model(batch)
                y = batch.y.view(-1)
                if self._classification:
                    prob = F.softmax(pred, dim=-1)[:, 1]
                    ys.append(y.cpu().numpy())
                    ps.append(prob.cpu().numpy())
                else:
                    p = pred.view(-1)
                    if self._normalizer is not None:
                        p = self._normalizer.denorm(p)
                    ys.append(y.cpu().numpy())
                    ps.append(p.cpu().numpy())
        return np.concatenate(ys), np.concatenate(ps)

    def evaluate(self, split: str) -> Dict[str, float]:
        loader = {"train": self._train_loader, "val": self._val_loader, "test": self._test_loader}.get(split)
        if loader is None:
            raise ValueError(split)

        y_true, y_scores = self._collect_preds(loader)
        if self._classification:
            y_t = y_true.astype(int)
            valid = y_t >= 0
            y_p = (y_scores[valid] >= 0.5).astype(int)
            return metric(y_t[valid], y_p, y_scores[valid], empty=-1)
        return metric_reg(y_true.ravel(), y_scores.ravel())
