# SPDX-License-Identifier: MIT
"""MolMCL ``GNNPredictor`` backend for MoleculeNet inside ChemVL (classification + regression, PyG)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from models.evaluate import metric, metric_reg, metric_reg_multitask
from utils.external.chemvl_external_backend import FinetuneBackend
from utils.external.molclr.molclr_normalizer import MolCLRLabelNormalizer
from utils.external.molmcl.moleculeace_pyg import MoleculeAceSmilesGraphDataset
from utils.external.molmcl.molmcl_moleculeace_backend import _optimize_prompt_weight_from_loaders
from utils.external.molmcl.molmcl_optimizer_shim import get_optimizer
from utils.external.molmcl.molmcl_external_config import load_merged_molmcl_config, molmcl_root_from_cfg


def _multitask_classification_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """Mean ROCAUC + per-task metrics (MolMCL label mask: 0 = missing, +/-1 = classes)."""
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_scores, -50.0, 50.0)))
    result_list: list[Any] = []
    rocs: list[float] = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if not (np.sum(col == 1) > 0 and np.sum(col == -1) > 0):
            result_list.append(None)
            continue
        is_valid = col**2 > 0
        if np.sum(is_valid) == 0:
            result_list.append(None)
            continue
        y_t = ((col[is_valid] + 1.0) / 2.0).astype(int)
        y_pr = y_prob[is_valid, i]
        y_p = (y_pr >= 0.5).astype(int)
        task_metrics = metric(y_t, y_p, y_pr, empty=-1)
        result_list.append(task_metrics)
        roc = task_metrics.get("ROCAUC")
        if roc is not None and not np.isnan(roc):
            rocs.append(float(roc))
    mean_roc = float(np.mean(rocs)) if rocs else float("nan")
    return {"ROCAUC": mean_roc, "result_list_dict_each_task": result_list}


class MolMCLMoleculeNetBackend(FinetuneBackend):
    """MolMCL-style MoleculeNet finetune (scaffold split in ``finetune_external``; yaml hyperparameters)."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        smiles: list[str],
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ):
        molmcl_root = molmcl_root_from_cfg(cfg)
        if molmcl_root not in sys.path:
            sys.path.insert(0, molmcl_root)

        self._molmcl_root = molmcl_root
        self._device = device
        self._mol_cfg = load_merged_molmcl_config(cfg)
        self._mol_cfg["device"] = "cuda" if device.type == "cuda" else "cpu"

        raw_task = str(self._mol_cfg.get("dataset", {}).get("task", "regression")).lower()
        self._classification = raw_task == "classification"

        feat_type = self._mol_cfg["dataset"]["feat_type"]
        full_ds = MoleculeAceSmilesGraphDataset(smiles, labels, feat_type=feat_type)

        nw = int(self._mol_cfg["dataset"]["num_workers"])
        self._train_loader = PyGDataLoader(
            Subset(full_ds, train_idx.tolist()),
            batch_size=self._mol_cfg["batch_size"],
            shuffle=True,
            num_workers=nw,
        )
        self._val_loader = PyGDataLoader(
            Subset(full_ds, val_idx.tolist()),
            batch_size=self._mol_cfg["batch_size"],
            shuffle=False,
            num_workers=nw,
        )
        self._test_loader = PyGDataLoader(
            Subset(full_ds, test_idx.tolist()),
            batch_size=self._mol_cfg["batch_size"],
            shuffle=False,
            num_workers=nw,
        )

        ft = feat_type
        if ft == "basic":
            atom_feat_dim, bond_feat_dim = None, None
        elif ft == "rich":
            atom_feat_dim, bond_feat_dim = 143, 14
        elif ft == "super_rich":
            atom_feat_dim, bond_feat_dim = 170, 14
        else:
            raise ValueError(ft)

        from molmcl.finetune.model import GNNPredictor

        mcfg = self._mol_cfg["model"]
        ckpt = mcfg.get("checkpoint")
        use_prompt = bool(mcfg.get("use_prompt"))
        if ckpt and not os.path.isfile(str(ckpt)):
            print(f"Warning: MolMCL checkpoint not found at {ckpt!r}; training without pretrained weights.")
            ckpt = None
            use_prompt = False
        if not ckpt:
            use_prompt = False
        self._mol_cfg["model"]["use_prompt"] = use_prompt
        self._mol_cfg["model"]["checkpoint"] = ckpt

        self._model = GNNPredictor(
            num_layer=mcfg["num_layer"],
            emb_dim=mcfg["emb_dim"],
            num_tasks=full_ds.num_task,
            normalize=mcfg["normalize"],
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            drop_ratio=mcfg["dropout_ratio"],
            attn_drop_ratio=mcfg["attn_dropout_ratio"],
            temperature=mcfg["temperature"],
            use_prompt=use_prompt,
            model_head=mcfg["heads"],
            layer_norm_out=mcfg["layernorm"],
            backbone=mcfg["backbone"],
        )

        if ckpt:
            print(f"Loading MolMCL checkpoint from {ckpt}")
            try:
                state = torch.load(str(ckpt), map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(str(ckpt), map_location=device)
            self._model.load_state_dict(state["wrapper"], strict=False)

        self._model = self._model.to(device)

        if self._mol_cfg["model"]["use_prompt"]:
            inits = self._mol_cfg.get("prompt_optim", {}).get("inits")
            best_init: Optional[torch.Tensor] = torch.Tensor(inits) if inits else None
            if best_init is None:
                try:
                    best_init = _optimize_prompt_weight_from_loaders(
                        self._model,
                        self._train_loader,
                        self._val_loader,
                        self._mol_cfg,
                        self._device,
                    )
                except ImportError as e:
                    raise ImportError(
                        "MolMCL use_prompt=True requires ``molmcl.finetune.prompt_optim``. "
                        "Install deps, set prompt_optim.inits, or disable checkpoint / use_prompt."
                    ) from e
            self._model.set_prompt_weight(best_init.to(device))
            if self._mol_cfg["verbose"]:
                print("Initial prompt prob:", self._model.get_prompt_weight("softmax").data.cpu())

        if getattr(self._model, "use_prompt", False) and hasattr(self._model, "aggrs"):
            self._model.freeze_aggr_module()

        self._optimizer = get_optimizer(self._model, self._mol_cfg["optim"])

        self._scheduler: Any = None
        sch = self._mol_cfg["optim"].get("scheduler")
        if sch == "cos_anneal":
            self._scheduler = CosineAnnealingLR(
                self._optimizer, T_max=self._mol_cfg["epochs"], eta_min=0.0001
            )
        elif sch == "poly_decay":
            from molmcl.utils.scheduler import PolynomialDecayLR

            n_batch = max(1, len(self._train_loader))
            self._scheduler = PolynomialDecayLR(
                self._optimizer,
                warmup_updates=self._mol_cfg["epochs"] * n_batch // 10,
                tot_updates=self._mol_cfg["epochs"] * n_batch,
                lr=float(self._mol_cfg["optim"]["finetune_lr"]),
                end_lr=1e-9,
                power=1,
            )

        self._data_name = str(self._mol_cfg["dataset"].get("data_name", "")).lower()
        self._normalizer: Optional[MolCLRLabelNormalizer] = None
        if not self._classification and self._data_name in ("qm7", "qm9"):
            train_vals = np.asarray(labels, dtype=np.float32).ravel()[train_idx]
            self._normalizer = MolCLRLabelNormalizer(
                torch.tensor(train_vals, dtype=torch.float32)
            )

        if self._classification:
            self._criterion = nn.BCEWithLogitsLoss(reduction="none")
        elif self._data_name in ("qm7", "qm8", "qm9"):
            self._criterion = nn.L1Loss(reduction="none")
        else:
            self._criterion = nn.MSELoss(reduction="none")

    @property
    def num_training_epochs(self) -> int:
        return int(self._mol_cfg["epochs"])

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
        if self._scheduler is None:
            return
        if self._mol_cfg["optim"].get("scheduler") == "cos_anneal":
            self._scheduler.step()

    def train_epoch(self, epoch: int) -> float:
        del epoch
        self._model.train()
        losses = []
        for batch in self._train_loader:
            batch = batch.to(self._device)
            out = self._model(batch)
            predict = out["predict"]
            label = batch.label.view(predict.shape)

            if self._classification:
                mask = label == 0
                loss = self._criterion(predict.double(), (label + 1.0) / 2.0) * (~mask)
                loss = loss.sum() / (~mask).sum().clamp_min(1.0)
            else:
                if self._normalizer is not None:
                    label = self._normalizer.norm(label)
                loss = self._criterion(predict, label).mean()

            self._optimizer.zero_grad()
            loss.backward()
            if float(self._mol_cfg["optim"]["gradient_clip"]) > 0:
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), float(self._mol_cfg["optim"]["gradient_clip"])
                )
            self._optimizer.step()

            if self._mol_cfg["optim"].get("scheduler") == "poly_decay" and self._scheduler is not None:
                self._scheduler.step()

            losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, split: str) -> Dict[str, float]:
        loader = {"train": self._train_loader, "val": self._val_loader, "test": self._test_loader}.get(split)
        if loader is None:
            raise ValueError(split)

        self._model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                predict = self._model(batch)["predict"]
                lab = batch.label.view(predict.shape)
                ys.append(lab.cpu().numpy())
                ps.append(predict.cpu().numpy())

        y_true = np.concatenate(ys, axis=0)
        y_scores = np.concatenate(ps, axis=0)
        if self._normalizer is not None:
            y_scores = (
                self._normalizer.denorm(torch.tensor(y_scores, dtype=torch.float32))
                .cpu()
                .numpy()
            )

        if self._classification:
            if y_true.shape[1] == 1:
                y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_scores.reshape(-1), -50.0, 50.0)))
                col = y_true.reshape(-1)
                is_valid = col**2 > 0
                y_t = ((col[is_valid] + 1.0) / 2.0).astype(int)
                y_pr = y_prob[is_valid]
                y_p = (y_pr >= 0.5).astype(int)
                return metric(y_t, y_p, y_pr, empty=-1)
            return _multitask_classification_metrics(y_true, y_scores)

        if y_true.shape[1] == 1:
            return metric_reg(y_true.ravel(), y_scores.ravel())

        return metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
