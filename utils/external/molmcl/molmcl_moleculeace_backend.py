# SPDX-License-Identifier: MIT
"""MolMCL ``GNNPredictor`` backend for MoleculeACE inside ChemVL (regression, PyG)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import to_dense_batch

from models.evaluate import metric_reg
from utils.external.chemvl_external_backend import FinetuneBackend
from utils.external.molmcl.moleculeace_pyg import MoleculeAceSmilesGraphDataset
from utils.external.molmcl.molmcl_optimizer_shim import get_optimizer
from utils.external.molmcl.molmcl_external_config import load_merged_molmcl_config, molmcl_root_from_cfg


def _optimize_prompt_weight_from_loaders(
    model: nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    mol_cfg: Dict[str, Any],
    device: torch.device,
    *,
    metric: str = "euclidean",
    max_num: int = 5000,
) -> torch.Tensor:
    """Mirror ``external/MolMCL/scripts/finetune.py::optimize_prompt_weight_ri`` (model + loaders → BO core)."""

    from molmcl.finetune.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_core

    temperature = float(mol_cfg["model"]["temperature"])
    skip_bo = bool((mol_cfg.get("prompt_optim") or {}).get("skip_bo", False))
    dev_str = "cuda" if device.type == "cuda" else "cpu"

    num = 0
    model.eval()
    graph_rep_list, label_list = [], []
    for loader in (train_loader, val_loader):
        if loader is None:
            continue
        for batch in loader:
            batch = batch.to(dev_str)
            with torch.no_grad():
                graph_reps = []
                if model.backbone == "gps":
                    h_g, node_repres = model.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    h_g, node_repres = model.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                batch_x, batch_mask = to_dense_batch(node_repres, batch.batch)

                for i in range(len(model.prompt_token)):
                    h_g, h_x, _ = model.aggrs[i](batch_x, batch_mask)
                    if mol_cfg["model"]["normalize"]:
                        h_g = F.normalize(h_g, dim=-1)
                    graph_reps.append(h_g)

            graph_reps_batch = torch.stack(graph_reps)
            labels_batch = batch.label.view(-1, model.num_tasks)

            is_valid = (labels_batch != 0).sum(-1) == labels_batch.size(1)
            graph_rep_list.append(graph_reps_batch[:, is_valid])
            label_list.append(labels_batch[is_valid])

            num += graph_rep_list[-1].size(1)
            if num > max_num:
                break

    graph_reps = torch.concat(graph_rep_list, dim=1).cpu()
    labels = torch.concat(label_list, dim=0).cpu()

    return optimize_prompt_weight_ri_core(
        graph_reps,
        labels,
        n_runs=50,
        n_inits=50,
        n_points=5,
        n_restarts=512,
        n_samples=512,
        temperature=temperature,
        metric=metric,
        skip_bo=skip_bo,
        verbose=mol_cfg.get("verbose", False),
    )


class MolMCLMoleculeACEBackend(FinetuneBackend):
    """Official-style MolMCL MoleculeACE finetune (MSE, optional prompt init, frozen aggr)."""

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
        if (cfg.get("dataset") or {}).get("task_type") != "regression":
            raise ValueError("MolMCLMoleculeACEBackend currently supports dataset.task_type == 'regression'.")

        molmcl_root = molmcl_root_from_cfg(cfg)
        if molmcl_root not in sys.path:
            sys.path.insert(0, molmcl_root)

        self._molmcl_root = molmcl_root
        self._device = device
        self._mol_cfg = load_merged_molmcl_config(cfg)
        self._mol_cfg["device"] = "cuda" if device.type == "cuda" else "cpu"

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
                        "MolMCL use_prompt=True requires ``molmcl.finetune.prompt_optim`` (e.g. botorch). "
                        "Install those deps, set prompt_optim.inits in yaml, or disable the checkpoint / use_prompt."
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

        self._criterion = nn.MSELoss(reduction="none")

    @property
    def num_training_epochs(self) -> int:
        """Epoch count from merged MolMCL yaml (not ChemVL ``training.epochs``)."""
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

        y_true = np.concatenate(ys, axis=0).ravel()
        y_scores = np.concatenate(ps, axis=0).ravel()
        return metric_reg(y_true, y_scores)
