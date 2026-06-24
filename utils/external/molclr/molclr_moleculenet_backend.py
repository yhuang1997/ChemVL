# SPDX-License-Identifier: MIT
"""
MolCLR GIN/GCN backend for MoleculeNet inside ChemVL.

- AddHs graph construction (``molclr_graph``)
- Dual-LR Adam (``config_finetune.yaml``)
- Classification with ``num_tasks > 1``: **per-target** independent models (MolCLR ``finetune.py``)
- QM7/qm9: train-set z-score + L1Loss + denorm on eval
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from utils.external.chemvl_external_backend import FinetuneBackend
from utils.external.molclr.molclr_external_config import load_merged_molclr_config
from utils.external.molclr.molclr_labels import prepare_molclr_classification_labels, task_column_labels
from utils.external.molclr.molclr_task_runner import MolCLRTaskRunner


def _mean_rocauc(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    vals = [float(m["ROCAUC"]) for m in metrics if m.get("ROCAUC") is not None and not np.isnan(m["ROCAUC"])]
    if not vals:
        return {"ROCAUC": float("nan")}
    return {"ROCAUC": float(np.mean(vals))}


class MolCLRMoleculeNetBackend(FinetuneBackend):
    """MolCLR-style MoleculeNet finetune; ChemVL split indices supplied by ``finetune_external``."""

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
        self._cfg = cfg
        self._device = device
        self._mol_cfg = load_merged_molclr_config(cfg)
        self._classification = self._mol_cfg["task_type"] == "classification"
        labels = np.asarray(labels, dtype=np.float32)
        if self._classification:
            labels = prepare_molclr_classification_labels(labels)

        num_tasks = int(labels.shape[1]) if labels.ndim == 2 else 1
        self._num_tasks = num_tasks
        per_target_cls = self._classification and num_tasks > 1

        self._runners: List[MolCLRTaskRunner] = []
        if per_target_cls:
            for t in range(num_tasks):
                col = task_column_labels(labels, t)
                self._runners.append(
                    MolCLRTaskRunner(
                        cfg,
                        device,
                        smiles,
                        col,
                        train_idx,
                        val_idx,
                        test_idx,
                        task_name=f"{self._mol_cfg.get('dataset_name', '')}_task{t}",
                    )
                )
        else:
            col = task_column_labels(labels, 0)
            self._runners.append(
                MolCLRTaskRunner(cfg, device, smiles, col, train_idx, val_idx, test_idx)
            )

        self._per_target_cls = per_target_cls

    @property
    def num_training_epochs(self) -> int:
        return int(self._mol_cfg.get("epochs", 100))

    @property
    def model(self) -> torch.nn.Module:
        if len(self._runners) == 1:
            return self._runners[0].model
        return nn.ModuleList([r.model for r in self._runners])

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._runners[0].optimizer

    @property
    def scheduler(self) -> Any:
        return self._runners[0].scheduler

    def scheduler_step_after_epoch(self, epoch: int) -> None:
        del epoch

    def train_epoch(self, epoch: int) -> float:
        losses = [r.train_epoch(epoch) for r in self._runners]
        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, split: str) -> Dict[str, float]:
        results = [r.evaluate(split) for r in self._runners]
        if self._per_target_cls:
            out = _mean_rocauc(results)
            out["result_list_dict_each_task"] = results
            return out
        return results[0]
