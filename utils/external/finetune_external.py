# SPDX-License-Identifier: MIT
"""
External fine-tune entry: **MoleculeACE** and **MoleculeNet** with swappable backends.

Supported MoleculeNet backends: ``molmcl_moleculenet``, ``molclr_moleculenet``, ``imagemol_moleculenet``.

ChemVL CLIP entry points ``moleculeace_finetune.py`` / ``extensive_finetune.py`` are unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from utils.finetune_utils import (
    extract_per_task_test_metrics,
    get_datafile,
    prepare_log_dir,
    save_run_config,
    get_metric,
    get_metric_moleculeace,
    get_split,
    get_split_moleculeace,
    multitask_column_names,
    prefix_add_formatted_time,
)
from utils.external.molmcl.moleculeace_tabular import (
    get_moleculeace_processed_ac_csv,
    load_moleculeace_tabular_multitask,
)
from utils.external.molmcl.molmcl_external_config import sync_chemvl_task_type_from_molmcl_yaml
from utils.external.molmcl.moleculenet_io import load_moleculenet_smiles_labels, resolve_moleculenet_csv
from utils.argparser import parse_args
from utils.plot_utils import plot_loss_rocauc
from utils.public_utils import setup_device, is_left_better_right
from utils.pl_init_utils import ensure_training_seeds
from utils.train_utils import apply_multi_view_train_batch_size_override, fix_train_random_seed, load_smiles
from utils.external.chemvl_external_backend import build_finetune_backend
from utils.external.external_effective_hparams import snapshot_external_effective_hparams

_MOLECULENET_BACKENDS = frozenset({"molmcl_moleculenet", "molclr_moleculenet", "imagemol_moleculenet"})


def _save_backend_ckpt(
    log_dir: str,
    filename_pre: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Any,
    result_dict: Dict[str, Any],
) -> None:
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if scheduler is None else scheduler.state_dict()
    state = {
        "epoch": epoch,
        "model_state_dict": model_cpu,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler,
        "result_dict": result_dict,
    }
    path = os.path.join(log_dir, f"{filename_pre}.pth")
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def main(cfg: Dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["basic"]["gpu"]
    device, device_ids = setup_device(cfg["basic"]["ngpu"])
    if len(device_ids) > 1:
        print("Note: external MolMCL backend uses a single process device; DataParallel is not applied.")

    ensure_training_seeds(cfg)
    fix_train_random_seed(cfg["training"]["runseed"])
    apply_multi_view_train_batch_size_override(cfg)

    cfg["basic"]["prefix"], cfg["basic"]["timestamp"] = prefix_add_formatted_time()
    log_dir = prepare_log_dir(cfg, mkdir=True)

    benchmark = str(cfg.get("dataset", {}).get("benchmark", "moleculeace")).lower()
    backend_name = (cfg.get("model") or {}).get("finetune_backend", "molmcl_moleculeace")

    if benchmark == "moleculeace":
        # ``protocol`` only applies to MoleculeACE (MolMCL activity-cliff vs other); not ChemVL scaffold keys.
        protocol = (cfg["dataset"].get("protocol") or "MolMCL").strip()
        if protocol != "MolMCL":
            raise NotImplementedError("finetune_external MoleculeACE path requires dataset.protocol == 'MolMCL'.")
        sync_chemvl_task_type_from_molmcl_yaml(cfg)
        task_type = cfg["dataset"]["task_type"]
        if task_type != "regression":
            raise NotImplementedError("finetune_external MoleculeACE MolMCL path supports regression only.")
        if backend_name != "molmcl_moleculeace":
            raise ValueError(f"Unsupported model.finetune_backend for MoleculeACE: {backend_name!r}")

        txt_file = get_moleculeace_processed_ac_csv(cfg)
        cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = txt_file

        names, labels = load_moleculeace_tabular_multitask(txt_file, task_type=task_type)
        names, labels = np.array(names), np.array(labels)
        smiles_list = load_smiles(txt_file)

        train_idx, val_idx, test_idx = get_split_moleculeace(cfg, names, labels, smiles_list)

    elif benchmark == "moleculenet":
        if backend_name not in _MOLECULENET_BACKENDS:
            raise ValueError(
                f"Unsupported model.finetune_backend for MoleculeNet: {backend_name!r} "
                f"(expected one of {sorted(_MOLECULENET_BACKENDS)!r})"
            )
        if backend_name == "molmcl_moleculenet":
            sync_chemvl_task_type_from_molmcl_yaml(cfg)
        task_type = cfg["dataset"]["task_type"]
        names_arr: np.ndarray | None = None
        if backend_name == "imagemol_moleculenet":
            from dataloader.image_dataloader import load_filenames_and_labels_multitask

            image_folder, txt_file = get_datafile(cfg)
            cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = txt_file
            names_arr, labels = load_filenames_and_labels_multitask(
                image_folder, txt_file, task_type=task_type
            )
            labels = np.asarray(labels, dtype=np.float32)
            names_arr = np.asarray(names_arr, dtype=object)
            smiles_list = load_smiles(txt_file)
            split_names = names_arr
        else:
            cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = resolve_moleculenet_csv(cfg)
            smiles_list, labels = load_moleculenet_smiles_labels(cfg)
            labels = np.asarray(labels, dtype=np.float32)
            split_names = np.array([str(i) for i in range(len(smiles_list))], dtype=object)

        train_idx, val_idx, test_idx = get_split(cfg, split_names, labels, smiles_list)
        train_idx, val_idx, test_idx = np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
        smiles_list = list(smiles_list)

    else:
        raise NotImplementedError(f"Unsupported dataset.benchmark: {benchmark!r}")

    backend = build_finetune_backend(
        backend_name,
        cfg,
        device,
        smiles_list,
        labels,
        train_idx,
        val_idx,
        test_idx,
        names=names_arr,
    )

    cfg["external_effective_hparams"] = snapshot_external_effective_hparams(cfg, backend_name)
    save_run_config(cfg, log_dir)

    num_epochs = getattr(backend, "num_training_epochs", None)
    if num_epochs is None:
        num_epochs = int((cfg.get("training") or {}).get("epochs", 1))
    start_epoch = int((cfg.get("training") or {}).get("start_epoch", 0))

    if benchmark == "moleculeace":
        eval_metric, valid_select, min_value, _criterion, _weights = get_metric_moleculeace(cfg, labels[train_idx])
    else:
        eval_metric, valid_select, min_value, _criterion, _weights = get_metric(cfg, labels[train_idx])
    metric_key = eval_metric.upper()
    labels_arr = np.asarray(labels)
    num_tasks = int(labels_arr.shape[1]) if labels_arr.ndim == 2 else 1
    task_names = multitask_column_names(cfg, num_tasks) if num_tasks > 1 else []

    loss_list = []
    valid_metric_list = []
    test_metric_list = []
    history_rows = []
    history_csv_path = os.path.join(log_dir, "train_val_test_history.csv")

    results: Dict[str, Any] = {
        "best_valid": min_value,
        "best_valid_epoch": 0,
        "best_train_loss": np.inf,
        "best_train_epoch": 0,
    }

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_step_loss = backend.train_epoch(epoch)

        val_results = backend.evaluate("val")
        test_results = backend.evaluate("test")

        backend.scheduler_step_after_epoch(epoch)

        loss_list.append(train_step_loss)
        valid_result = float(val_results[metric_key])
        test_result = float(test_results[metric_key])
        valid_metric_list.append(valid_result)
        test_metric_list.append(test_result)

        history_rows.append(
            {
                "epoch": epoch,
                "train_step_loss": train_step_loss,
                f"valid_{eval_metric}": valid_result,
                f"test_{eval_metric}": test_result,
            }
        )

        print(
            {
                "Epoch": epoch,
                "train_step_loss": train_step_loss,
                f"valid_{eval_metric}": valid_result,
                f"test_{eval_metric}": test_result,
            }
        )

        if is_left_better_right(train_step_loss, results["best_train_loss"], standard="min"):
            results["best_train_loss"] = train_step_loss
            results["best_train_on_test"] = test_result
            results["best_train_epoch"] = epoch
            if task_names:
                results["best_train_on_test_per_task"] = extract_per_task_test_metrics(
                    test_results, task_names, metric_key
                )
            if cfg["basic"]["save_finetune_ckpt"]:
                _save_backend_ckpt(
                    log_dir,
                    "train_best",
                    backend.model,
                    backend.optimizer,
                    epoch,
                    backend.scheduler,
                    results,
                )

        if is_left_better_right(valid_result, results["best_valid"], standard=valid_select):
            results["best_valid"] = valid_result
            results["best_valid_on_test"] = test_result
            results["best_valid_epoch"] = epoch
            if task_names:
                results["best_valid_on_test_per_task"] = extract_per_task_test_metrics(
                    test_results, task_names, metric_key
                )
            if cfg["basic"]["save_finetune_ckpt"]:
                _save_backend_ckpt(
                    log_dir,
                    "valid_best",
                    backend.model,
                    backend.optimizer,
                    epoch,
                    backend.scheduler,
                    results,
                )

        reg_plot = eval_metric.lower() if task_type == "regression" else "rmse"
        plot_loss_rocauc(loss_list, valid_metric_list, task_type, log_dir=log_dir, regression_metric=reg_plot)

    pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)
    with open(os.path.join(log_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
