"""
Descriptor-only training session: RDKit @feature (vector + MLP) and shared train/eval utilities.

Used by ``scripts/.../descriptor_only_finetune.py`` (full finetune + text branch loop) and by
``utils.fs_zs_support.run_few_shot_pt`` (few-shot K-shot scaler). Lives under ``utils/`` so
descriptor-only work can be cherry-picked without pulling in ``fs_zs_support``.

``run_descriptor_only_session`` implements **feature** mode only; ``text`` remains in the
standalone script after local loader setup.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataloader.descriptor_only_dataloader import build_descriptor_only_dataloaders
from models.clip_model_utils import save_finetune_ckpt
from models.descriptor_only import build_descriptor_only_model
from models.evaluate import metric as utils_evaluate_metric
from models.evaluate import metric_multitask as utils_evaluate_metric_multitask
from models.evaluate import metric_reg as utils_evaluate_metric_reg
from models.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask
from utils.descriptor_vector import compute_descriptor_matrix, fit_transform_descriptor_features
from utils.finetune_utils import get_metric, get_scheduler, get_logdir_and_save_config, prefix_add_formatted_time
from utils.plot_utils import plot_loss_rocauc
from utils.public_utils import is_left_better_right, setup_device
from utils.train_utils import fix_train_random_seed


def _multitask_classification_probs_preds(y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logits layout (N, C, T) from DescriptorOnly*Model.stack(..., dim=-1).
    Returns y_pred (N, T), positive-class probabilities (N, T) clipped to [0, 1].
    """
    logits = torch.as_tensor(y_scores, dtype=torch.float32)
    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
    if logits.dim() != 3:
        raise ValueError(f"Expected classification logits (N, C, T), got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError(
            f"Binary/multitask classification needs >=2 logits per task (C={logits.shape[1]}). "
            "Check dataset.class_names and model heads."
        )
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    y_prob = probs[:, 1, :]
    if not np.isfinite(y_prob).all():
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_pred, y_prob


@torch.no_grad()
def evaluate_descriptor_only(model, data_loader, device, task_type: str, mode: str):
    model.eval()
    y_scores, y_true = [], []
    print("Calculating probs...")
    for step, data in enumerate(data_loader):
        if mode == "text":
            smiles, labels = data
            labels = labels.to(device)
            pred = model(smiles=smiles)
        else:
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
        if task_type == "classification":
            labels = labels.to(torch.int64)
        else:
            labels = labels.to(torch.float32)
        y_true.append(labels)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if task_type == "regression":
        if y_scores.shape[1] == 1 and len(y_scores.shape) == 3:
            y_scores = y_scores.squeeze(1)

    print("Calculating metrics...")
    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pred, y_pro = _multitask_classification_probs_preds(y_scores)
            return utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), {}
        return utils_evaluate_metric_reg(y_true, y_scores), {}
    if task_type == "classification":
        y_pred, y_pro = _multitask_classification_probs_preds(y_scores)
        return utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1), {}
    return utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1]), {}


def train_one_epoch_descriptor(
    model, optimizer, data_loader, criterion, weights, device, task_type: str, mode: str, scheduler
):
    model.train()
    optimizer.zero_grad()
    accu_loss = torch.zeros(1).to(device)
    step = -1
    for step, data in enumerate(data_loader):
        if mode == "text":
            smiles, labels = data
            labels = labels.to(device)
            pred = model(smiles=smiles)
        else:
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
        if task_type == "classification":
            labels = labels.to(torch.int64)
            is_valid = labels != -1
            loss_mat = criterion(pred, labels)
            loss_mat = torch.where(
                is_valid, loss_mat, torch.zeros(loss_mat.shape, device=loss_mat.device, dtype=loss_mat.dtype)
            )
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        else:
            labels = labels.to(torch.float32)
            pred = pred.squeeze(1)
            loss = criterion(pred, labels)

        print(f"step: {step}, loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accu_loss += loss.detach()

    if scheduler is not None:
        scheduler.step()
    return accu_loss.item() / max(1, step + 1)


def build_descriptor_only_optimizer(cfg: Dict[str, Any], model: nn.Module):
    name = (cfg.get("training") or {}).get("optimizer", "AdamW")
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=wd, momentum=float(cfg["training"].get("momentum", 0.9))
        )
    raise ValueError(f"Unsupported optimizer for descriptor-only: {name!r}")


def descriptor_training_main_loop(
    cfg: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    weights,
    device: torch.device,
    device_ids,
    task_type: str,
    mode: str,
    log_dir: str,
    eval_metric: str,
    valid_select: str,
    min_value: float,
) -> Dict[str, Any]:
    """Epoch loop shared by feature session and text path in ``descriptor_only_finetune``."""
    loss_list = []
    valid_metric_list = []
    history_rows = []
    history_csv_path = os.path.join(log_dir, "train_val_test_history.csv")
    results: Dict[str, Any] = {
        "best_valid": min_value,
        "best_valid_epoch": 0,
        "best_train_loss": np.inf,
        "best_train_epoch": 0,
    }

    for epoch in range(cfg["training"]["start_epoch"], cfg["training"]["epochs"]):
        train_step_loss = train_one_epoch_descriptor(
            model,
            optimizer,
            train_loader,
            criterion,
            weights,
            device,
            task_type,
            mode,
            scheduler,
        )
        val_results, _ = evaluate_descriptor_only(model, val_loader, device, task_type, mode)
        test_results, _ = evaluate_descriptor_only(model, test_loader, device, task_type, mode)

        loss_list.append(train_step_loss)
        valid_result = float(val_results[eval_metric.upper()])
        test_result = float(test_results[eval_metric.upper()])
        valid_metric_list.append(valid_result)

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

        m = model.module if isinstance(model, nn.DataParallel) else model
        if is_left_better_right(train_step_loss, results["best_train_loss"], standard="min"):
            results["best_train_loss"] = train_step_loss
            results["best_train_on_test"] = test_result
            results["best_train_epoch"] = epoch
            if cfg["basic"].get("save_finetune_ckpt"):
                save_finetune_ckpt(
                    m,
                    optimizer,
                    epoch,
                    log_dir,
                    "train_best",
                    lr_scheduler=scheduler,
                    result_dict=results,
                )

        if is_left_better_right(valid_result, results["best_valid"], standard=valid_select):
            results["best_valid"] = valid_result
            results["best_valid_on_test"] = test_result
            results["best_valid_epoch"] = epoch
            if cfg["basic"].get("save_finetune_ckpt"):
                save_finetune_ckpt(
                    m,
                    optimizer,
                    epoch,
                    log_dir,
                    "valid_best",
                    lr_scheduler=scheduler,
                    result_dict=results,
                )

        plot_loss_rocauc(loss_list, valid_metric_list, task_type, log_dir=log_dir)

    pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)
    result_path = os.path.join(log_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    return results


def run_descriptor_only_session(
    cfg: Dict[str, Any],
    smiles: np.ndarray,
    labels: np.ndarray,
    train_row_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[str, Dict[str, Any]]:
    """
    Descriptor-only MLP on RDKit feature vectors. Fits StandardScaler on ``train_row_idx`` rows only
    (few-shot: pass K-shot indices; full finetuning: pass full scaffold train indices).

    Raises:
        NotImplementedError: if ``dataset.descriptor_only_mode`` is not ``feature``.
    """
    mode = (cfg["dataset"].get("descriptor_only_mode") or "").strip().lower()
    if mode != "feature":
        raise NotImplementedError(
            "utils.descriptor_only_session.run_descriptor_only_session only supports "
            f"descriptor_only_mode='feature' (got {mode!r}). Use the text branch in "
            "scripts/.../descriptor_only_finetune.py for text mode."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["basic"]["gpu"])
    device, device_ids = setup_device(cfg["basic"]["ngpu"])
    fix_train_random_seed(cfg["training"]["runseed"])

    cfg["basic"]["prefix"], cfg["basic"]["timestamp"] = prefix_add_formatted_time(cfg["basic"].get("prefix"))
    log_dir = get_logdir_and_save_config(cfg, mkdir=True)

    task_type = cfg["dataset"]["task_type"]
    train_row_idx = np.asarray(train_row_idx, dtype=np.int64)
    prior_ver = (cfg["dataset"].get("prior_descriptor_version") or "all").strip()
    X_all = compute_descriptor_matrix(smiles, prior_version=prior_ver)
    X_scaled, _scaler = fit_transform_descriptor_features(X_all, train_row_idx)

    train_loader, val_loader, test_loader = build_descriptor_only_dataloaders(
        cfg, "feature", X_scaled, smiles, labels, train_row_idx, val_idx, test_idx
    )

    eval_metric, valid_select, min_value, criterion, weights = get_metric(cfg, labels[train_row_idx])

    model = build_descriptor_only_model(cfg)
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    core = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = build_descriptor_only_optimizer(cfg, core)
    scheduler = get_scheduler(cfg, optimizer)

    results = descriptor_training_main_loop(
        cfg,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        weights,
        device,
        device_ids,
        task_type,
        "feature",
        log_dir,
        eval_metric,
        valid_select,
        min_value,
    )
    return log_dir, results
