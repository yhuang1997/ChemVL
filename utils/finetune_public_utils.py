"""Public-release finetune helpers (test eval only after training)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from models.clip_model_utils import evaluate_on_multitask
from utils.finetune_utils import extract_per_task_test_metrics
from utils.interpretability_support.visual_utils import load_finetuned_model


def _eval_ckpt_on_test(
    *,
    ckpt_path: str,
    cfg_path: str,
    test_dataloader,
    device,
    task_type: str,
    eval_with_tta: bool,
    metric_key: str,
    task_names: Optional[List[str]],
) -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        return {"error": f"missing checkpoint: {ckpt_path}"}

    model = load_finetuned_model(cfg_path, model_weights_path=ckpt_path, device=device)
    test_results, _ = evaluate_on_multitask(
        model=model,
        data_loader=test_dataloader,
        device=device,
        task_type=task_type,
        return_data_dict=True,
        tta=eval_with_tta,
    )
    out: Dict[str, Any] = {metric_key.lower(): float(test_results[metric_key])}
    if task_names:
        out["per_task"] = extract_per_task_test_metrics(test_results, task_names, metric_key)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def final_eval_saved_ckpts_on_test(
    *,
    log_dir: str,
    test_dataloader,
    device,
    task_type: str,
    eval_with_tta: bool,
    eval_metric: str,
    results: Dict[str, Any],
    task_names: Optional[List[str]] = None,
) -> None:
    """Load ``train_best`` / ``valid_best`` from ``log_dir`` and evaluate on test once each.

    Updates ``results`` in place with ``final_test_train_best`` and ``final_test_valid_best``.
    These fields are written only after training; they are never used for checkpoint selection.
    """
    cfg_path = os.path.join(log_dir, "config.json")
    metric_key = eval_metric.upper()
    for tag in ("train_best", "valid_best"):
        ckpt_path = os.path.join(log_dir, f"{tag}.pth")
        results[f"final_test_{tag}"] = _eval_ckpt_on_test(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            test_dataloader=test_dataloader,
            device=device,
            task_type=task_type,
            eval_with_tta=eval_with_tta,
            metric_key=metric_key,
            task_names=task_names,
        )
