# SPDX-License-Identifier: MIT
"""ImageMol-style metrics (sigmoid + ROC-AUC / RMSE)."""

from __future__ import annotations

import numpy as np
import torch

from models.evaluate import metric, metric_multitask, metric_reg


@torch.no_grad()
def evaluate_imagemol_predictions(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    *,
    task_type: str,
    num_tasks: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_logits = np.asarray(y_logits, dtype=np.float32)

    if task_type == "classification":
        y_prob = 1.0 / (1.0 + np.exp(-y_logits))
        if num_tasks == 1:
            y_t = y_true.reshape(-1).astype(int)
            valid = y_t >= 0
            y_p = (y_prob.reshape(-1)[valid] >= 0.5).astype(int)
            y_pr = y_prob.reshape(-1)[valid]
            return metric(y_t[valid], y_p, y_pr, empty=-1)
        y_p = (y_prob >= 0.5).astype(int)
        return metric_multitask(y_true, y_p, y_prob, num_tasks=num_tasks, empty=-1)

    if y_logits.ndim == 1:
        y_logits = y_logits.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape[1] == 1:
        return metric_reg(y_true.ravel(), y_logits.ravel())
    from models.evaluate import metric_reg_multitask

    return metric_reg_multitask(y_true, y_logits, num_tasks=y_true.shape[1])
