# SPDX-License-Identifier: MIT
"""Label conventions for MolCLR finetune inside ChemVL."""

from __future__ import annotations

import numpy as np


def prepare_molclr_classification_labels(labels: np.ndarray) -> np.ndarray:
    """
    Map ChemVL / MolMCL tabular labels to MolCLR binary {0, 1} with -1 = missing.

    - If column uses MolMCL-style {-1, 1}: map -1 -> 0, 1 -> 1.
    - If column uses {0, 1} (processed_ac): keep as-is.
    """
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    out = np.full_like(labels, -1.0, dtype=np.float32)
    for j in range(labels.shape[1]):
        col = labels[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        uniq = set(np.unique(finite).tolist())
        if uniq <= {-1.0, 1.0} and 1.0 in uniq:
            out[:, j] = np.where(col == 1, 1.0, np.where(col == -1, 0.0, -1.0))
        else:
            out[:, j] = col
    return out


def task_column_labels(labels: np.ndarray, task_index: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim == 1:
        return labels.reshape(-1)
    return labels[:, task_index].reshape(-1)
