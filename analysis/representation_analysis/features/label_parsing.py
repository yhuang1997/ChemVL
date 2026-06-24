"""Shared label parsing for multitask CSV columns (e.g. Tox21 space-separated floats)."""

from __future__ import annotations

from typing import List

import numpy as np


def _space_separated_rows_to_float_matrix(targets: np.ndarray) -> np.ndarray:
    """Parse each row string into a list of floats; return shape (n_rows, n_tasks)."""
    rows: List[List[float]] = []
    widths: List[int] = []
    for t in targets:
        parts = str(t).strip().split()
        widths.append(len(parts))
        if not parts:
            rows.append([float("nan")])
            continue
        row: List[float] = []
        for p in parts:
            try:
                row.append(float(p))
            except ValueError:
                row.append(float("nan"))
        rows.append(row)
    if len(set(widths)) > 1:
        raise ValueError(
            f"Inconsistent multitask label width across rows: min={min(widths)} max={max(widths)}"
        )
    return np.asarray(rows, dtype=np.float64)


def parse_space_separated_multitask_matrix(targets: np.ndarray) -> np.ndarray:
    """
    Each row is whitespace-separated task labels (e.g. ChemVL ``label`` column: ``0 1 -1 ...``).
    Returns ``float64`` array of shape ``(n_rows, n_tasks)``.
    """
    parsed = _space_separated_rows_to_float_matrix(targets)
    if parsed.ndim != 2 or parsed.shape[1] < 1:
        raise ValueError(f"Expected 2D multitask labels, got shape {parsed.shape}")
    return parsed


def parse_space_separated_multitask_column(targets: np.ndarray, task_id: int) -> np.ndarray:
    """
    Each row is whitespace-separated task labels; tokens may be integers or floats (``0.0``).
    Returns the ``task_id`` column as ``float64`` (NaN for parse failures / empty cells).
    """
    parsed = _space_separated_rows_to_float_matrix(targets)
    if parsed.ndim != 2 or task_id < 0 or task_id >= parsed.shape[1]:
        raise ValueError(
            f"label_parse_space_separated: task_id={task_id} invalid for parsed shape {parsed.shape}"
        )
    return parsed[:, task_id]
