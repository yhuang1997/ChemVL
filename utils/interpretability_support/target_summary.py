"""Aggregate per-dataset task_summary.json files into target_summary.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def find_task_summaries(root: Path) -> List[Path]:
    return sorted(root.glob("*/task_summary.json"))


def load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid task_summary: {path}")
    data["task_summary_path"] = str(path.resolve())
    data["run_dir"] = str(path.parent.resolve())
    return data


def aggregate_target_summary(root: Path, output: Path) -> pd.DataFrame:
    paths = find_task_summaries(root)
    if not paths:
        return pd.DataFrame()

    rows = [load_summary(p) for p in paths]
    df = pd.DataFrame(rows)

    front = [
        "dataset_id",
        "split",
        "checkpoint_path",
        "run_dir",
        "n_molecules",
        "n_molecules_in_csv",
        "n_train",
        "n_val",
        "n_test",
        "series_max_abs_error",
        "n_molecules_series_pool",
        "n_molecules_excluded_series_filter",
        "n_activity_cliffs_delta2",
        "n_activity_cliffs_delta3",
        "gradcam_available",
        "timestamp_utc",
    ]
    metric_cols = sorted(c for c in df.columns if c.endswith("_r2") or c.endswith("_rmse"))
    tail = ["task_summary_path"]
    ordered = [c for c in front if c in df.columns] + metric_cols + tail
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    df = df.sort_values(["dataset_id", "split"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, float_format="%.6f")
    return df
