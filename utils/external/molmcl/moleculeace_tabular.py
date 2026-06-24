# SPDX-License-Identifier: MIT
"""MoleculeACE tabular paths and labels for graph-only / external backends (no PNG directory)."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def get_moleculeace_processed_ac_csv(cfg: Dict[str, Any], data_type: str = "processed") -> str:
    """
    Path to ``{dataroot}/{dataset}/{data_type}/{dataset}_processed_ac.csv`` (same CSV as ChemVL image pipeline).

    Does **not** require the depiction image subdirectory to exist.
    """
    dataset = cfg["dataset"]["dataset"]
    dataroot = cfg["dataset"]["dataroot"]
    txt_file = os.path.join(dataroot, dataset, data_type, f"{dataset}_processed_ac.csv")
    if not os.path.isfile(txt_file):
        raise FileNotFoundError(f"MoleculeACE processed CSV not found: {txt_file}")
    return txt_file


def load_moleculeace_tabular_multitask(txt_file: str, task_type: str = "regression") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read ``index`` / ``label`` columns like ``load_filenames_and_labels_multitask``, but row names are
    string indices only (no ``image_folder`` / ``.png`` paths).
    """
    assert task_type in ("classification", "regression")
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(" ")).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    names = np.array([str(i) for i in index], dtype=object)
    if not (len(index) == labels.shape[0] == len(names)):
        raise ValueError("index / label row count mismatch in processed CSV.")
    return names, labels
