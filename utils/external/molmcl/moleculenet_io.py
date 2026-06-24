# SPDX-License-Identifier: MIT
"""MoleculeNet CSV layout + MolMCL-style scaffold split for ``finetune_external``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.external.molmcl.molmcl_external_config import chemvl_repo_root


def moleculenet_search_roots(dataroot: str) -> List[str]:
    """
    Roots under which ``MPP/classification|regression/<task>/processed/`` may live.

    If ``dataroot`` ends with ``MoleculeNet`` (legacy placeholder), also search its parent
    (e.g. ``.../finetuning_datasets``) so paths like ``.../finetuning_datasets/MPP/classification/bace/...`` resolve.
    """
    root = os.path.abspath(os.path.expanduser(dataroot.strip()))
    bases = [root]
    if os.path.basename(root.rstrip(os.sep)).lower() == "moleculenet":
        parent = os.path.dirname(root.rstrip(os.sep))
        if parent:
            bases.append(parent)
    seen: set[str] = set()
    out: List[str] = []
    for b in bases:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _mpp_subdir(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("dataset") or {}).get("moleculenet_mpp_subdir") or "MPP").strip().strip("/") or "MPP"


def _split_search_order(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Prefer the split matching ``dataset.task_type`` when resolving CSV path."""
    tt = str((cfg.get("dataset") or {}).get("task_type", "classification")).lower()
    cls, reg = "classification", "regression"
    if tt == "regression":
        return reg, cls
    return cls, reg


def _resolve_under_repo(repo: Path, path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return str((repo / path).resolve())


def _processed_ac_path(base: str, mpp: str, split: str, stem: str) -> str:
    return os.path.join(base, mpp, split, stem, "processed", f"{stem}_processed_ac.csv")


def _normalize_moleculenet_dataroot(dataroot: str, mpp: str) -> str:
    """
    Strip trailing ``MPP/{classification|regression}`` if present.

    Task overlays from integrated configs often set dataroot to ``.../MPP/classification``;
    ``resolve_moleculenet_csv`` then appends ``MPP/...`` again unless normalized.
    """
    root = os.path.abspath(os.path.expanduser(dataroot.strip()))
    norm = root.replace("\\", "/").rstrip("/")
    mpp = mpp.strip("/")
    for split in ("classification", "regression"):
        suffix = f"{mpp}/{split}"
        if norm.endswith(suffix):
            return os.path.dirname(os.path.dirname(root))
    if norm.endswith(f"/{mpp}"):
        return os.path.dirname(root.rstrip(os.sep))
    return root


def resolve_moleculenet_csv(cfg: Dict[str, Any], repo: Path | None = None) -> str:
    """
    Return path to ChemVL MPP ``*_processed_ac.csv`` or legacy raw ``<task>.csv``.

    Preferred layout (same as MoleculeACE naming)::

        {dataroot_or_parent}/MPP/classification/{task}/processed/{task}_processed_ac.csv
        {dataroot_or_parent}/MPP/regression/{task}/processed/{task}_processed_ac.csv
    """
    repo = repo or chemvl_repo_root()
    ds = cfg.get("dataset") or {}
    name = str(ds.get("dataset", "")).strip().lower()
    if not name:
        raise ValueError("dataset.dataset is empty.")
    mpp = _mpp_subdir(cfg)
    root = _resolve_under_repo(repo, str(ds.get("dataroot", "")).strip())
    root = _normalize_moleculenet_dataroot(root, mpp)
    first, second = _split_search_order(cfg)

    stems = [name]
    if name == "lipo":
        stems.append("lipophilicity")
    elif name == "lipophilicity":
        stems.append("lipo")

    tried: list[str] = []
    for stem in stems:
        for base in moleculenet_search_roots(root):
            for split in (first, second):
                c = _processed_ac_path(base, mpp, split, stem)
                tried.append(c)
                if os.path.isfile(c):
                    return c

        for base in moleculenet_search_roots(root):
            candidates = [
                os.path.join(base, stem, f"{stem}.csv"),
                os.path.join(base, f"{stem}.csv"),
                os.path.join(base, stem.upper(), f"{stem}.csv"),
                os.path.join(base, stem.upper(), f"{stem.upper()}.csv"),
            ]
            for c in candidates:
                tried.append(c)
                if os.path.isfile(c):
                    return c
    raise FileNotFoundError(f"MoleculeNet data CSV not found for {name!r}. Tried (first ~12): {tried[:12]!r} …")


def discover_moleculenet_from_dataroot(dataroot: str, mpp_subdir: str = "MPP") -> List[str]:
    """List task stems that have ``{mpp}/{cls|reg}/{stem}/processed/{stem}_processed_ac.csv`` under search roots."""
    found: set[str] = set()
    for base in moleculenet_search_roots(dataroot):
        mpp_root = os.path.join(base, mpp_subdir)
        if not os.path.isdir(mpp_root):
            continue
        for split in ("classification", "regression"):
            split_dir = os.path.join(mpp_root, split)
            if not os.path.isdir(split_dir):
                continue
            for stem in os.listdir(split_dir):
                s = stem.strip().lower()
                p = _processed_ac_path(base, mpp_subdir, split, s)
                if os.path.isfile(p):
                    found.add(s)
    return sorted(found)


def _bbbp(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    smiles = df["smiles"].to_list()
    y = df[["p_np"]].replace(0, -1).values.astype(np.float32)
    return smiles, y


def _clintox(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    smiles = df["smiles"].to_list()
    y = df[["FDA_APPROVED", "CT_TOX"]].replace(0, -1).values.astype(np.float32)
    return smiles, y


def _tox21(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    cols = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    smiles = df["smiles"].to_list()
    y = df[cols].replace(0, -1).fillna(0).values.astype(np.float32)
    return smiles, y


def _sider(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    smiles = df["smiles"].to_list()
    label_cols = [c for c in df.columns if c != "smiles"]
    y = df[label_cols].replace(0, -1).values.astype(np.float32)
    return smiles, y


def _bace(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    col = "mol" if "mol" in df.columns else "smiles"
    smiles = df[col].to_list()
    y = df[["Class"]].replace(0, -1).values.astype(np.float32)
    return smiles, y


def _hiv(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    smiles = df["smiles"].to_list()
    y = df[["HIV_active"]].replace(0, -1).values.astype(np.float32)
    return smiles, y


def _regression_smiles_y(df: pd.DataFrame, smiles_cols: List[str], y_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    sc = next(c for c in smiles_cols if c in df.columns)
    yc = next(c for c in y_cols if c in df.columns)
    smiles = df[sc].astype(str).to_list()
    y = df[[yc]].values.astype(np.float32)
    return smiles, y


def _is_processed_ac_path(path: str) -> bool:
    return path.rstrip().lower().endswith("_processed_ac.csv")


def _molmcl_classification_labels(y: np.ndarray) -> np.ndarray:
    """Map ChemVL 0/1 columns to MolMCL {-1, 1} per column when that column is binary (MolMCL ``loader`` convention)."""
    y = np.asarray(y, dtype=np.float32)
    out = y.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        u = np.unique(finite.astype(np.int64))
        if set(u.tolist()) <= {0, 1}:
            out[:, j] = np.where(col == 0, -1.0, col)
    return out


def load_moleculenet_smiles_labels(cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    """Load (smiles, labels) for MolMCL-style training."""
    repo = chemvl_repo_root()
    path = resolve_moleculenet_csv(cfg, repo)
    name = str((cfg.get("dataset") or {}).get("dataset", "")).strip().lower()
    task_type = str((cfg.get("dataset") or {}).get("task_type", "classification")).lower()

    if _is_processed_ac_path(path):
        from utils.external.molmcl.moleculeace_tabular import load_moleculeace_tabular_multitask
        from utils.train_utils import load_smiles

        _names, labels = load_moleculeace_tabular_multitask(path, task_type=task_type)
        smiles = load_smiles(path)
        labels = np.asarray(labels, dtype=np.float32)
        if task_type == "classification":
            labels = _molmcl_classification_labels(labels)
        return smiles, labels

    df = pd.read_csv(path)
    if name == "bbbp":
        smiles, y = _bbbp(df)
    elif name == "clintox":
        smiles, y = _clintox(df)
    elif name == "tox21":
        smiles, y = _tox21(df)
    elif name == "sider":
        smiles, y = _sider(df)
    elif name == "bace":
        smiles, y = _bace(df)
    elif name == "hiv":
        smiles, y = _hiv(df)
    elif name == "esol":
        smiles, y = _regression_smiles_y(
            df,
            ["smiles", "SMILES", "compound_iso_smiles"],
            [
                "measured log solubility in mols per litre",
                "measured log solubility in mols per litre ",
                "y",
                "exp",
            ],
        )
    elif name == "freesolv":
        smiles, y = _regression_smiles_y(df, ["smiles", "SMILES"], ["expt", "y", "exp"])
    elif name in ("lipo", "lipophilicity"):
        smiles, y = _regression_smiles_y(
            df, ["smiles", "SMILES"], ["exp", "y", "value", "label", "lipo_exp"]
        )
    elif name == "qm7":
        smiles, y = _regression_smiles_y(
            df,
            ["smiles", "SMILES", "molecule", "mol"],
            ["u0_atom", "U0_atom", "y", "exp", "internal_energy_at_0K"],
        )
    else:
        raise NotImplementedError(
            f"MoleculeNet dataset {name!r} has no ChemVL loader mapping; extend moleculenet_io.load_moleculenet_smiles_labels."
        )

    return smiles, y
