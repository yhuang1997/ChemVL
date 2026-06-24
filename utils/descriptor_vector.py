"""RDKit descriptor matrices aligned with ``PriorKnowledgeLib`` key order (e.g. version ``all``)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

from utils.path_utils import get_descriptor_only_text_cache_dir
from utils.prior_knowledge import PriorKnowledgeLib

SMILES_COL = "smiles"


def _descriptor_csv_cache_path(prior_version: str) -> str:
    """One CSV per ``prior_version`` under the same root as descriptor-only text cache."""
    base = Path(get_descriptor_only_text_cache_dir())
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"rdkit_descriptors_by_smiles_{prior_version}.csv")


def _ordered_descriptor_names(lib: PriorKnowledgeLib) -> List[str]:
    """Same column order as ``_compute_one_descriptor_row`` / ``Descriptors.descList`` ∩ ``prior_keys``."""
    prior_set = frozenset(lib.prior_keys)
    return [name for name, _ in Descriptors.descList if name in prior_set]


def _csv_fieldnames(ordered_names: Sequence[str]) -> List[str]:
    return [SMILES_COL] + list(ordered_names)


def _parse_csv_cell(s: Optional[str]) -> float:
    if s is None:
        return float("nan")
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return float("nan")
    return float(t)


def _format_csv_cell(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return repr(float(v))


def _load_smiles_descriptor_csv(path: str, ordered_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """``smiles`` -> 1d float array aligned with ``ordered_names``."""
    if not os.path.isfile(path):
        return {}
    expected = _csv_fieldnames(ordered_names)
    out: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        got = [h.strip() for h in reader.fieldnames]
        if got != expected:
            return {}
        for row in reader:
            smi = row.get(SMILES_COL)
            if smi is None or str(smi).strip() == "":
                continue
            smi = str(smi).strip()
            try:
                vals = [_parse_csv_cell(row.get(name)) for name in ordered_names]
            except (TypeError, ValueError):
                continue
            arr = np.asarray(vals, dtype=np.float64)
            if arr.shape[0] == len(ordered_names):
                out[smi] = arr
    return out


def _save_smiles_descriptor_csv(path: str, ordered_names: Sequence[str], data: Dict[str, np.ndarray]) -> None:
    fieldnames = _csv_fieldnames(ordered_names)
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for smi in sorted(data.keys()):
            arr = data[smi]
            row_dict: Dict[str, str] = {SMILES_COL: smi}
            for i, name in enumerate(ordered_names):
                row_dict[name] = _format_csv_cell(float(arr[i]))
            w.writerow(row_dict)
    os.replace(tmp, path)


def _compute_one_descriptor_row(mol, lib: PriorKnowledgeLib) -> np.ndarray:
    descriptors = {name: float(func(mol)) for name, func in Descriptors.descList if name in lib.prior_keys}
    row = [descriptors[key] for key, _ in Descriptors.descList if key in descriptors]
    if len(row) != len(lib.prior_keys):
        raise RuntimeError(
            f"Descriptor count mismatch: got {len(row)}, expected {len(lib.prior_keys)} for version={lib.version!r}"
        )
    return np.asarray(row, dtype=np.float64)


def compute_descriptor_matrix(
    smiles_list: Sequence[str],
    prior_version: str = "all",
    use_rdkit_cache: bool = True,
) -> np.ndarray:
    """
    One row per SMILES, columns in ``Descriptors.descList`` order restricted to ``prior_keys``
    (same order as ``PriorKnowledgeLib.load_prior_knowledge_features`` dict iteration).

    When ``use_rdkit_cache`` is True (default), load/merge/save one CSV::

        {get_descriptor_only_text_cache_dir()}/rdkit_descriptors_by_smiles_{prior_version}.csv

    Column 1 is ``smiles``; remaining columns are RDKit descriptor names (same order as matrix columns).
    ``prior_version`` is encoded in the filename. If the header does not match the current descriptor set,
    the file is ignored and rebuilt on save.
    """
    lib = PriorKnowledgeLib(version=prior_version)
    d_expected = len(lib.prior_keys)
    ordered_names = _ordered_descriptor_names(lib)
    if len(ordered_names) != d_expected:
        raise RuntimeError(
            f"Ordered descriptor name count {len(ordered_names)} != len(prior_keys)={d_expected} for version={prior_version!r}"
        )

    cache_path = _descriptor_csv_cache_path(prior_version)
    cache_data: Dict[str, np.ndarray] = {}
    if use_rdkit_cache:
        cache_data = _load_smiles_descriptor_csv(cache_path, ordered_names)

    rows: List[np.ndarray] = []
    dirty = False
    for smi in smiles_list:
        s = str(smi)
        row: Optional[np.ndarray] = None
        if use_rdkit_cache and s in cache_data:
            row = cache_data[s]
            if row.shape != (d_expected,):
                row = None
        if row is None:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES for descriptor computation: {smi!r}")
            row = _compute_one_descriptor_row(mol, lib)
            if use_rdkit_cache:
                cache_data[s] = row.copy()
                dirty = True
        rows.append(row)

    if use_rdkit_cache and dirty:
        _save_smiles_descriptor_csv(cache_path, ordered_names, cache_data)

    return np.vstack(rows).astype(np.float64)


def _fill_nan_inf_train_medians(X: np.ndarray, train_row_mask: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    train = X[train_row_mask]
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = ~np.isfinite(X)
    if inds.any():
        X[inds] = np.take(med, np.nonzero(inds)[1])
    return X


def fit_transform_descriptor_features(
    X: np.ndarray, train_idx: np.ndarray
) -> Tuple[np.ndarray, StandardScaler]:
    """MoleculeACE-style: fit ``StandardScaler`` on train rows only, then transform full ``X``."""
    mask = np.zeros(len(X), dtype=bool)
    mask[np.asarray(train_idx, dtype=np.int64)] = True
    X_clean = _fill_nan_inf_train_medians(X, mask)
    scaler = StandardScaler()
    scaler.fit(X_clean[mask])
    X_scaled = scaler.transform(X_clean)
    return X_scaled.astype(np.float32), scaler
