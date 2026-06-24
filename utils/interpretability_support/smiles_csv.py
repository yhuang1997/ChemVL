"""Resolve SMILES from curated group CSV by (dataset_id, export_id)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from utils.interpretability_support.selection_xlsx import SelectionRow, parse_dataset_id_from_rel_path


def parse_group_csv_from_export_id(export_id: str, dataset_id: str, base_dir: Path) -> Optional[Path]:
    ds_root = base_dir / dataset_id
    m_series = re.match(r"^(\d+)_Ki_all_series_(\d{4})_\d{4}$", export_id)
    if m_series:
        short, group = m_series.group(1), int(m_series.group(2))
        return ds_root / "series" / "csv" / f"{short}_Ki_all_series_{group:04d}.csv"
    m_cliff = re.match(r"^ac_(delta\d+)_(\d{4})_\d{4}$", export_id)
    if m_cliff:
        delta_key, group = m_cliff.group(1), int(m_cliff.group(2))
        return ds_root / "activity_cliffs" / delta_key / "csv" / f"ac_{delta_key}_{group:04d}.csv"
    return None


def resolve_molecule_from_csv(row: SelectionRow, base_dir: Path) -> Tuple[str, Optional[float], str]:
    """Return (smiles, gt, csv_path) for a selection row."""
    dataset_id = parse_dataset_id_from_rel_path(row.rel_path)
    csv_path = parse_group_csv_from_export_id(row.export_id, dataset_id, base_dir)
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(f"cannot locate group CSV for {dataset_id}/{row.export_id}")
    df = pd.read_csv(csv_path)
    if "export_id" not in df.columns or "smiles" not in df.columns:
        raise ValueError(f"CSV missing export_id/smiles columns: {csv_path}")
    hit = df.loc[df["export_id"].astype(str) == row.export_id]
    if len(hit) == 0:
        raise KeyError(f"export_id {row.export_id} not in {csv_path}")
    smiles = str(hit["smiles"].iloc[0])
    gt: Optional[float] = None
    if "gt" in hit.columns:
        gt_raw = hit["gt"].iloc[0]
        if pd.notna(gt_raw):
            gt = float(gt_raw)
    return smiles, gt, str(csv_path.resolve())


def resolve_smiles_from_csv(row: SelectionRow, base_dir: Path) -> Tuple[str, str]:
    smiles, _gt, csv_path = resolve_molecule_from_csv(row, base_dir)
    return smiles, csv_path
