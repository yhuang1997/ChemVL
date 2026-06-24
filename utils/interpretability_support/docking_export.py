"""Generate flat 2D/3D SDF batches from curated CSV SMILES for docking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd

from utils.interpretability_support.selection_xlsx import (
    SelectionRow,
    flat_name_for_row,
    load_selection_xlsx,
)
from utils.interpretability_support.sdf_2d import write_sdf_2d
from utils.interpretability_support.sdf_3d import write_sdf_3d_multi
from utils.interpretability_support.smiles_csv import resolve_smiles_from_csv

SdfDim = Literal["2d", "3d"]

FLAT_DIR_NAMES = {
    "2d": "SDF_2d_flat_from_smiles",
    "3d": "SDF_3d_flat_from_smiles",
}


def default_flat_dir(base_dir: Path, sdf_dim: SdfDim) -> Path:
    return (base_dir / FLAT_DIR_NAMES[sdf_dim]).resolve()


def flat_rel_prefix(sdf_dim: SdfDim) -> str:
    return FLAT_DIR_NAMES[sdf_dim]


def write_selection_xlsx_outputs(
    flat_dir: Path,
    xlsx_src: Path,
    table_rows: List[Dict[str, Any]],
    *,
    sdf_dim: SdfDim,
) -> None:
    """Copy original xlsx and write enriched workbook with smiles column."""
    flat_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xlsx_src, flat_dir / xlsx_src.name)

    df = pd.DataFrame(table_rows)
    out_xlsx = flat_dir / "temp_selected_sdf_with_smiles.xlsx"
    try:
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Writing xlsx requires openpyxl. Install: pip install openpyxl"
        ) from exc


def _write_sdf(
    smiles: str,
    dst: Path,
    *,
    export_id: str,
    sdf_dim: SdfDim,
    num_confs: int,
) -> Dict[str, Any]:
    if sdf_dim == "2d":
        if not write_sdf_2d(smiles, dst):
            raise RuntimeError("2D SDF write failed")
        return {"out_path": str(dst)}
    return write_sdf_3d_multi(smiles, dst, export_id=export_id, num_confs=num_confs)


def generate_flat_from_smiles(
    base_dir: Path,
    selection: List[SelectionRow],
    *,
    flat_dir: Path,
    xlsx_src: Path,
    sdf_dim: SdfDim = "3d",
    num_confs: int = 20,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Embed from CSV SMILES into a single flat directory (no SDF copy).

    Returns (successes, failures, xlsx_table_rows).
    """
    flat_dir = flat_dir.resolve()
    flat_col = f"flat_sdf_{sdf_dim}"
    rel_prefix = flat_rel_prefix(sdf_dim)
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    xlsx_rows: List[Dict[str, Any]] = []

    for row in selection:
        dataset_id, flat_name = flat_name_for_row(row)
        dst = flat_dir / flat_name
        flat_rel = f"{rel_prefix}/{flat_name}"

        try:
            smiles, csv_path = resolve_smiles_from_csv(row, base_dir)
        except Exception as exc:
            failures.append(
                {
                    "index": row.index,
                    "export_id": row.export_id,
                    "dataset_id": dataset_id,
                    "rel_path": row.rel_path,
                    "error": f"smiles_lookup: {exc}",
                }
            )
            xlsx_rows.append(
                {
                    "index": row.index,
                    "filename": row.filename,
                    "rel_path": row.rel_path,
                    "smiles": "",
                    flat_col: flat_rel,
                    "csv_path": "",
                    "error": str(exc),
                }
            )
            continue

        xlsx_row = {
            "index": row.index,
            "filename": row.filename,
            "rel_path": row.rel_path,
            "smiles": smiles,
            flat_col: flat_rel,
            "csv_path": csv_path,
        }
        xlsx_rows.append(xlsx_row)

        if dry_run:
            successes.append({**xlsx_row, "export_id": row.export_id, "dataset_id": dataset_id})
            continue

        try:
            meta = _write_sdf(
                smiles,
                dst,
                export_id=row.export_id,
                sdf_dim=sdf_dim,
                num_confs=num_confs,
            )
            rec = {
                "index": row.index,
                "export_id": row.export_id,
                "dataset_id": dataset_id,
                "rel_path": row.rel_path,
                "flat_name": flat_name,
                "smiles": smiles,
                "csv_path": csv_path,
                "smiles_source": "curated_group_csv",
                "sdf_dim": sdf_dim,
                **meta,
            }
            successes.append(rec)
        except Exception as exc:
            failures.append(
                {
                    "index": row.index,
                    "export_id": row.export_id,
                    "dataset_id": dataset_id,
                    "rel_path": row.rel_path,
                    "smiles": smiles,
                    "csv_path": csv_path,
                    "error": str(exc),
                }
            )
            xlsx_rows[-1]["error"] = str(exc)

    if not dry_run:
        write_selection_xlsx_outputs(flat_dir, xlsx_src, xlsx_rows, sdf_dim=sdf_dim)
        report = {
            "base_dir": str(base_dir),
            "flat_dir": str(flat_dir),
            "sdf_dim": sdf_dim,
            "smiles_source": "curated_group_csv",
            "n_total": len(selection),
            "n_ok": len(successes),
            "n_fail": len(failures),
            "successes": successes,
            "failures": failures,
        }
        report_path = flat_dir / f"generation_report_{sdf_dim}_from_smiles.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return successes, failures, xlsx_rows


__all__ = [
    "FLAT_DIR_NAMES",
    "SdfDim",
    "default_flat_dir",
    "flat_rel_prefix",
    "generate_flat_from_smiles",
    "load_selection_xlsx",
    "write_selection_xlsx_outputs",
]
