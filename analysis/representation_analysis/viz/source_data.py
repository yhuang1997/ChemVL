from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


def build_tsne_source_rows(
    *,
    smiles: Iterable[str],
    points,
    stage: str,
    dataset: str,
    feature_type: str,
    reducer: str,
    reducer_params: Dict[str, Any],
    descriptor: Optional[str] = None,
    task_id: Optional[int] = None,
    text_target_mode: Optional[str] = None,
    combined_alpha: Optional[float] = None,
    labels: Optional[Iterable[Any]] = None,
    source_refs: Optional[Dict[str, Any]] = None,
    cluster_ids: Optional[Sequence[int]] = None,
    fp_dataset_cluster_ids: Optional[Sequence[int]] = None,
    cliff_mol: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    labels_list = list(labels) if labels is not None else [None] * len(points)
    cliff_list = list(cliff_mol) if cliff_mol is not None else None
    if cliff_list is not None and len(cliff_list) != len(points):
        raise ValueError(f"cliff_mol length {len(cliff_list)} != n_points {len(points)}")
    rows: List[Dict[str, Any]] = []
    for idx, (smi, pt, lb) in enumerate(zip(smiles, points, labels_list)):
        row: Dict[str, Any] = {
            "point_index": int(idx),
            "stage": stage,
            "dataset": dataset,
            "smiles": str(smi),
            "descriptor": descriptor,
            "task_id": task_id,
            "text_target_mode": text_target_mode,
            "feature_type": feature_type,
            "reducer": reducer,
            "reducer_params_json": json.dumps(reducer_params, ensure_ascii=False, sort_keys=True),
            "x": float(pt[0]),
            "y": float(pt[1]),
            "label_or_target": lb,
            "combined_alpha": combined_alpha,
            "source_refs_json": json.dumps(source_refs or {}, ensure_ascii=False, sort_keys=True),
        }
        if cluster_ids is not None:
            row["cluster_id"] = int(cluster_ids[idx])
        if fp_dataset_cluster_ids is not None:
            row["fp_dataset_cluster_id"] = int(fp_dataset_cluster_ids[idx])
        if cliff_list is not None:
            row["cliff_mol"] = int(cliff_list[idx])
        rows.append(row)
    return rows


def write_source_data(rows: List[Dict[str, Any]], out_csv, *, file_format: Optional[str] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    fmt = (file_format or "").strip().lower()
    if not fmt:
        if out_path.suffix == ".parquet":
            fmt = "parquet"
        elif out_path.suffix == ".gz":
            fmt = "csv.gz"
        else:
            fmt = "csv"
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt in {"csv", "csv.gz"}:
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported source data format: {fmt}")
    return df

