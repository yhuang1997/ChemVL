"""Export curated MoleculeACE series / activity-cliff selections to CSV, annotated plots, and SDF."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.interpretability_support.moleculeace_result_bundle import _render_series_preview
from utils.interpretability_support.sdf_2d import write_sdf_2d
from utils.interpretability_support.sdf_3d import write_sdf_3d_multi


def resolve_run_dir(results_root: Path, dataset_id: str, split: str) -> Path:
    return results_root / f"{dataset_id}_{split}"


def default_short_label(dataset_id: str) -> str:
    """CHEMBL204_Ki -> 204."""
    m = re.match(r"CHEMBL(\d+)_", dataset_id, re.IGNORECASE)
    if m:
        return m.group(1)
    return dataset_id.replace("CHEMBL", "").split("_")[0]


def find_interpretability_csv(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("*_interpretability.csv"))
    if not matches:
        return None
    preferred = run_dir / f"{run_dir.name}_interpretability.csv"
    if preferred.is_file():
        return preferred
    return matches[0]


def _parse_json_list(raw: Any) -> List[Any]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    return json.loads(str(raw))


def load_series_candidates(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "series_candidates.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing series_candidates.csv: {path}")
    return pd.read_csv(path)


def load_activity_cliffs(run_dir: Path, delta_key: str) -> pd.DataFrame:
    dir_name = f"activity_cliffs_{delta_key}"
    path = run_dir / dir_name / "activity_cliffs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def validate_index(name: str, index: int, n_rows: int) -> None:
    if index < 0 or index >= n_rows:
        raise IndexError(f"{name} index {index} out of range [0, {n_rows})")


def make_series_export_id(short_label: str, split: str, series_index: int, member_index: int) -> str:
    return f"{short_label}_Ki_{split}_series_{series_index:04d}_{member_index:04d}"


def make_series_csv_stem(short_label: str, split: str, series_index: int) -> str:
    return f"{short_label}_Ki_{split}_series_{series_index:04d}"


def make_cliff_export_id(delta_key: str, cliff_index: int, member_index: int) -> str:
    return f"ac_{delta_key}_{cliff_index:04d}_{member_index:04d}"


def make_cliff_csv_stem(delta_key: str, cliff_index: int) -> str:
    return f"ac_{delta_key}_{cliff_index:04d}"


def expand_members(
    group_row: pd.Series,
    interp_df: pd.DataFrame,
    *,
    group_type: str,
    group_index: int,
    cliff_delta: Optional[str] = None,
) -> pd.DataFrame:
    """Join interpretability rows in molecule_ids order; add export metadata columns."""
    molecule_ids = [int(x) for x in _parse_json_list(group_row.get("molecule_ids"))]
    if not molecule_ids:
        raise ValueError(f"group {group_type} index {group_index}: empty molecule_ids")

    interp_by_idx = interp_df.set_index("molecule_idx", drop=False)
    rows: List[Dict[str, Any]] = []
    for member_index, mol_id in enumerate(molecule_ids):
        if mol_id not in interp_by_idx.index:
            raise KeyError(
                f"group {group_type} index {group_index}: molecule_idx {mol_id} "
                f"not in interpretability CSV"
            )
        rec = interp_by_idx.loc[mol_id]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        out = rec.to_dict()
        out["member_index"] = member_index
        out["group_index"] = group_index
        out["group_type"] = group_type
        out["series_id"] = group_row.get("series_id", "")
        out["scaffold_or_cluster_id"] = group_row.get("scaffold_or_cluster_id", "")
        if cliff_delta is not None:
            out["cliff_delta"] = cliff_delta
            out["cliff_delta_threshold"] = group_row.get("cliff_delta_threshold", "")
        rows.append(out)

    return pd.DataFrame(rows)


def resolve_n_upstream_panels(run_dir: Path) -> int:
    """Infer upstream descriptor panel count from run_manifest.json."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return 0
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    args = data.get("args") or {}
    if not args.get("upstream"):
        return 0
    desc_list = args.get("descriptors") or []
    return len(desc_list) if desc_list else 4


def _cliff_title_suffix(group_row: pd.Series) -> str:
    threshold = group_row.get("cliff_delta_threshold")
    if threshold is None or (isinstance(threshold, float) and pd.isna(threshold)):
        return ""
    return f" | cliff d>={float(threshold):g}"


def render_curated_group_preview(
    members_df: pd.DataFrame,
    run_dir: Path,
    out_path: Path,
    *,
    group_row: pd.Series,
    title: str,
    n_upstream_panels: int,
    title_suffix: str = "",
) -> bool:
    """Render one horizontal group preview (series or activity cliff)."""
    n_members = len(members_df)
    if n_members < 1:
        return False

    gts = members_df["gt"].astype(float).tolist()
    max_delta = float(max(gts) - min(gts)) if gts else None

    contains_ac = group_row.get("contains_ac_pair")
    if contains_ac is None or (isinstance(contains_ac, float) and pd.isna(contains_ac)):
        contains_ac_val: Optional[bool] = None
    else:
        contains_ac_val = bool(contains_ac)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _render_series_preview(
        members_df,
        series_id=str(group_row.get("series_id", "")),
        scaffold=str(group_row.get("scaffold_or_cluster_id", "")),
        n_members=n_members,
        max_delta=max_delta,
        contains_ac=contains_ac_val,
        out_path=str(out_path),
        run_dir=str(run_dir),
        n_upstream_panels=n_upstream_panels,
        panel_width=224,
        title_suffix=title_suffix,
        member_export_ids=members_df["export_id"].astype(str).tolist(),
        title_override=title,
    )
    return out_path.is_file()


@dataclass
class ExportRecord:
    export_id: str
    group_type: str
    group_index: int
    member_index: int
    dataset_id: str
    csv_path: Optional[str] = None
    plot_path: Optional[str] = None
    sdf_2d_path: Optional[str] = None
    sdf_3d_path: Optional[str] = None
    source_plot: Optional[str] = None
    plot_missing: bool = False
    sdf_2d_skipped: bool = False
    sdf_3d_skipped: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class GroupExportResult:
    group_type: str
    group_index: int
    csv_path: str
    n_members: int
    records: List[ExportRecord]
    group_plot_path: Optional[str] = None


def _assign_export_ids(
    members_df: pd.DataFrame,
    *,
    short_label: str,
    split: str,
    group_type: str,
    group_index: int,
    cliff_delta: Optional[str],
) -> pd.DataFrame:
    out = members_df.copy()
    export_ids: List[str] = []
    for _, row in out.iterrows():
        mi = int(row["member_index"])
        if group_type == "series":
            eid = make_series_export_id(short_label, split, group_index, mi)
        else:
            assert cliff_delta is not None
            eid = make_cliff_export_id(cliff_delta, group_index, mi)
        export_ids.append(eid)
    out["export_id"] = export_ids
    return out


def export_group(
    members_df: pd.DataFrame,
    *,
    group_row: pd.Series,
    run_dir: Path,
    out_base: Path,
    short_label: str,
    split: str,
    group_type: str,
    group_index: int,
    dataset_id: str,
    cliff_delta: Optional[str] = None,
    dry_run: bool = False,
    write_sdf_3d: bool = False,
) -> GroupExportResult:
    """Write group CSV, one composite preview per group, and per-member SDF (2D; optional 3D)."""
    members_df = _assign_export_ids(
        members_df,
        short_label=short_label,
        split=split,
        group_type=group_type,
        group_index=group_index,
        cliff_delta=cliff_delta,
    )

    if group_type == "series":
        csv_stem = make_series_csv_stem(short_label, split, group_index)
        csv_dir = out_base / "series" / "csv"
        plots_dir = out_base / "series" / "plots"
        sdf_2d_dir = out_base / "series" / "sdf_2d"
        sdf_3d_dir = out_base / "series" / "sdf_3d"
        title_suffix = ""
    else:
        assert cliff_delta is not None
        csv_stem = make_cliff_csv_stem(cliff_delta, group_index)
        csv_dir = out_base / "activity_cliffs" / cliff_delta / "csv"
        plots_dir = out_base / "activity_cliffs" / cliff_delta / "plots"
        sdf_2d_dir = out_base / "activity_cliffs" / cliff_delta / "sdf_2d"
        sdf_3d_dir = out_base / "activity_cliffs" / cliff_delta / "sdf_3d"
        title_suffix = _cliff_title_suffix(group_row)

    csv_path = csv_dir / f"{csv_stem}.csv"
    group_plot_path = plots_dir / f"{csv_stem}.png"
    records: List[ExportRecord] = []
    n_upstream = resolve_n_upstream_panels(run_dir)

    any_plot_missing = False
    for _, row in members_df.iterrows():
        plot_rel = str(row.get("plot_file", "") or "")
        src_plot = run_dir / plot_rel if plot_rel else None
        if not src_plot or not src_plot.is_file():
            any_plot_missing = True

    group_plot_ok = False
    if not dry_run:
        csv_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        members_df.to_csv(csv_path, index=False, float_format="%.6f")
        for export_id in members_df["export_id"].astype(str):
            stale = plots_dir / f"{export_id}.png"
            if stale.is_file():
                stale.unlink()
        group_plot_ok = render_curated_group_preview(
            members_df,
            run_dir,
            group_plot_path,
            group_row=group_row,
            title=csv_stem,
            n_upstream_panels=n_upstream,
            title_suffix=title_suffix,
        )

    for _, row in members_df.iterrows():
        export_id = str(row["export_id"])
        plot_rel = str(row.get("plot_file", "") or "")
        src_plot = run_dir / plot_rel if plot_rel else None
        dst_sdf_2d = sdf_2d_dir / f"{export_id}.sdf"
        dst_sdf_3d = sdf_3d_dir / f"{export_id}.sdf"
        smiles = str(row["smiles"])

        rec = ExportRecord(
            export_id=export_id,
            group_type=group_type,
            group_index=group_index,
            member_index=int(row["member_index"]),
            dataset_id=dataset_id,
            csv_path=str(csv_path),
            plot_path=str(group_plot_path),
            sdf_2d_path=str(dst_sdf_2d),
            sdf_3d_path=str(dst_sdf_3d) if write_sdf_3d else None,
            source_plot=str(src_plot) if src_plot else None,
        )

        if dry_run:
            if not src_plot or not src_plot.is_file():
                rec.plot_missing = True
            records.append(rec)
            continue

        if not group_plot_ok:
            rec.plot_missing = True
            rec.warnings.append("group_plot_failed")
        elif any_plot_missing:
            rec.warnings.append("some_source_plots_missing")

        if not write_sdf_2d(smiles, dst_sdf_2d):
            rec.sdf_2d_skipped = True
            rec.warnings.append("sdf_2d_write_failed")

        if write_sdf_3d:
            try:
                write_sdf_3d_multi(smiles, dst_sdf_3d, export_id=export_id)
            except Exception:
                rec.sdf_3d_skipped = True
                rec.warnings.append("sdf_3d_write_failed")

        records.append(rec)

    return GroupExportResult(
        group_type=group_type,
        group_index=group_index,
        csv_path=str(csv_path),
        n_members=len(members_df),
        records=records,
        group_plot_path=str(group_plot_path),
    )


def export_series_selection(
    run_dir: Path,
    interp_df: pd.DataFrame,
    series_df: pd.DataFrame,
    series_index: int,
    *,
    out_base: Path,
    short_label: str,
    split: str,
    dataset_id: str,
    dry_run: bool = False,
    write_sdf_3d: bool = False,
) -> GroupExportResult:
    validate_index("series", series_index, len(series_df))
    row = series_df.iloc[series_index]
    members = expand_members(
        row,
        interp_df,
        group_type="series",
        group_index=series_index,
    )
    return export_group(
        members,
        group_row=row,
        run_dir=run_dir,
        out_base=out_base,
        short_label=short_label,
        split=split,
        group_type="series",
        group_index=series_index,
        dataset_id=dataset_id,
        dry_run=dry_run,
        write_sdf_3d=write_sdf_3d,
    )


def export_cliff_selection(
    run_dir: Path,
    interp_df: pd.DataFrame,
    cliff_df: pd.DataFrame,
    cliff_index: int,
    *,
    delta_key: str,
    out_base: Path,
    short_label: str,
    split: str,
    dataset_id: str,
    dry_run: bool = False,
    write_sdf_3d: bool = False,
) -> GroupExportResult:
    validate_index(f"activity_cliff {delta_key}", cliff_index, len(cliff_df))
    row = cliff_df.iloc[cliff_index]
    members = expand_members(
        row,
        interp_df,
        group_type="activity_cliff",
        group_index=cliff_index,
        cliff_delta=delta_key,
    )
    return export_group(
        members,
        group_row=row,
        run_dir=run_dir,
        out_base=out_base,
        short_label=short_label,
        split=split,
        group_type="activity_cliff",
        group_index=cliff_index,
        dataset_id=dataset_id,
        cliff_delta=delta_key,
        dry_run=dry_run,
        write_sdf_3d=write_sdf_3d,
    )


def manifest_entry(rec: ExportRecord, group_plot_path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "export_id": rec.export_id,
        "dataset_id": rec.dataset_id,
        "group_type": rec.group_type,
        "group_index": rec.group_index,
        "member_index": rec.member_index,
        "csv_path": rec.csv_path,
        "plot_path": rec.plot_path,
        "group_plot_path": group_plot_path or rec.plot_path,
        "sdf_2d_path": rec.sdf_2d_path,
        "sdf_3d_path": rec.sdf_3d_path,
        "source_plot": rec.source_plot,
        "plot_missing": rec.plot_missing,
        "sdf_2d_skipped": rec.sdf_2d_skipped,
        "sdf_3d_skipped": rec.sdf_3d_skipped,
        "warnings": rec.warnings,
    }


def build_manifest(groups: List[GroupExportResult]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for grp in groups:
        for rec in grp.records:
            entries.append(manifest_entry(rec, group_plot_path=grp.group_plot_path))
    return entries
