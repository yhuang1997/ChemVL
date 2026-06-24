"""Build standardized MoleculeACE interpretability bundle outputs per dataset run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from utils.splitter import generate_scaffold

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit import DataStructs

    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


@dataclass
class SeriesMiningConfig:
    scope: str = "gradcam_sample_only"
    grouping: str = "murcko_scaffold"
    fingerprint: str = "morgan_r2_2048"
    min_members: int = 2
    ac_tanimoto_min: float = 0.85
    ac_delta_activity_min: float = 1.0
    # Keep molecules with |pred - gt| <= this for series mining only (None = no filter).
    series_max_abs_error: Optional[float] = None
    # Murcko-scaffold groups with any pair |gt_i - gt_j| >= t are exported per threshold.
    cliff_delta_thresholds: Tuple[float, ...] = (2.0, 3.0)
    cliff_score_col: str = "gt"

    def to_manifest_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "scope": self.scope,
            "grouping": self.grouping,
            "fingerprint": self.fingerprint,
            "min_members": self.min_members,
            "ac_tanimoto_min": self.ac_tanimoto_min,
            "ac_delta_activity_min": self.ac_delta_activity_min,
        }
        out["series_max_abs_error"] = self.series_max_abs_error
        out["cliff_delta_thresholds"] = list(self.cliff_delta_thresholds)
        out["cliff_score_col"] = self.cliff_score_col
        return out


def filter_series_candidate_pool(
    df: pd.DataFrame,
    max_abs_error: float,
) -> Tuple[pd.DataFrame, int]:
    """Return rows with |pred - label| <= max_abs_error and count of excluded rows."""
    if "abs_error" in df.columns:
        err = df["abs_error"].astype(float)
    else:
        err = (df["pred"].astype(float) - df["gt"].astype(float)).abs()
    mask = err <= float(max_abs_error)
    excluded = int((~mask).sum())
    return df.loc[mask].copy().reset_index(drop=True), excluded


def _json_list(values: List[Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _json_dict(values: Dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _split_counts(membership_splits: List[str]) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for s in membership_splits:
        if s in counts:
            counts[s] += 1
    return counts


def _scaffold_id(smiles: str) -> Tuple[str, str]:
    """Return (series_id, scaffold_or_cluster_id)."""
    if not _RDKIT_OK:
        return "scaffold_missing_rdkit", "missing_rdkit"
    try:
        scaf = generate_scaffold(smiles, include_chirality=True)
        if not scaf:
            h = hashlib.md5(smiles.encode()).hexdigest()[:8]
            return f"invalid_{h}", "invalid"
        h = hashlib.md5(scaf.encode()).hexdigest()[:8]
        return f"scaffold_{h}", scaf
    except Exception:
        h = hashlib.md5(smiles.encode()).hexdigest()[:8]
        return f"invalid_{h}", "invalid"


def _morgan_fp(smiles: str):
    if not _RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _pairwise_tanimoto(smiles_list: List[str]) -> Tuple[Optional[float], Optional[float]]:
    fps = [_morgan_fp(s) for s in smiles_list]
    if any(fp is None for fp in fps):
        return None, None
    sims: List[float] = []
    for i, j in combinations(range(len(fps)), 2):
        sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    if not sims:
        return None, None
    return float(np.mean(sims)), float(np.min(sims))


def _ac_pair_info(
    df: pd.DataFrame,
    ac_tanimoto_min: float,
    ac_delta_activity_min: float,
) -> Tuple[Optional[bool], Optional[str]]:
    n = len(df)
    if n < 2:
        return None, None
    if not _RDKIT_OK:
        return None, None

    idxs = df["molecule_idx"].tolist()
    smiles_list = df["smiles"].astype(str).tolist()
    gts = df["gt"].astype(float).tolist()
    fps = [_morgan_fp(s) for s in smiles_list]
    if any(fp is None for fp in fps):
        return None, None

    best_pair: Optional[Tuple[int, int, float]] = None
    found = False
    for a, b in combinations(range(n), 2):
        tani = float(DataStructs.TanimotoSimilarity(fps[a], fps[b]))
        delta = abs(gts[a] - gts[b])
        if tani >= ac_tanimoto_min and delta >= ac_delta_activity_min:
            found = True
            if best_pair is None or delta > best_pair[2]:
                best_pair = (idxs[a], idxs[b], delta)

    if not found:
        return False, None
    assert best_pair is not None
    return True, _json_list([best_pair[0], best_pair[1]])


def _activity_values(df: pd.DataFrame, score_col: str) -> List[float]:
    if score_col not in df.columns:
        raise KeyError(f"score column {score_col!r} not in dataframe")
    return df[score_col].astype(float).tolist()


def scaffold_max_activity_delta(df: pd.DataFrame, score_col: str = "gt") -> Optional[float]:
    """Max pairwise |score_i - score_j| within a scaffold group."""
    values = _activity_values(df, score_col)
    if len(values) < 2:
        return None
    return float(max(abs(a - b) for a, b in combinations(values, 2)))


def scaffold_has_cliff(df: pd.DataFrame, threshold: float, score_col: str = "gt") -> bool:
    """True if any pair in the group has |score_i - score_j| >= threshold."""
    max_delta = scaffold_max_activity_delta(df, score_col)
    if max_delta is None:
        return False
    return max_delta >= float(threshold)


def _scaffold_cliff_pair_info(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "gt",
) -> Tuple[Optional[bool], Optional[str], Optional[float]]:
    """Return (qualifies, best_pair_ids_json, max_delta) for scaffold cliff export."""
    n = len(df)
    if n < 2:
        return None, None, None
    idxs = df["molecule_idx"].astype(int).tolist()
    values = _activity_values(df, score_col)
    best_pair: Optional[Tuple[int, int, float]] = None
    for a, b in combinations(range(n), 2):
        delta = abs(values[a] - values[b])
        if delta >= float(threshold):
            if best_pair is None or delta > best_pair[2]:
                best_pair = (idxs[a], idxs[b], delta)
    if best_pair is None:
        max_delta = scaffold_max_activity_delta(df, score_col)
        return False, None, max_delta
    return True, _json_list([best_pair[0], best_pair[1]]), float(best_pair[2])


def _assign_scaffold_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["series_id"] = ""
    out["scaffold_or_cluster_id"] = ""
    for i, smi in enumerate(out["smiles"].astype(str)):
        sid, scaf = _scaffold_id(smi)
        out.at[i, "series_id"] = sid
        out.at[i, "scaffold_or_cluster_id"] = scaf
    return out


def _cliff_dir_name(threshold: float) -> str:
    t = float(threshold)
    if t == int(t):
        return f"activity_cliffs_delta{int(t)}"
    return f"activity_cliffs_delta{t:g}".replace(".", "p")


def build_activity_cliff_exports(
    df: pd.DataFrame,
    dataset_id: str,
    run_dir: str,
    mining: SeriesMiningConfig,
    *,
    n_upstream_panels: int = 0,
    panel_width: int = 224,
) -> Dict[str, Any]:
    """
    Export Murcko-scaffold groups with |gt_i - gt_j| >= each cliff threshold.

    Writes per threshold: ``activity_cliffs_delta{t}/activity_cliffs.csv`` and
    ``activity_cliffs_delta{t}/previews/cliff_XXXX.png``.
    """
    if not mining.cliff_delta_thresholds:
        return {"activity_cliffs": {}, "n_activity_cliffs": {}}

    score_col = mining.cliff_score_col
    df = _assign_scaffold_columns(df)
    result_dirs: Dict[str, str] = {}
    result_csvs: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for threshold in mining.cliff_delta_thresholds:
        t = float(threshold)
        dir_name = _cliff_dir_name(t)
        cliff_dir = os.path.join(run_dir, dir_name)
        previews_dir = os.path.join(cliff_dir, "previews")
        os.makedirs(previews_dir, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        cliff_counter = 0
        for series_id, grp in df.groupby("series_id", sort=True):
            n_members = len(grp)
            if n_members < mining.min_members:
                continue

            qualifies, selected_pair, pair_delta = _scaffold_cliff_pair_info(
                grp, t, score_col=score_col
            )
            if not qualifies:
                continue

            smiles_list = grp["smiles"].astype(str).tolist()
            labels = grp["gt"].astype(float).tolist()
            preds = grp["pred"].astype(float).tolist()
            max_delta = scaffold_max_activity_delta(grp, score_col)

            if "membership_split" in grp.columns:
                membership_splits = grp["membership_split"].astype(str).tolist()
            else:
                membership_splits = [str(grp["split"].iloc[0])] * n_members

            preview_name = f"cliff_{cliff_counter:04d}.png"
            preview_rel = os.path.join(dir_name, "previews", preview_name)
            preview_abs = os.path.join(run_dir, preview_rel)

            title_suffix = f" | cliff d>={t:g}"
            _render_series_preview(
                grp,
                series_id=str(series_id),
                scaffold=str(grp["scaffold_or_cluster_id"].iloc[0]),
                n_members=n_members,
                max_delta=max_delta,
                contains_ac=True,
                out_path=preview_abs,
                run_dir=run_dir,
                n_upstream_panels=n_upstream_panels,
                panel_width=panel_width,
                title_suffix=title_suffix,
            )

            rows.append(
                {
                    "series_id": series_id,
                    "dataset_id": dataset_id,
                    "scaffold_or_cluster_id": str(grp["scaffold_or_cluster_id"].iloc[0]),
                    "cliff_delta_threshold": t,
                    "molecule_ids": _json_list(grp["molecule_idx"].astype(int).tolist()),
                    "pool_indices": _json_list(grp["pool_index"].astype(int).tolist()),
                    "smiles_list": _json_list(smiles_list),
                    "labels": _json_list(labels),
                    "predictions": _json_list(preds),
                    "membership_splits": _json_list(membership_splits),
                    "split_counts": _json_dict(_split_counts(membership_splits)),
                    "n_members": int(n_members),
                    "max_delta_activity": max_delta,
                    "cliff_pair_delta": pair_delta,
                    "selected_pair_ids": selected_pair,
                    "preview_path": preview_rel,
                }
            )
            cliff_counter += 1

        csv_path = os.path.join(cliff_dir, "activity_cliffs.csv")
        cliff_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[
                "series_id",
                "dataset_id",
                "scaffold_or_cluster_id",
                "cliff_delta_threshold",
                "molecule_ids",
                "pool_indices",
                "smiles_list",
                "labels",
                "predictions",
                "membership_splits",
                "split_counts",
                "n_members",
                "max_delta_activity",
                "cliff_pair_delta",
                "selected_pair_ids",
                "preview_path",
            ]
        )
        cliff_df.to_csv(csv_path, index=False, float_format="%.6f")
        result_dirs[dir_name] = cliff_dir
        result_csvs[dir_name] = csv_path
        counts[dir_name] = int(len(cliff_df))

    return {
        "activity_cliffs": result_dirs,
        "activity_cliffs_csv": result_csvs,
        "n_activity_cliffs": counts,
    }


def build_series_candidates(
    df: pd.DataFrame,
    dataset_id: str,
    run_dir: str,
    mining: SeriesMiningConfig,
    *,
    n_upstream_panels: int = 0,
    panel_width: int = 224,
) -> pd.DataFrame:
    """Group interpretability sample by Murcko scaffold; write series_previews."""
    previews_dir = os.path.join(run_dir, "series_previews")
    os.makedirs(previews_dir, exist_ok=True)

    df = _assign_scaffold_columns(df)

    rows: List[Dict[str, Any]] = []
    series_counter = 0
    for series_id, grp in df.groupby("series_id", sort=True):
        n_members = len(grp)
        if n_members < mining.min_members:
            continue

        smiles_list = grp["smiles"].astype(str).tolist()
        tani_mean, tani_min = _pairwise_tanimoto(smiles_list)
        labels = grp["gt"].astype(float).tolist()
        preds = grp["pred"].astype(float).tolist()
        max_delta = float(max(labels) - min(labels)) if labels else None

        contains_ac, selected_pair = _ac_pair_info(grp, mining.ac_tanimoto_min, mining.ac_delta_activity_min)

        if "membership_split" in grp.columns:
            membership_splits = grp["membership_split"].astype(str).tolist()
        else:
            membership_splits = [str(grp["split"].iloc[0])] * n_members

        preview_name = f"series_{series_counter:04d}.png"
        preview_rel = os.path.join("series_previews", preview_name)
        preview_abs = os.path.join(run_dir, preview_rel)

        _render_series_preview(
            grp,
            series_id=str(series_id),
            scaffold=str(grp["scaffold_or_cluster_id"].iloc[0]),
            n_members=n_members,
            max_delta=max_delta,
            contains_ac=contains_ac,
            out_path=preview_abs,
            run_dir=run_dir,
            n_upstream_panels=n_upstream_panels,
            panel_width=panel_width,
        )

        rows.append(
            {
                "series_id": series_id,
                "dataset_id": dataset_id,
                "scaffold_or_cluster_id": str(grp["scaffold_or_cluster_id"].iloc[0]),
                "molecule_ids": _json_list(grp["molecule_idx"].astype(int).tolist()),
                "pool_indices": _json_list(grp["pool_index"].astype(int).tolist()),
                "smiles_list": _json_list(smiles_list),
                "labels": _json_list(labels),
                "predictions": _json_list(preds),
                "membership_splits": _json_list(membership_splits),
                "split_counts": _json_dict(_split_counts(membership_splits)),
                "n_members": int(n_members),
                "pairwise_tanimoto_mean": tani_mean,
                "pairwise_tanimoto_min": tani_min,
                "max_delta_activity": max_delta,
                "contains_ac_pair": contains_ac,
                "selected_pair_ids": selected_pair,
                "preview_path": preview_rel,
            }
        )
        series_counter += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "series_id",
                "dataset_id",
                "scaffold_or_cluster_id",
                "molecule_ids",
                "pool_indices",
                "smiles_list",
                "labels",
                "predictions",
                "membership_splits",
                "split_counts",
                "n_members",
                "pairwise_tanimoto_mean",
                "pairwise_tanimoto_min",
                "max_delta_activity",
                "contains_ac_pair",
                "selected_pair_ids",
                "preview_path",
            ]
        )
    return pd.DataFrame(rows)


def _draw_corner_label(im: Image.Image, label: str) -> None:
    """Draw label on bottom-right of an RGB image (in-place)."""
    draw = ImageDraw.Draw(im)
    w, h = im.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font is not None:
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
    else:
        tw, th = len(label) * 6, 11
    pad = 4
    box_h = th + 2 * pad
    box_w = min(tw + 2 * pad, w)
    x0 = w - box_w
    y0 = h - box_h
    draw.rectangle([x0, y0, w, h], fill=(255, 255, 255))
    draw.text((x0 + pad, y0 + pad), label, fill=(0, 0, 0), font=font)


def _render_series_preview(
    grp: pd.DataFrame,
    *,
    series_id: str,
    scaffold: str,
    n_members: int,
    max_delta: Optional[float],
    contains_ac: Optional[bool],
    out_path: str,
    run_dir: str,
    n_upstream_panels: int,
    panel_width: int,
    title_suffix: str = "",
    member_export_ids: Optional[List[str]] = None,
    title_override: Optional[str] = None,
) -> None:
    """Lightweight horizontal preview for one series candidate."""
    col_w, col_h = 240, 320
    canvas = Image.new("RGB", (col_w * n_members, col_h + 40), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title_override:
        title = title_override
    else:
        ac_str = "null" if contains_ac is None else str(contains_ac)
        delta_str = f"{max_delta:.2f}" if max_delta is not None else "null"
        title = f"{series_id} | n={n_members} | d_act={delta_str} | ac={ac_str}{title_suffix}"
    draw.text((4, 2), title[: col_w * n_members - 8], fill=(0, 0, 0))

    for col, (_, row) in enumerate(grp.iterrows()):
        x0 = col * col_w
        y0 = 36
        mol_idx = int(row["molecule_idx"])
        gt = float(row["gt"])
        pred = float(row["pred"])
        plot_rel = str(row.get("plot_file", ""))
        plot_abs = os.path.join(run_dir, plot_rel) if plot_rel else ""

        # Structure depiction
        struct_img = _draw_structure(str(row["smiles"]), size=(col_w - 8, 100))
        canvas.paste(struct_img, (x0 + 4, y0))

        # Grad-CAM crop (rightmost panel_width of molecule plot row)
        cam_y = y0 + 104
        cam_img = _crop_gradcam_panel(plot_abs, n_upstream_panels, panel_width)
        if member_export_ids is not None and col < len(member_export_ids):
            _draw_corner_label(cam_img, str(member_export_ids[col]))
        canvas.paste(cam_img, (x0 + 4, cam_y))

        mem = str(row.get("membership_split", row.get("split", "")))
        mem_tag = f" [{mem}]" if mem else ""
        caption = f"#{mol_idx}{mem_tag} gt={gt:.2f} pred={pred:.2f} |e|={abs(gt - pred):.2f}"
        draw.text((x0 + 4, cam_y + panel_width + 4), caption[: col_w - 10], fill=(30, 30, 30))

    canvas.save(out_path)


def _draw_structure(smiles: str, size: Tuple[int, int]) -> Image.Image:
    w, h = size
    if not _RDKIT_OK:
        img = Image.new("RGB", size, (220, 220, 220))
        ImageDraw.Draw(img).text((4, 4), "no RDKit", fill=(0, 0, 0))
        return img
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        img = Image.new("RGB", size, (220, 220, 220))
        ImageDraw.Draw(img).text((4, 4), "bad SMILES", fill=(0, 0, 0))
        return img
    try:
        raw = Draw.MolToImage(mol, size=size)
        return raw.convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (220, 220, 220))
        ImageDraw.Draw(img).text((4, 4), "draw fail", fill=(0, 0, 0))
        return img


def _crop_gradcam_panel(plot_abs: str, n_upstream_panels: int, panel_width: int) -> Image.Image:
    """Rightmost panel_width strip from per-molecule plot (finetuned Grad-CAM)."""
    if not plot_abs or not os.path.isfile(plot_abs):
        img = Image.new("RGB", (panel_width, panel_width), (200, 200, 200))
        ImageDraw.Draw(img).text((8, 100), "plot_missing", fill=(0, 0, 0))
        return img
    im = Image.open(plot_abs).convert("RGB")
    w, h = im.size
    # layout: original | upstream*n | finetuned
    n_cols = 1 + n_upstream_panels + 1
    if w >= n_cols * panel_width:
        left = w - panel_width
        return im.crop((left, 0, w, min(h, panel_width))).resize((panel_width, panel_width))
    return im.resize((panel_width, panel_width))


def build_task_summary(
    *,
    dataset_id: str,
    split: str,
    checkpoint_path: str,
    run_dir: str,
    n_molecules: int,
    split_metrics: Dict[str, Optional[float]],
    gradcam_available: bool,
    timestamp_utc: Optional[str] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Dict[str, Any]:
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "split": split,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "run_dir": os.path.abspath(run_dir),
        "n_molecules": int(n_molecules),
        "gradcam_available": bool(gradcam_available),
        "timestamp_utc": ts,
    }
    if n_train is not None:
        summary["n_train"] = int(n_train)
    if n_val is not None:
        summary["n_val"] = int(n_val)
    if n_test is not None:
        summary["n_test"] = int(n_test)
    for key, val in split_metrics.items():
        summary[key] = val
    return summary


def write_task_summary(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_task_summary(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "task_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def patch_task_summary_cliff_counts(run_dir: str, cliff_info: Dict[str, Any]) -> None:
    """Merge activity-cliff counts into an existing task_summary.json."""
    path = os.path.join(run_dir, "task_summary.json")
    summary = load_task_summary(run_dir) if os.path.isfile(path) else {}
    for dir_name, count in cliff_info.get("n_activity_cliffs", {}).items():
        summary[f"n_{dir_name}"] = int(count)
    write_task_summary(path, summary)


def series_mining_from_manifest(manifest: Dict[str, Any]) -> SeriesMiningConfig:
    sm = manifest.get("series_mining") or {}
    thresholds = sm.get("cliff_delta_thresholds")
    if thresholds is None:
        cliff_thresholds = (2.0, 3.0)
    else:
        cliff_thresholds = tuple(float(t) for t in thresholds)
    return SeriesMiningConfig(
        scope=str(sm.get("scope", "gradcam_sample_only")),
        grouping=str(sm.get("grouping", "murcko_scaffold")),
        fingerprint=str(sm.get("fingerprint", "morgan_r2_2048")),
        min_members=int(sm.get("min_members", 2)),
        ac_tanimoto_min=float(sm.get("ac_tanimoto_min", 0.85)),
        ac_delta_activity_min=float(sm.get("ac_delta_activity_min", 1.0)),
        series_max_abs_error=sm.get("series_max_abs_error"),
        cliff_delta_thresholds=cliff_thresholds,
        cliff_score_col=str(sm.get("cliff_score_col", "gt")),
    )


def export_activity_cliffs_for_run(
    run_dir: str,
    df: pd.DataFrame,
    dataset_id: str,
    mining: SeriesMiningConfig,
    *,
    n_upstream_panels: int = 0,
    update_task_summary: bool = True,
) -> Dict[str, Any]:
    """Build activity-cliff folders from an interpretability dataframe (no Grad-CAM)."""
    series_source = df
    if mining.series_max_abs_error is not None:
        series_source, _ = filter_series_candidate_pool(df, mining.series_max_abs_error)
    cliff_info = build_activity_cliff_exports(
        series_source,
        dataset_id,
        run_dir,
        mining,
        n_upstream_panels=n_upstream_panels,
    )
    if update_task_summary:
        patch_task_summary_cliff_counts(run_dir, cliff_info)
    return cliff_info


def build_bundle(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    split: str,
    run_dir: str,
    checkpoint_path: str,
    split_metrics: Dict[str, Optional[float]],
    gradcam_available: bool,
    mining: SeriesMiningConfig,
    n_molecules: int,
    n_upstream_panels: int = 0,
    timestamp_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Write task_summary.json, series_candidates.csv, series_previews/; return paths + mining config."""
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

    n_train = n_val = n_test = None
    if split == "all" and "membership_split" in df.columns:
        counts = _split_counts(df["membership_split"].astype(str).tolist())
        n_train, n_val, n_test = counts["train"], counts["val"], counts["test"]

    task_summary = build_task_summary(
        dataset_id=dataset_id,
        split=split,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        n_molecules=int(n_molecules),
        split_metrics=split_metrics,
        gradcam_available=gradcam_available,
        timestamp_utc=ts,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    task_summary["n_molecules_in_csv"] = int(len(df))
    series_source = df
    if mining.series_max_abs_error is not None:
        series_source, n_excluded = filter_series_candidate_pool(df, mining.series_max_abs_error)
        task_summary["series_max_abs_error"] = float(mining.series_max_abs_error)
        task_summary["n_molecules_series_pool"] = int(len(series_source))
        task_summary["n_molecules_excluded_series_filter"] = int(n_excluded)
    else:
        task_summary["series_max_abs_error"] = None
        task_summary["n_molecules_series_pool"] = int(len(df))
        task_summary["n_molecules_excluded_series_filter"] = 0

    task_summary_path = os.path.join(run_dir, "task_summary.json")
    write_task_summary(task_summary_path, task_summary)

    series_df = build_series_candidates(
        series_source,
        dataset_id,
        run_dir,
        mining,
        n_upstream_panels=n_upstream_panels,
    )
    series_csv_path = os.path.join(run_dir, "series_candidates.csv")
    series_df.to_csv(series_csv_path, index=False, float_format="%.6f")

    cliff_info = build_activity_cliff_exports(
        series_source,
        dataset_id,
        run_dir,
        mining,
        n_upstream_panels=n_upstream_panels,
    )
    for dir_name, count in cliff_info.get("n_activity_cliffs", {}).items():
        task_summary[f"n_{dir_name}"] = int(count)

    out: Dict[str, Any] = {
        "task_summary": task_summary_path,
        "series_candidates": series_csv_path,
        "series_previews_dir": os.path.join(run_dir, "series_previews"),
        "series_mining": mining.to_manifest_dict(),
        "n_series_candidates": int(len(series_df)),
    }
    out.update(cliff_info)
    write_task_summary(task_summary_path, task_summary)
    return out


def _evaluate_one_split(
    model: Any,
    cfg: Dict[str, Any],
    names: np.ndarray,
    labels: np.ndarray,
    smiles: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split: str,
    device: str,
    *,
    train_dl: Any = None,
    val_dl: Any = None,
    test_dl: Any = None,
) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    """Evaluate a single split; optionally reuse pre-built dataloaders."""
    from functools import partial

    from models.clip_model_utils import evaluate_on_multitask

    split_loaders = {"train": 0, "val": 1, "test": 2}
    idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    r2_key = f"{split}_r2"
    rmse_key = f"{split}_rmse"
    try:
        if train_dl is None or val_dl is None or test_dl is None:
            from utils.finetune_utils import get_dataloader

            train_dl, val_dl, test_dl = get_dataloader(
                cfg, names, labels, smiles, train_idx, val_idx, test_idx
            )
        loader = [train_dl, val_dl, test_dl][split_loaders[split]]
        split_smiles = np.array(smiles)[idx_map[split]]

        finetune_strategy = (cfg.get("training") or {}).get("finetune_strategy", "")
        representation = cfg["dataset"].get("representation", "image")
        if finetune_strategy == "prior_guided_tuning" or representation == "graph":
            inner = model.module if hasattr(model, "module") else model
            inner.forward = partial(inner.forward, smiles=split_smiles)

        task_type = cfg["dataset"]["task_type"]
        eval_with_tta = representation == "image"
        metrics, _ = evaluate_on_multitask(
            model=model,
            data_loader=loader,
            device=device,
            task_type=task_type,
            return_data_dict=True,
            tta=eval_with_tta,
        )
        return {
            r2_key: float(metrics.get("R2", float("nan"))),
            rmse_key: float(metrics.get("RMSE", float("nan"))),
        }, None
    except Exception as exc:
        return {r2_key: None, rmse_key: None}, str(exc)


def evaluate_split_metrics(
    model: Any,
    cfg: Dict[str, Any],
    names: np.ndarray,
    labels: np.ndarray,
    smiles: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split: str,
    device: str,
) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    """
    Run evaluate_on_multitask on the full requested split.
    Returns ({split}_r2, {split}_rmse), error_message.
    """
    return _evaluate_one_split(
        model, cfg, names, labels, smiles, train_idx, val_idx, test_idx, split, device
    )


def evaluate_all_splits_metrics(
    model: Any,
    cfg: Dict[str, Any],
    names: np.ndarray,
    labels: np.ndarray,
    smiles: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[str]]]:
    """
    Evaluate train, val, and test splits separately.
    Returns merged metrics dict and per-split error messages (null if ok).
    """
    from utils.finetune_utils import get_dataloader

    metrics: Dict[str, Optional[float]] = {}
    errors: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    try:
        train_dl, val_dl, test_dl = get_dataloader(
            cfg, names, labels, smiles, train_idx, val_idx, test_idx
        )
    except Exception as exc:
        for sp in ("train", "val", "test"):
            metrics[f"{sp}_r2"] = None
            metrics[f"{sp}_rmse"] = None
            errors[sp] = str(exc)
        return metrics, errors

    for sp in ("train", "val", "test"):
        part, err = _evaluate_one_split(
            model,
            cfg,
            names,
            labels,
            smiles,
            train_idx,
            val_idx,
            test_idx,
            sp,
            device,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
        )
        metrics.update(part)
        errors[sp] = err
    return metrics, errors
