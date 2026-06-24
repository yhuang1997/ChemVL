from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.representation_analysis.clustering.reducer import reduce_features
from analysis.representation_analysis.core.io import resolve_output_dir, write_manifest
from analysis.representation_analysis.features.feature_builder import build_features_from_preset
from analysis.representation_analysis.metrics.embedding_neighbor_metrics import (
    label_smoothness_at_k,
    neighborhood_preservation_at_k,
)
from analysis.representation_analysis.metrics.moleculeace_cliff import (
    load_cliff_mol_flags,
    resolve_moleculeace_raw_csv_from_processed,
)
from analysis.representation_analysis.viz.plots import (
    plot_tsne_classification,
    plot_tsne_regression,
    plot_tsne_regression_ac_highlight,
    plot_tsne_regression_alpha_montage,
)
from analysis.representation_analysis.viz.source_data import build_tsne_source_rows, write_source_data


def load_preset(path: Path) -> Dict[str, Any]:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema_version", 1) != 1:
        raise ValueError("Only schema_version=1 is supported")
    if str(p.get("mode", "")).lower() != "downstream":
        raise ValueError("Only mode=downstream is supported in this public bundle")
    if "common" not in p or "downstream" not in p:
        raise ValueError("Preset missing `common` or `downstream`")
    return p


def _feature_combo(image_features: np.ndarray, text_features: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * image_features + alpha * text_features


def _preset_flag_disabled(value: Any) -> bool:
    if value is False or value == 0:
        return True
    if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
        return True
    return False


def _resolve_downstream_cliff_flags(dcfg: Dict[str, Any], smiles: List[str]) -> Optional[np.ndarray]:
    if not bool(dcfg.get("overlay_ac_mols", False)):
        return None
    csv_path = str(dcfg.get("downstream_csv_file") or "").strip()
    if not csv_path:
        warnings.warn("downstream: downstream_csv_file is empty; cannot load cliff_mol.", stacklevel=2)
        return None
    rawp = resolve_moleculeace_raw_csv_from_processed(csv_path)
    if rawp is None:
        warnings.warn(
            "downstream: could not infer MoleculeACE raw CSV from downstream_csv_file; cliff_mol unavailable.",
            stacklevel=2,
        )
        return None
    flags = load_cliff_mol_flags(rawp, smiles)
    if flags is None:
        warnings.warn(f"downstream: could not read cliff_mol from {rawp}.", stacklevel=2)
    return flags


def run_preset(preset_path: Path) -> int:
    preset = load_preset(preset_path)
    out_dir = resolve_output_dir(preset)
    out_dir.mkdir(parents=True, exist_ok=True)

    outs = preset.get("outputs") or {}
    source_name = str(outs.get("source_csv", "tsne_source_data.csv"))
    source_path = out_dir / source_name
    source_format = outs.get("source_format")

    run_data = build_features_from_preset(preset)
    all_rows: List[Dict[str, Any]] = []
    plot_paths: List[Path] = []
    regression_metric_rows: List[Dict[str, Any]] = []
    dcfg = preset.get("downstream") or {}

    for rec in run_data["records"]:
        smiles = rec["smiles"]
        targets = rec["targets"]
        montage_panels: List[Tuple[np.ndarray, float]] = []
        cliff_flags: Optional[np.ndarray] = None
        if str(dcfg.get("task_type", "classification")).lower() == "regression":
            cliff_flags = _resolve_downstream_cliff_flags(dcfg, list(smiles))

        for alpha_idx, alpha in enumerate(rec["combined_alpha_list"]):
            alpha_tag = f"a{alpha_idx}_{str(alpha).replace('.', 'p').replace('-', 'm')}"
            combined = _feature_combo(rec["image_features"], rec["text_features"], float(alpha))
            points, reducer_name, reducer_params = reduce_features(combined, run_data["reducer_cfg"])
            task_or_desc = rec.get("descriptor") or f"task{rec.get('task_id', 0)}"
            title = f"{rec['dataset']}_{task_or_desc}_{alpha_tag}"
            out_plot = out_dir / "plots" / "tsne" / title

            d_task = str(dcfg.get("task_type", "classification")).lower()
            db = None
            if d_task == "regression":
                cb_lbl = dcfg.get("tsne_colorbar_label")
                cb_str = str(cb_lbl).strip() if isinstance(cb_lbl, str) else None
                plot_tsne_regression(
                    points,
                    targets,
                    out_plot,
                    title=title,
                    colorbar_label=cb_str or None,
                )
                montage_panels.append((np.array(points, dtype=np.float64, copy=True), float(alpha)))
                tm = dcfg.get("tsne_metrics") or {}
                if not _preset_flag_disabled(tm.get("enabled", True)):
                    yv = np.asarray(targets, dtype=np.float64).reshape(-1)
                    ks = [int(x) for x in (tm.get("neighbor_ks") or [5, 10, 20])]
                    emb_hd = np.asarray(combined, dtype=np.float64)
                    ls_space = str(tm.get("ls_neighbor_space", "fused_hd")).strip().lower()
                    if ls_space not in ("fused_hd", "tsne2d"):
                        ls_space = "fused_hd"
                    ls_emb = np.asarray(points, dtype=np.float64) if ls_space == "tsne2d" else emb_hd
                    mrow: Dict[str, Any] = {
                        "dataset": rec["dataset"],
                        "alpha_index": int(alpha_idx),
                        "combined_alpha": float(alpha),
                        "n_molecules": int(len(smiles)),
                        "ls_neighbor_space": ls_space,
                    }
                    mrow.update(label_smoothness_at_k(ls_emb, yv, ks))
                    if not _preset_flag_disabled(tm.get("np_enabled", False)):
                        mrow.update(
                            neighborhood_preservation_at_k(
                                list(smiles),
                                emb_hd,
                                ks,
                                morgan_radius=int(tm.get("np_morgan_radius", 2)),
                                morgan_n_bits=int(tm.get("np_morgan_nbits", 1024)),
                            )
                        )
                    if cliff_flags is not None:
                        mrow["n_cliff_mols"] = int(np.sum(cliff_flags == 1))
                    regression_metric_rows.append(mrow)

                if cliff_flags is not None and np.any(cliff_flags == 1):
                    out_ac = out_plot.with_name(out_plot.name + "_acmols")
                    plot_tsne_regression_ac_highlight(
                        points,
                        targets,
                        cliff_flags,
                        out_ac,
                        title=title + " (AC mols)",
                        colorbar_label=cb_str or None,
                    )
                    plot_paths.extend([out_ac.with_suffix(".png"), out_ac.with_suffix(".svg")])
            else:
                db = plot_tsne_classification(points, targets, out_plot, title=title)

            plot_paths.extend([out_plot.with_suffix(".png"), out_plot.with_suffix(".svg")])
            all_rows.extend(
                build_tsne_source_rows(
                    smiles=smiles,
                    points=points,
                    stage=rec["stage"],
                    dataset=rec["dataset"],
                    feature_type="combined",
                    reducer=reducer_name,
                    reducer_params=reducer_params,
                    descriptor=rec.get("descriptor"),
                    task_id=rec.get("task_id"),
                    text_target_mode=rec.get("text_target_mode"),
                    combined_alpha=float(alpha),
                    labels=targets,
                    source_refs=rec["source_refs"],
                    cliff_mol=cliff_flags,
                )
            )
            if db is not None:
                all_rows.append(
                    {
                        "point_index": -1,
                        "stage": rec["stage"],
                        "dataset": rec["dataset"],
                        "smiles": "__summary__",
                        "descriptor": rec.get("descriptor"),
                        "task_id": rec.get("task_id"),
                        "text_target_mode": rec.get("text_target_mode"),
                        "feature_type": "summary",
                        "reducer": reducer_name,
                        "reducer_params_json": json.dumps(reducer_params, sort_keys=True),
                        "x": np.nan,
                        "y": np.nan,
                        "label_or_target": db,
                        "combined_alpha": float(alpha),
                        "source_refs_json": json.dumps(rec["source_refs"], sort_keys=True),
                    }
                )

        if (
            str(dcfg.get("task_type", "classification")).lower() == "regression"
            and montage_panels
            and not _preset_flag_disabled(dcfg.get("tsne_alpha_montage", True))
        ):
            mcols = int(dcfg.get("tsne_montage_max_cols", 5) or 5)
            if mcols < 1:
                mcols = 5
            task_od = rec.get("descriptor") or f"task{rec.get('task_id', 0)}"
            cb_m = dcfg.get("tsne_colorbar_label")
            cb_s = str(cb_m).strip() if isinstance(cb_m, str) else None
            tgt_arr = np.asarray(targets, dtype=np.float64)
            base_g = out_dir / "plots" / "tsne" / f"{rec['dataset']}_{task_od}_montage_global"
            plot_paths.extend(
                plot_tsne_regression_alpha_montage(
                    montage_panels,
                    tgt_arr,
                    base_g,
                    colorbar_label=cb_s or None,
                    max_cols=mcols,
                    suptitle=f"{rec['dataset']} — t-SNE (all α, global)",
                )
            )
            if cliff_flags is not None and np.any(cliff_flags == 1):
                base_a = out_dir / "plots" / "tsne" / f"{rec['dataset']}_{task_od}_montage_acmols"
                plot_paths.extend(
                    plot_tsne_regression_alpha_montage(
                        montage_panels,
                        tgt_arr,
                        base_a,
                        colorbar_label=cb_s or None,
                        max_cols=mcols,
                        cliff_mol=cliff_flags,
                        suptitle=f"{rec['dataset']} — t-SNE (all α, AC mols)",
                    )
                )

    extra_manifest: Dict[str, Any] = {}
    if regression_metric_rows:
        metrics_path = out_dir / "downstream_tsne_metrics.csv"
        pd.DataFrame(regression_metric_rows).to_csv(metrics_path, index=False)
        extra_manifest["downstream_tsne_metrics_csv"] = str(metrics_path.resolve())

    write_source_data(all_rows, source_path, file_format=source_format)
    write_manifest(
        out_dir / "manifest.json",
        preset_path=preset_path,
        preset=preset,
        source_csv=source_path,
        plot_paths=list((out_dir / "plots").rglob("*.*")) if (out_dir / "plots").exists() else [],
        extra_artifacts=extra_manifest,
    )
    return 0
