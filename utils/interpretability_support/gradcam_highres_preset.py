"""YAML preset helpers for high-resolution Grad-CAM runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from utils.interpretability_support.gradcam_highres import collect_moleculeace_attributions
from utils.interpretability_support.gradcam_utils import (
    parse_normalize_percentile,
    resolve_normalize_mode,
)
from utils.interpretability_support.highres_render import (
    render_gradcam_highres,
    render_heatmap_highres,
    render_molecule_highres,
)

MOLECULE_KEYS = frozenset({"smiles", "gt", "cam_ids"})


@dataclass
class HighresRunItem:
    dataset_name: str
    mol_key: str
    log_dir: Path
    ckpt: Path
    cfg: Dict[str, Any]


def load_highres_preset(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a mapping: {path}")
    version = data.get("version")
    if version != 2:
        raise ValueError(f"Unsupported preset version (expected 2, got {version!r}): {path}")
    return data


def _resolve_output_dir(preset: Dict[str, Any], repo_root: Path) -> Path:
    global_cfg = preset.get("global") or {}
    raw = global_cfg.get("output_dir")
    if not raw:
        raise ValueError("preset global.output_dir is required")
    out = Path(str(raw)).expanduser()
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    else:
        out = out.resolve()
    return out


def _resolve_path(raw: Any, repo_root: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _normalize_render_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract display-normalization settings from merged preset cfg."""
    mode = resolve_normalize_mode(cfg.get("normalize", False))
    p_low, p_high = parse_normalize_percentile(cfg.get("normalize_percentile"))
    return {
        "normalize": False if mode == "none" else mode,
        "normalize_percentile": (p_low, p_high),
        "normalize_mode": mode,
    }


def _merge_cfg(*layers: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if key in MOLECULE_KEYS:
                out[key] = value
            elif value is not None:
                out[key] = value
    return out


def expand_highres_runs(preset: Dict[str, Any], repo_root: Path) -> List[HighresRunItem]:
    global_cfg: Dict[str, Any] = dict(preset.get("global") or {})
    datasets = preset.get("datasets") or []
    if not datasets:
        raise ValueError("preset has no datasets")

    items: List[HighresRunItem] = []
    for ds_entry in datasets:
        if not isinstance(ds_entry, dict):
            raise ValueError("each datasets[] entry must be a mapping")
        dataset_name = str(ds_entry.get("name") or "")
        if not dataset_name:
            raise ValueError("datasets[].name is required")
        if "log_dir" not in ds_entry or "ckpt" not in ds_entry:
            raise ValueError(f"{dataset_name}: log_dir and ckpt are required")
        log_dir = _resolve_path(ds_entry["log_dir"], repo_root)
        ckpt = _resolve_path(ds_entry["ckpt"], repo_root)
        dataset_cfg = {
            k: v
            for k, v in ds_entry.items()
            if k not in {"name", "log_dir", "ckpt", "molecules"}
        }
        molecules = ds_entry.get("molecules") or {}
        if not isinstance(molecules, dict):
            raise ValueError(f"{dataset_name}: molecules must be a mapping")
        if not molecules:
            raise ValueError(f"{dataset_name}: molecules is empty")

        for mol_key, mol_entry in molecules.items():
            if not isinstance(mol_entry, dict):
                raise ValueError(f"{dataset_name}/{mol_key}: molecule entry must be a mapping")
            if not mol_entry.get("smiles"):
                raise ValueError(f"{dataset_name}/{mol_key}: smiles is required")
            merged = _merge_cfg(global_cfg, dataset_cfg, mol_entry)
            merged["dataset_id"] = dataset_name
            items.append(
                HighresRunItem(
                    dataset_name=dataset_name,
                    mol_key=str(mol_key),
                    log_dir=log_dir,
                    ckpt=ckpt,
                    cfg=merged,
                )
            )
    return items


def run_highres_preset(
    preset_path: Path,
    *,
    repo_root: Path,
    dry_run: bool = False,
    only_molecules: Optional[List[str]] = None,
    only_datasets: Optional[List[str]] = None,
    output_dir_override: Optional[str] = None,
) -> int:
    preset = load_highres_preset(preset_path)
    os.chdir(repo_root)
    global_cfg: Dict[str, Any] = dict(preset.get("global") or {})
    if output_dir_override:
        global_cfg["output_dir"] = output_dir_override
        preset = {**preset, "global": global_cfg}
    try:
        run_items = expand_highres_runs(preset, repo_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2

    output_base = _resolve_output_dir(preset, repo_root)
    target_resolution = int(global_cfg.get("target_resolution", 1024))
    cmap_style = str(global_cfg.get("cmap_style", "jet_white"))
    image_weight = float(global_cfg.get("image_weight", 0.5))
    default_norm = _normalize_render_kwargs(global_cfg)

    if only_datasets:
        only_ds = set(only_datasets)
        run_items = [item for item in run_items if item.dataset_name in only_ds]
        if not run_items:
            print(f"ERROR: no runs match --only-datasets {only_datasets}", file=__import__("sys").stderr)
            return 2

    if only_molecules:
        only_mol = set(only_molecules)
        run_items = [item for item in run_items if item.mol_key in only_mol]
        if not run_items:
            print(f"ERROR: no runs match --only-molecules {only_molecules}", file=__import__("sys").stderr)
            return 2

    print(f"Preset: {preset_path} ({preset.get('name', preset_path.stem)})")
    print(f"Output base: {output_base}")
    print(f"Resolution: {target_resolution}px, cmap: {cmap_style}")
    print(f"aug_smooth: {global_cfg.get('aug_smooth', True)}")
    print(
        f"normalize: {default_norm['normalize_mode']} "
        f"(percentile {default_norm['normalize_percentile'][0]:g}–"
        f"{default_norm['normalize_percentile'][1]:g})"
    )
    print(f"Runs: {len(run_items)}")
    if dry_run:
        print("DRY RUN\n")

    ok: List[str] = []
    failed: List[Tuple[str, str]] = []

    for item in run_items:
        cfg = item.cfg
        smiles = str(cfg.get("smiles") or "")
        dataset_id = str(cfg.get("dataset_id") or item.dataset_name)
        out_dir = output_base / item.dataset_name / item.mol_key
        cam_ids = cfg.get("cam_ids")
        run_label = f"{item.dataset_name}/{item.mol_key}"

        print(f"=== {run_label} ===")
        print(f"  dataset: {dataset_id}")
        print(f"  log_dir: {item.log_dir}")
        print(f"  ckpt:    {item.ckpt}")
        print(f"  output:  {out_dir}")

        if dry_run:
            ok.append(run_label)
            continue

        try:
            gt_raw = cfg.get("gt")
            gt = None if gt_raw is None else float(gt_raw)
            pretrained_ckpt = cfg.get("pretrained_ckpt")
            pre_path = _resolve_path(pretrained_ckpt, repo_root) if pretrained_ckpt else None
            descriptors = cfg.get("descriptors")

            attributions, cam_labels, meta = collect_moleculeace_attributions(
                smiles,
                log_dir=item.log_dir,
                ckpt=item.ckpt,
                dataset_id=dataset_id,
                gt=gt,
                upstream=bool(cfg.get("upstream", False)),
                descriptors=descriptors,
                pretrained_ckpt=pre_path,
                task_id=int(cfg.get("task_id", 0)),
                eigen_smooth=bool(cfg.get("eigen_smooth", True)),
                aug_smooth=bool(cfg.get("aug_smooth", True)),
            )

            selected_ids = list(range(attributions.shape[0])) if cam_ids is None else [int(x) for x in cam_ids]
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "attributions.npy", attributions)

            molecule_path = out_dir / f"molecule_px{target_resolution}.png"
            render_molecule_highres(
                smiles,
                output_path=str(molecule_path),
                target_resolution=target_resolution,
            )

            norm_kw = _normalize_render_kwargs(cfg)
            png_paths: List[str] = []
            heatmap_paths: List[str] = []
            for cam_id in selected_ids:
                if cam_id < 0 or cam_id >= attributions.shape[0]:
                    raise IndexError(f"cam_id {cam_id} out of range [0, {attributions.shape[0]})")
                label = cam_labels[cam_id].replace("/", "_")
                png_path = out_dir / f"cam_{cam_id:02d}_{label}_px{target_resolution}.png"
                heatmap_path = out_dir / f"heatmap_{cam_id:02d}_{label}_px{target_resolution}.png"
                render_gradcam_highres(
                    smiles,
                    attributions[cam_id],
                    output_path=str(png_path),
                    target_resolution=target_resolution,
                    cmap_style=cmap_style,
                    image_weight=image_weight,
                    normalize=norm_kw["normalize"],
                    normalize_percentile=norm_kw["normalize_percentile"],
                )
                render_heatmap_highres(
                    attributions[cam_id],
                    output_path=str(heatmap_path),
                    target_resolution=target_resolution,
                    cmap_style=cmap_style,
                    normalize=norm_kw["normalize"],
                    normalize_percentile=norm_kw["normalize_percentile"],
                )
                png_paths.append(str(png_path.resolve()))
                heatmap_paths.append(str(heatmap_path.resolve()))

            meta["dataset_name"] = item.dataset_name
            meta["mol_key"] = item.mol_key
            meta["cam_ids_rendered"] = selected_ids
            meta["png_paths"] = png_paths
            meta["heatmap_paths"] = heatmap_paths
            meta["molecule_png_path"] = str(molecule_path.resolve())
            meta["attributions_path"] = str((out_dir / "attributions.npy").resolve())
            meta["normalize"] = norm_kw["normalize_mode"]
            meta["normalize_percentile"] = list(norm_kw["normalize_percentile"])
            (out_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            gt_display = meta["gt_used"]
            gt_str = f"{gt_display:.4f}" if gt_display is not None else "null"
            print(
                f"  pred={meta['pred']:.4f} gt_used={gt_str} "
                f"mode={meta.get('gradcam_target_mode')} cams={len(png_paths)}"
            )
            ok.append(run_label)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=__import__("sys").stderr)
            failed.append((run_label, str(exc)))

    print("\n--- Summary ---")
    print(f"OK ({len(ok)}): {', '.join(ok) if ok else '(none)'}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for nm, reason in failed:
            print(f"  {nm}: {reason}")
        return 1
    return 0
