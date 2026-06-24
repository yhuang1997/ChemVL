"""224px MoleculeACE Grad-CAM batch pipeline (dataset pool + series bundle)."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataloader.image_dataloader import load_filenames_and_labels_multitask
from models.clip_model_utils import AdaptedCLIP, ExtendedCLIPVisual
from utils.argparser import load_config
from utils.finetune_utils import get_datafile, get_split_moleculeace
from utils.interpretability_support.gradcam_utils import benchmark
from utils.interpretability_support.moleculeace_gradcam_common import (
    DEFAULT_DESCRIPTORS,
    RESOLUTION,
    labels_for_split_indices,
    membership_split,
    prepare_dataset_images,
    resolve_ckpt,
    validate_moleculeace_cfg,
)
from utils.interpretability_support.moleculeace_result_bundle import (
    SeriesMiningConfig,
    build_bundle,
    evaluate_all_splits_metrics,
    load_task_summary,
)
from utils.interpretability_support.visual_utils import load_finetuned_model, load_pretrained_model
from utils.mol_utils import get_descriptor_value
from utils.path_utils import get_data_root
from utils.train_utils import load_smiles

DEMO_SCRIPT_NAME = "interpretability/moleculeace_gradcam.py"


@dataclass
class MoleculeAceGradcamConfig:
    log_dir: str
    output_dir: str = "results/moleculeace_interpretability"
    ckpt: Optional[str] = None
    split: str = "val"
    max_molecules: int = 6
    gradcam_batch_size: int = 8
    seed: int = 123
    upstream: bool = False
    pretrained_ckpt: Optional[str] = None
    descriptors: Optional[List[str]] = None
    eigen_smooth: bool = True
    aug_smooth: bool = True
    task_id: int = 0
    typing: bool = True
    series_min_members: int = 2
    series_max_abs_error: Optional[float] = None
    no_series_pred_filter: bool = False
    ac_tanimoto_min: float = 0.85
    ac_delta_activity_min: float = 1.0
    skip_series_bundle: bool = False
    bundle_only: bool = False
    cliff_delta_thresholds: List[float] = field(default_factory=lambda: [2.0, 3.0])
    preset_path: Optional[str] = None
    preset_dataset: Optional[str] = None
    reproduce_script_path: Optional[str] = None
    command_argv: Optional[List[str]] = None
    command_line: Optional[str] = None

    @classmethod
    def from_demo_args(cls, args: Any) -> MoleculeAceGradcamConfig:
        ckpt_path = Path(str(args.ckpt)).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        log_dir = ckpt_path.parent
        cfg_path = log_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"missing config.json next to checkpoint (expected {cfg_path})"
            )

        return cls(
            log_dir=str(log_dir),
            ckpt=str(ckpt_path),
            output_dir=str(args.output_dir),
            split=str(args.split),
            max_molecules=int(args.max_molecules),
            gradcam_batch_size=int(args.gradcam_batch_size),
            seed=int(args.seed),
            upstream=bool(args.upstream),
            pretrained_ckpt=args.pretrained_ckpt,
            descriptors=list(args.descriptors) if args.descriptors else None,
            skip_series_bundle=True,
            command_argv=list(sys.argv),
            command_line=" ".join(sys.argv),
        )

    @classmethod
    def from_preset_merge(
        cls,
        global_cfg: Dict[str, Any],
        run_entry: Dict[str, Any],
        *,
        log_dir: Path,
        ckpt: Path,
        output_dir: Path,
        preset_path: Optional[Path] = None,
        preset_dataset: Optional[str] = None,
    ) -> MoleculeAceGradcamConfig:
        merged = dict(global_cfg)
        for k, v in run_entry.items():
            if k in ("log_dir", "ckpt"):
                continue
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v

        series_max_abs_error = merged.get("series_max_abs_error")
        if merged.get("series_pred_filter") is False:
            no_series_pred_filter = True
        else:
            no_series_pred_filter = series_max_abs_error is None and "series_max_abs_error" not in merged

        return cls(
            log_dir=str(log_dir),
            ckpt=str(ckpt),
            output_dir=str(output_dir),
            split=str(merged.get("split", "val")),
            max_molecules=int(merged.get("max_molecules", 6)),
            gradcam_batch_size=int(merged.get("gradcam_batch_size", 8)),
            seed=int(merged.get("seed", 123)),
            upstream=bool(merged.get("upstream", False)),
            pretrained_ckpt=merged.get("pretrained_ckpt"),
            descriptors=list(merged["descriptors"]) if merged.get("descriptors") else None,
            eigen_smooth=merged.get("eigen_smooth", True) is not False,
            aug_smooth=merged.get("aug_smooth", True) is not False,
            task_id=int(merged.get("task_id", 0)),
            typing=merged.get("typing", True) is not False,
            series_min_members=int(merged.get("series_min_members", 2)),
            series_max_abs_error=float(series_max_abs_error) if series_max_abs_error is not None else None,
            no_series_pred_filter=no_series_pred_filter,
            ac_tanimoto_min=float(merged.get("ac_tanimoto_min", 0.85)),
            ac_delta_activity_min=float(merged.get("ac_delta_activity_min", 1.0)),
            skip_series_bundle=bool(merged.get("skip_series_bundle", False)),
            bundle_only=bool(merged.get("bundle_only", False)),
            cliff_delta_thresholds=[float(t) for t in merged.get("cliff_delta_thresholds", [2.0, 3.0])],
            preset_path=str(preset_path.resolve()) if preset_path else None,
            preset_dataset=preset_dataset,
        )

    def summary_line(self) -> str:
        return (
            f"log_dir={self.log_dir} ckpt={self.ckpt or '(auto)'} split={self.split} "
            f"max_molecules={self.max_molecules} output_dir={self.output_dir}"
        )


def _config_to_dict(config: MoleculeAceGradcamConfig) -> Dict[str, Any]:
    return {
        "log_dir": config.log_dir,
        "ckpt": config.ckpt,
        "split": config.split,
        "max_molecules": config.max_molecules,
        "gradcam_batch_size": config.gradcam_batch_size,
        "seed": config.seed,
        "output_dir": config.output_dir,
        "upstream": config.upstream,
        "pretrained_ckpt": config.pretrained_ckpt,
        "descriptors": config.descriptors,
        "eigen_smooth": config.eigen_smooth,
        "aug_smooth": config.aug_smooth,
        "task_id": config.task_id,
        "typing": config.typing,
        "series_min_members": config.series_min_members,
        "series_max_abs_error": config.series_max_abs_error,
        "no_series_pred_filter": config.no_series_pred_filter,
        "ac_tanimoto_min": config.ac_tanimoto_min,
        "ac_delta_activity_min": config.ac_delta_activity_min,
        "skip_series_bundle": config.skip_series_bundle,
        "bundle_only": config.bundle_only,
        "cliff_delta_thresholds": list(config.cliff_delta_thresholds),
    }


def _run_gradcam_in_chunks(
    *,
    finetuned_model: Any,
    chosen: List[int],
    smiles_arr: np.ndarray,
    category: np.ndarray,
    names: np.ndarray,
    target_layers: List[Any],
    config: MoleculeAceGradcamConfig,
    pre_models: Optional[Dict[str, Any]] = None,
    desc_list: Optional[List[str]] = None,
    text_template: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    n_molecules = len(chosen)
    batch_size = max(1, int(config.gradcam_batch_size))
    n_upstream_panels = len(desc_list) if desc_list else 0

    finetuned_base_forward: Optional[Any] = None
    if isinstance(finetuned_model, AdaptedCLIP):
        finetuned_base_forward = finetuned_model.forward

    fin_records: List[Dict[str, Any]] = []
    vis_rows: List[np.ndarray] = []

    for start in range(0, n_molecules, batch_size):
        end = min(start + batch_size, n_molecules)
        chunk_chosen = chosen[start:end]
        chunk_smiles = smiles_arr[start:end]
        chunk_category = category[start:end]

        rgb_chunk, _pils, tensor_chunk = prepare_dataset_images(names, chunk_chosen)
        chunk_vis = (np.concatenate(rgb_chunk, axis=0) * 255).astype(np.uint8)

        if config.upstream and pre_models and desc_list:
            for descriptor in desc_list:
                descriptor_targets = [
                    get_descriptor_value(str(smi), [descriptor])[descriptor] for smi in chunk_smiles
                ]
                pre_model = pre_models[descriptor]
                pre_target_layers = [pre_model.image_encoder.layer4[-1]]
                typing_info = {
                    "descriptor": descriptor,
                    "text_template": text_template,
                    "descriptor_targets": descriptor_targets,
                }
                pre_cam = benchmark(
                    pre_model,
                    rgb_chunk,
                    tensor_chunk,
                    pre_target_layers,
                    eigen_smooth=config.eigen_smooth,
                    aug_smooth=config.aug_smooth,
                    category=descriptor_targets,
                    target_fn_name="softmax",
                    info=typing_info,
                    task_id=None,
                    typing=config.typing,
                )
                chunk_vis = np.concatenate([chunk_vis, pre_cam], axis=1)

        if finetuned_base_forward is not None:
            finetuned_model.forward = partial(finetuned_base_forward, smiles=chunk_smiles)

        typing_info = {"text_template": text_template, "descriptor_targets": chunk_category.tolist()}
        fin_cam, records_chunk = benchmark(
            finetuned_model,
            rgb_chunk,
            tensor_chunk,
            target_layers,
            eigen_smooth=config.eigen_smooth,
            aug_smooth=config.aug_smooth,
            category=chunk_category.tolist(),
            target_fn_name="rmse",
            info=typing_info,
            typing=config.typing,
            task_id=config.task_id,
            task_type="regression",
            return_records=True,
        )
        chunk_vis = np.concatenate([chunk_vis, fin_cam], axis=1)
        fin_records.extend(records_chunk)
        vis_rows.append(chunk_vis)

        del tensor_chunk, rgb_chunk, fin_cam, chunk_vis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if finetuned_base_forward is not None:
        finetuned_model.forward = finetuned_base_forward

    visualization = np.concatenate(vis_rows, axis=0) if vis_rows else np.zeros((0, RESOLUTION, 3), dtype=np.uint8)
    return visualization, fin_records, n_upstream_panels


def _save_per_molecule_plots(visualization: np.ndarray, plots_dir: str, n_molecules: int) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    rel_paths: List[str] = []
    for i in range(n_molecules):
        row = visualization[i * RESOLUTION : (i + 1) * RESOLUTION, :]
        fname = f"mol_{i:04d}.png"
        out_path = os.path.join(plots_dir, fname)
        Image.fromarray(row).save(out_path)
        rel_paths.append(os.path.join("plots", fname))
    return rel_paths


def _build_results_csv(
    chosen: List[int],
    smiles_arr: np.ndarray,
    names: np.ndarray,
    split: str,
    membership_splits: List[str],
    fin_records: List[Dict[str, Any]],
    plot_files: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(fin_records):
        rows.append(
            {
                "molecule_idx": i,
                "pool_index": int(chosen[i]),
                "smiles": str(smiles_arr[i]),
                "image_path": str(names[chosen[i]]),
                "split": split,
                "membership_split": membership_splits[i],
                "gt": rec["gt"],
                "pred": rec["pred"],
                "abs_error": rec["abs_error"],
                "ssim_mean": rec["ssim_mean"],
                "ssim_std": rec["ssim_std"],
                "ssim_hflip": rec["ssim_hflip"],
                "ssim_vflip": rec["ssim_vflip"],
                "ssim_hvflip": rec["ssim_hvflip"],
                "plot_file": plot_files[i],
            }
        )
    return pd.DataFrame(rows)


def _split_metrics_from_task_summary(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for key, val in summary.items():
        if key.endswith("_r2") or key.endswith("_rmse"):
            metrics[key] = float(val) if val is not None else None
    return metrics


def _mining_config_from_config(config: MoleculeAceGradcamConfig) -> SeriesMiningConfig:
    if config.no_series_pred_filter:
        series_max_abs_error = None
    elif config.series_max_abs_error is not None:
        series_max_abs_error = float(config.series_max_abs_error)
    else:
        series_max_abs_error = None
    return SeriesMiningConfig(
        scope="full_dataset" if config.split == "all" else "gradcam_sample_only",
        min_members=config.series_min_members,
        ac_tanimoto_min=config.ac_tanimoto_min,
        ac_delta_activity_min=config.ac_delta_activity_min,
        series_max_abs_error=series_max_abs_error,
        cliff_delta_thresholds=tuple(float(t) for t in config.cliff_delta_thresholds),
        cliff_score_col="gt",
    )


def _reproduce_argv(config: MoleculeAceGradcamConfig) -> List[str]:
    script = config.reproduce_script_path or DEMO_SCRIPT_NAME
    ckpt_path = os.path.abspath(resolve_ckpt(config.log_dir, config.ckpt))
    reproduce = [
        sys.executable,
        script,
        "--ckpt",
        ckpt_path,
        "--split",
        config.split,
        "--seed",
        str(config.seed),
        "--max-molecules",
        str(config.max_molecules),
        "--output-dir",
        os.path.abspath(config.output_dir),
        "--gradcam-batch-size",
        str(int(config.gradcam_batch_size)),
    ]
    if config.upstream:
        reproduce.append("--upstream")
    if config.pretrained_ckpt:
        reproduce.extend(["--pretrained-ckpt", config.pretrained_ckpt])
    if config.descriptors:
        reproduce.append("--descriptors")
        reproduce.extend(config.descriptors)
    return reproduce


def save_run_manifest(
    run_dir: str,
    config: MoleculeAceGradcamConfig,
    *,
    ckpt_path: str,
    dataset_id: str,
    run_name: str,
    chosen: List[int],
    cfg_path: str,
    outputs: Optional[Dict[str, str]] = None,
    series_mining: Optional[Dict[str, Any]] = None,
    split_eval_error: Optional[str] = None,
    membership_splits: Optional[List[str]] = None,
    split_eval_errors: Optional[Dict[str, Optional[str]]] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> str:
    script_path = config.reproduce_script_path or DEMO_SCRIPT_NAME
    manifest: Dict[str, Any] = {
        "script": os.path.basename(script_path),
        "script_path": os.path.abspath(script_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": config.command_argv or [],
        "command_line": config.command_line or "",
        "args": _config_to_dict(config),
        "resolved": {
            "log_dir": os.path.abspath(config.log_dir),
            "ckpt": os.path.abspath(ckpt_path),
            "dataset_id": dataset_id,
            "split": config.split,
            "output_run_dir": os.path.abspath(run_dir),
            "training_config_source": os.path.abspath(cfg_path),
        },
        "sample": {
            "n_molecules": len(chosen),
            "pool_indices": [int(i) for i in chosen],
            "membership_splits": membership_splits or [],
            "seed": int(config.seed),
            "gradcam_batch_size": int(config.gradcam_batch_size),
        },
    }
    if n_train is not None:
        manifest["sample"]["n_train"] = int(n_train)
    if n_val is not None:
        manifest["sample"]["n_val"] = int(n_val)
    if n_test is not None:
        manifest["sample"]["n_test"] = int(n_test)
    if config.preset_path:
        manifest["batch"] = {
            "preset": os.path.abspath(config.preset_path),
            "preset_dataset": config.preset_dataset,
        }

    dest_cfg = os.path.join(run_dir, "training_config.json")
    shutil.copy2(cfg_path, dest_cfg)
    manifest["resolved"]["training_config_copy"] = dest_cfg

    reproduce = _reproduce_argv(config)
    manifest["reproduce_command_argv"] = reproduce
    manifest["reproduce_command_line"] = " ".join(reproduce)

    if outputs:
        manifest["outputs"] = {
            k: os.path.abspath(v) if not k.endswith("_dir") else os.path.abspath(v) for k, v in outputs.items()
        }
    if series_mining:
        manifest["series_mining"] = series_mining
    if split_eval_error:
        manifest["split_eval_error"] = split_eval_error
    if split_eval_errors is not None:
        manifest["split_eval_errors"] = split_eval_errors

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest_path


def run_bundle_only(config: MoleculeAceGradcamConfig) -> int:
    log_dir = os.path.abspath(config.log_dir)
    cfg_path = os.path.join(log_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: missing {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_config([cfg_path])
    validate_moleculeace_cfg(cfg)
    dataset_id = cfg["dataset"]["dataset"]
    run_name = f"{dataset_id}_{config.split}"
    run_dir = os.path.join(os.path.abspath(config.output_dir), run_name)
    csv_path = os.path.join(run_dir, f"{run_name}_interpretability.csv")
    plots_dir = os.path.join(run_dir, "plots")

    if not os.path.isfile(csv_path):
        print(f"ERROR: bundle-only requires interpretability CSV: {csv_path}", file=sys.stderr)
        return 2
    if not os.path.isdir(plots_dir) or not any(
        f.startswith("mol_") and f.endswith(".png") for f in os.listdir(plots_dir)
    ):
        print(f"ERROR: bundle-only requires plots under: {plots_dir}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)
    ckpt_path = resolve_ckpt(log_dir, config.ckpt)
    existing_summary = load_task_summary(run_dir)
    split_metrics = _split_metrics_from_task_summary(existing_summary)
    split_eval_errors: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    split_eval_error = None

    n_molecules = len(df)
    if config.split == "all" and "membership_split" in df.columns:
        membership_splits = df["membership_split"].astype(str).tolist()
        n_train = sum(1 for s in membership_splits if s == "train")
        n_val = sum(1 for s in membership_splits if s == "val")
        n_test = sum(1 for s in membership_splits if s == "test")
    else:
        membership_splits = df["split"].astype(str).tolist() if "split" in df.columns else [config.split] * n_molecules
        n_train = n_val = n_test = None

    chosen = df["pool_index"].astype(int).tolist() if "pool_index" in df.columns else list(range(n_molecules))

    manifest_path_old = os.path.join(run_dir, "run_manifest.json")
    n_upstream_panels = 0
    if os.path.isfile(manifest_path_old):
        try:
            with open(manifest_path_old, encoding="utf-8") as f:
                old_manifest = json.load(f)
            if old_manifest.get("args", {}).get("upstream"):
                desc_list = old_manifest.get("args", {}).get("descriptors") or []
                n_upstream_panels = len(desc_list) if desc_list else len(DEFAULT_DESCRIPTORS)
        except (json.JSONDecodeError, OSError):
            pass

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    mining = _mining_config_from_config(config)
    gradcam_available = os.path.isfile(os.path.join(run_dir, "gradcam_gallery.png")) and n_molecules > 0

    bundle_outputs: Dict[str, str] = {
        "interpretability_csv": csv_path,
        "gradcam_gallery": os.path.join(run_dir, "gradcam_gallery.png"),
        "plots_dir": plots_dir,
    }
    series_mining_manifest: Optional[Dict[str, Any]] = None

    if not config.skip_series_bundle:
        bundle_info = build_bundle(
            df,
            dataset_id=dataset_id,
            split=config.split,
            run_dir=run_dir,
            checkpoint_path=ckpt_path,
            split_metrics=split_metrics,
            gradcam_available=gradcam_available,
            mining=mining,
            n_molecules=n_molecules,
            n_upstream_panels=n_upstream_panels,
            timestamp_utc=timestamp_utc,
        )
        bundle_outputs.update(
            {
                "task_summary": bundle_info["task_summary"],
                "series_candidates": bundle_info["series_candidates"],
                "series_previews_dir": bundle_info["series_previews_dir"],
            }
        )
        for dir_name, cliff_dir in (bundle_info.get("activity_cliffs") or {}).items():
            bundle_outputs[dir_name] = cliff_dir
        series_mining_manifest = bundle_info["series_mining"]

    manifest_path = save_run_manifest(
        run_dir,
        config,
        ckpt_path=ckpt_path,
        dataset_id=dataset_id,
        run_name=run_name,
        chosen=chosen,
        cfg_path=cfg_path,
        outputs=bundle_outputs,
        series_mining=series_mining_manifest,
        split_eval_error=split_eval_error,
        membership_splits=membership_splits,
        split_eval_errors=split_eval_errors,
        n_train=n_train if config.split == "all" else None,
        n_val=n_val if config.split == "all" else None,
        n_test=n_test if config.split == "all" else None,
    )

    print(f"Bundle-only refresh: {run_dir}")
    if not config.skip_series_bundle:
        print(f"  series_candidates: {bundle_outputs.get('series_candidates')}")
        for key in sorted(k for k in bundle_outputs if k.startswith("activity_cliffs")):
            print(f"  {key}: {bundle_outputs[key]}")
    print(f"  manifest: {manifest_path}")
    return 0


def run_moleculeace_gradcam(config: MoleculeAceGradcamConfig) -> int:
    if config.bundle_only:
        return run_bundle_only(config)

    log_dir = os.path.abspath(config.log_dir)
    cfg_path = os.path.join(log_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: missing {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_config([cfg_path])
    validate_moleculeace_cfg(cfg)

    image_folder, txt_file = get_datafile(cfg)
    cfg.setdefault("regression_scheduler", {})["labels_csv_path"] = txt_file

    task_type = cfg["dataset"]["task_type"]
    names, labels = load_filenames_and_labels_multitask(image_folder, txt_file, task_type=task_type)
    names, labels = np.array(names), np.array(labels)
    smiles = load_smiles(txt_file)

    train_idx, val_idx, test_idx = get_split_moleculeace(cfg, names, labels, smiles)
    train_set = set(int(i) for i in train_idx)
    val_set = set(int(i) for i in val_idx)
    test_set = set(int(i) for i in test_idx)

    if config.split == "train":
        pool = train_idx.tolist()
    elif config.split == "val":
        pool = val_idx.tolist()
    elif config.split == "test":
        pool = test_idx.tolist()
    else:
        pool = train_idx.tolist() + val_idx.tolist() + test_idx.tolist()

    if len(pool) == 0:
        print(f"ERROR: no samples in pool for split {config.split!r}", file=sys.stderr)
        return 2

    if config.max_molecules == -1:
        chosen = list(pool)
    else:
        rng = random.Random(config.seed)
        k = min(config.max_molecules, len(pool))
        chosen = rng.sample(pool, k=k)

    if config.split == "all":
        membership_splits = [membership_split(int(i), train_set, val_set, test_set) for i in chosen]
    else:
        membership_splits = [config.split] * len(chosen)

    n_molecules = len(chosen)
    n_train = sum(1 for s in membership_splits if s == "train")
    n_val = sum(1 for s in membership_splits if s == "val")
    n_test = sum(1 for s in membership_splits if s == "test")
    smiles_arr = np.array([smiles[i] for i in chosen], dtype=object)
    category = labels_for_split_indices(labels, np.array(chosen))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = resolve_ckpt(log_dir, config.ckpt)

    fd, tmp_cfg_path = tempfile.mkstemp(suffix="_moleculeace_interpret.json")
    os.close(fd)
    try:
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f, indent=4)
        finetuned_model = load_finetuned_model(tmp_cfg_path, ckpt_path, device=device, verbose=False)
    except Exception:
        try:
            os.unlink(tmp_cfg_path)
        except OSError:
            pass
        raise

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    split_metrics: Dict[str, Optional[float]] = {}
    split_eval_errors: Dict[str, Optional[str]] = {}
    split_eval_error: Optional[str] = None

    if config.skip_series_bundle:
        pass
    else:
        split_metrics, split_eval_errors = evaluate_all_splits_metrics(
            finetuned_model,
            cfg,
            names,
            labels,
            smiles,
            train_idx,
            val_idx,
            test_idx,
            device,
        )
        failed = [sp for sp, err in split_eval_errors.items() if err]
        if failed:
            split_eval_error = "; ".join(f"{sp}: {split_eval_errors[sp]}" for sp in failed)
            print(f"Warning: split metrics failed for {failed}: {split_eval_error}", file=sys.stderr)

        del finetuned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finetuned_model = load_finetuned_model(tmp_cfg_path, ckpt_path, device=device, verbose=False)

    if not isinstance(finetuned_model, (AdaptedCLIP, ExtendedCLIPVisual)):
        print(
            f"ERROR: unsupported model type {type(finetuned_model).__name__} for Grad-CAM. "
            "Use AdaptedCLIP or ExtendedCLIPVisual.",
            file=sys.stderr,
        )
        return 2

    if not hasattr(finetuned_model, "image_encoder") or not hasattr(finetuned_model.image_encoder, "layer4"):
        print(
            f"ERROR: model has no image_encoder.layer4 (got {type(finetuned_model).__name__}).",
            file=sys.stderr,
        )
        return 2

    target_layers = [finetuned_model.image_encoder.layer4[-1]]

    text_template = None
    pre_models: Optional[Dict[str, Any]] = None
    desc_list: Optional[List[str]] = None
    if config.upstream:
        pre_ckpt = config.pretrained_ckpt or str(get_data_root() / "checkpoints/pretraining/RN50px224.ckpt")
        if not os.path.isfile(pre_ckpt):
            print(f"ERROR: pretrained checkpoint not found: {pre_ckpt}", file=sys.stderr)
            return 2
        desc_list = config.descriptors if config.descriptors is not None else DEFAULT_DESCRIPTORS
        pre_models = {}
        for descriptor in desc_list:
            print(f"Loading pretrained model for descriptor {descriptor}...")
            pre_models[descriptor] = load_pretrained_model(
                pre_ckpt, descriptor=descriptor, text_template=text_template, device=device
            )

    print(
        f"Running Grad-CAM in chunks of {max(1, config.gradcam_batch_size)} "
        f"({n_molecules} molecules total)...",
    )
    visualization, fin_records, n_upstream_panels = _run_gradcam_in_chunks(
        finetuned_model=finetuned_model,
        chosen=chosen,
        smiles_arr=smiles_arr,
        category=category,
        names=names,
        target_layers=target_layers,
        config=config,
        pre_models=pre_models,
        desc_list=desc_list,
        text_template=text_template,
    )

    dataset_id = cfg["dataset"]["dataset"]
    run_name = f"{dataset_id}_{config.split}"
    run_dir = os.path.join(os.path.abspath(config.output_dir), run_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_dir, exist_ok=True)

    plot_files = _save_per_molecule_plots(visualization, plots_dir, n_molecules)

    gallery_path = os.path.join(run_dir, "gradcam_gallery.png")
    Image.fromarray(visualization).save(gallery_path)
    mosaic_name = f"{run_name}_Grad-CAM.png"
    mosaic_path = os.path.join(run_dir, mosaic_name)
    Image.fromarray(visualization).save(mosaic_path)

    df = _build_results_csv(chosen, smiles_arr, names, config.split, membership_splits, fin_records, plot_files)
    csv_name = f"{run_name}_interpretability.csv"
    csv_path = os.path.join(run_dir, csv_name)
    df.to_csv(csv_path, index=False, float_format="%.6f")

    gradcam_available = os.path.isfile(gallery_path) and n_molecules > 0
    bundle_outputs: Dict[str, str] = {
        "interpretability_csv": csv_path,
        "gradcam_gallery": gallery_path,
        "gradcam_gallery_legacy": mosaic_path,
        "plots_dir": plots_dir,
    }
    series_mining_manifest: Optional[Dict[str, Any]] = None

    if not config.skip_series_bundle:
        mining = _mining_config_from_config(config)
        bundle_info = build_bundle(
            df,
            dataset_id=dataset_id,
            split=config.split,
            run_dir=run_dir,
            checkpoint_path=ckpt_path,
            split_metrics=split_metrics,
            gradcam_available=gradcam_available,
            mining=mining,
            n_molecules=n_molecules,
            n_upstream_panels=n_upstream_panels,
            timestamp_utc=timestamp_utc,
        )
        bundle_outputs.update(
            {
                "task_summary": bundle_info["task_summary"],
                "series_candidates": bundle_info["series_candidates"],
                "series_previews_dir": bundle_info["series_previews_dir"],
            }
        )
        for dir_name, cliff_dir in (bundle_info.get("activity_cliffs") or {}).items():
            bundle_outputs[dir_name] = cliff_dir
        series_mining_manifest = bundle_info["series_mining"]

    manifest_path = save_run_manifest(
        run_dir,
        config,
        ckpt_path=ckpt_path,
        dataset_id=dataset_id,
        run_name=run_name,
        chosen=chosen,
        cfg_path=cfg_path,
        outputs=bundle_outputs,
        series_mining=series_mining_manifest,
        split_eval_error=split_eval_error,
        membership_splits=membership_splits,
        split_eval_errors=split_eval_errors,
        n_train=n_train if config.split == "all" else None,
        n_val=n_val if config.split == "all" else None,
        n_test=n_test if config.split == "all" else None,
    )

    print(f"Run directory: {run_dir}")
    print(f"Saved gallery ({n_molecules} rows): {gallery_path}")
    print(f"Saved legacy mosaic: {mosaic_path}")
    print(f"Saved {n_molecules} plots under: {plots_dir}")
    print(f"Saved CSV: {csv_path}")
    if not config.skip_series_bundle:
        print(f"Saved task_summary: {bundle_outputs.get('task_summary')}")
        print(f"Saved series_candidates: {bundle_outputs.get('series_candidates')}")
        for key in sorted(k for k in bundle_outputs if k.startswith("activity_cliffs")):
            print(f"Saved {key}: {bundle_outputs[key]}")
    print(f"Saved run manifest: {manifest_path}")
    try:
        os.unlink(tmp_cfg_path)
    except OSError:
        pass
    return 0
