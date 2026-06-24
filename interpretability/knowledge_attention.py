#!/usr/bin/env python3
"""Knowledge (text) attention inference for downstream MPP tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

from dataloader.image_dataloader import ImageDataset, get_datasets, load_filenames_and_labels_multitask
from utils.clip_train_utils import calculate_attention_on_multitask
from utils.interpretability_support.case_study_molecules import CASE_STUDY_MOLECULES
from utils.interpretability_support.visual_utils import get_ckpt_epoch, load_finetuned_model
from utils.mol_utils import calculate_descriptors
from utils.path_utils import get_data_root
from utils.plot_utils import highlight_descriptors_v2, plot_attention_v2, plot_macro_metric
from utils.splitter import scaffold_split_train_val_test
from utils.train_utils import load_smiles
from utils.transform_utils import get_default_transforms

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATASETS = ("bbbp", "bace")
CASE_STUDY_VERSION = {"bbbp": "bbbp", "bace": "bace_domain"}


def _import_bulk_preset_utils():
    """Legacy bulk CLI helpers; public users should use ``interpret.py knowledge run``."""
    import sys

    preset_dir = REPO_ROOT / "analysis/publication_figures/MAT-19_knowledge_attention_statistics"
    if not (preset_dir / "preset_utils.py").is_file():
        raise RuntimeError(
            "Bulk knowledge-attention CLI is not available in this repository layout. "
            "Use: python interpret.py knowledge run --preset analysis/interpret/presets/knowledge_cases.yaml"
        )
    if str(preset_dir) not in sys.path:
        sys.path.insert(0, str(preset_dir))
    from preset_utils import resolve_preset, resolve_preset_arg

    return resolve_preset, resolve_preset_arg


def parse_args() -> argparse.Namespace:
    resolve_preset, resolve_preset_arg = _import_bulk_preset_utils()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Optional bulk preset JSON (maintainer layout only).",
    )
    pre_args, _ = pre_parser.parse_known_args()
    preset_path = resolve_preset_arg(pre_args.preset)
    preset = resolve_preset(preset_path)
    inference = preset["inference"]

    parser = argparse.ArgumentParser(
        description="Run knowledge attention inference on MPP test splits.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DEFAULT_DATASETS,
        dest="datasets",
        help="Dataset to run (repeatable). Default: from preset.",
    )
    parser.add_argument("--batch-size", type=int, default=inference["batch_size"])
    parser.add_argument("--num-workers", type=int, default=inference["num_workers"])
    parser.add_argument("--task-id", type=int, default=inference["task_id"])
    parser.add_argument(
        "--hit-ratio-k",
        type=int,
        default=None,
        help="Top-K for hit-ratio CSV column (default: preset hit_ratio_k, usually 10).",
    )
    parser.add_argument(
        "--hit-ratio-ks",
        type=int,
        nargs="+",
        default=None,
        help="Optional advanced: multiple K columns for hit-ratio CSV (overrides single --hit-ratio-k).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write optional diagnostic plots under output_root/diagnostics/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override default {CHEMVL_DATA_ROOT}/results/knowledge_attention.",
    )
    args = parser.parse_args()
    preset_path = resolve_preset_arg(args.preset)
    overrides: Dict[str, object] = {}
    if args.hit_ratio_k is not None:
        overrides["hit_ratio_k"] = args.hit_ratio_k
    if args.hit_ratio_ks is not None:
        overrides["hit_ratio_ks"] = args.hit_ratio_ks
    args.preset_config = resolve_preset(preset_path, overrides=overrides)
    args.hit_ratio_ks = list(args.preset_config["hit_ratio_ks"])
    if args.datasets:
        args.preset_config["datasets"] = list(args.datasets)
    return args


def normalize_hit_ratio_ks(values: Sequence[int]) -> List[int]:
    if not values:
        raise ValueError("hit_ratio_ks must contain at least one positive integer")
    unique = sorted(set(values))
    if any(k <= 0 for k in unique):
        raise ValueError("hit_ratio_ks values must be positive integers")
    return unique


def resolve_paths(dataset: str, output_root: Path | None) -> tuple[Path, Path, Path, Path]:
    data_root = get_data_root()
    finetuning_ckpt_root = data_root / "checkpoints/finetuning/presets"
    cfg_path = finetuning_ckpt_root / "knowledge_prompt_tuning" / dataset / "config.json"
    ckpt_path = finetuning_ckpt_root / "knowledge_prompt_tuning" / dataset / "ckpt.pth"
    dataroot = data_root / "finetuning_datasets/MPP/classification"
    if output_root is None:
        output_root = data_root / "results/knowledge_attention"
    out_dir = output_root / dataset
    for path in (cfg_path, ckpt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    return cfg_path, ckpt_path, dataroot, out_dir


def build_test_dataloader(
    dataset: str,
    dataroot: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    task_type = "classification"
    image_folder, txt_file = get_datasets(dataset, str(dataroot), data_type="processed")
    smiles = np.array(load_smiles(txt_file))
    names, labels = load_filenames_and_labels_multitask(image_folder, txt_file, task_type=task_type)
    names, labels = np.array(names), np.array(labels)

    _, _, test_idx = scaffold_split_train_val_test(
        list(range(len(smiles))),
        smiles,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        include_chirality=True,
    )
    name_test = names[test_idx]
    labels_test = labels[test_idx]
    test_smiles = smiles[test_idx]

    test_dataset = ImageDataset(
        name_test,
        labels_test,
        img_transformer=get_default_transforms(),
        normalize=None,
        smiles=test_smiles,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_dataloader, test_smiles, labels_test, name_test


def compute_filtered_keys(test_smiles: Sequence[str], prior_keys: Sequence[str]) -> List[List[str]]:
    descriptor_targets = [calculate_descriptors(smi, prior_keys) for smi in test_smiles]
    return [[k for k, v in item.items() if v == 0] for item in descriptor_targets]


def compute_hit_ratio_multi_k(
    class_attention: np.ndarray,
    prior_keys: Sequence[str],
    labels_test: np.ndarray,
    filtered_keys_list: Sequence[Sequence[str]],
    task_id: int,
    top_ks: Sequence[int],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    max_k = max(top_ks)
    indicators = {k: {descriptor: [] for descriptor in prior_keys} for k in top_ks}
    n_samples = len(labels_test)

    for i, flks in enumerate(filtered_keys_list):
        attention_values = class_attention[i, :, task_id]
        attention = {k: v for k, v in zip(prior_keys, attention_values)}
        filtered_attention = {k: v for k, v in attention.items() if k not in flks}
        sorted_attention = sorted(filtered_attention.items(), key=lambda item: item[1], reverse=True)
        top_descriptors = [k for k, _ in sorted_attention[:max_k]]
        for k in top_ks:
            top_k_set = set(top_descriptors[:k])
            for descriptor in prior_keys:
                indicators[k][descriptor].append(1.0 if descriptor in top_k_set else 0.0)

    hit_ratios: Dict[int, Dict[str, float]] = {}
    hit_ratio_stds: Dict[int, Dict[str, float]] = {}
    for k in top_ks:
        hit_ratios[k] = {}
        hit_ratio_stds[k] = {}
        for descriptor in prior_keys:
            vals = np.asarray(indicators[k][descriptor], dtype=float)
            hit_ratios[k][descriptor] = float(np.mean(vals))
            hit_ratio_stds[k][descriptor] = (
                float(np.std(vals, ddof=1)) if n_samples > 1 else 0.0
            )
    return hit_ratios, hit_ratio_stds


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def build_canonical_smiles_index(smiles: Sequence[str]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        index[Chem.MolToSmiles(mol, isomericSmiles=True)] = idx
    return index


def resolve_case_study_samples(
    dataset: str,
    dataroot: Path,
) -> List[Dict[str, object]]:
    """Locate case-study molecules in the full dataset (any split)."""
    version = CASE_STUDY_VERSION[dataset]
    case_molecules = CASE_STUDY_MOLECULES[version]
    image_folder, txt_file = get_datasets(dataset, str(dataroot), data_type="processed")
    all_smiles = load_smiles(txt_file)
    names, labels = load_filenames_and_labels_multitask(
        image_folder, txt_file, task_type="classification"
    )
    canon_index = build_canonical_smiles_index(all_smiles)

    samples: List[Dict[str, object]] = []
    for name, info in case_molecules.items():
        case_smiles = str(info["smiles"])
        canon = canonical_smiles(case_smiles)
        if canon not in canon_index:
            raise KeyError(
                f"Case-study molecule {name!r} not found in {dataset} dataset: {case_smiles}"
            )
        idx = canon_index[canon]
        samples.append(
            {
                "name": name,
                "smiles": case_smiles,
                "dataset_smiles": str(all_smiles[idx]),
                "image_path": str(names[idx]),
                "label": int(info["label"]),
            }
        )
    return samples


def compute_case_study_attention(
    model,
    device: str,
    dataset: str,
    dataroot: Path,
    prior_keys: Sequence[str],
    task_id: int,
    batch_size: int,
    num_workers: int,
) -> List[Dict[str, object]]:
    """Run knowledge-attention inference on curated case-study molecules."""
    samples = resolve_case_study_samples(dataset, dataroot)
    names = [s["image_path"] for s in samples]
    labels = np.array([[s["label"]] for s in samples], dtype=np.int64)
    smiles = [s["dataset_smiles"] for s in samples]

    dataset_obj = ImageDataset(
        names,
        labels,
        img_transformer=get_default_transforms(),
        normalize=None,
        smiles=smiles,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=min(batch_size, len(samples)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    _attention, class_attention, _axis = calculate_attention_on_multitask(
        task_type="classification",
        model=model,
        data_loader=data_loader,
        device=device,
    )

    rows: List[Dict[str, object]] = []
    for i, sample in enumerate(samples):
        case_smiles = str(sample["smiles"])
        filtered_keys = [
            k for k, v in calculate_descriptors(case_smiles, prior_keys).items() if v == 0
        ]
        attention_values = class_attention[i, :, task_id]
        for descriptor, weight in zip(prior_keys, attention_values):
            rows.append(
                {
                    "molecule_name": str(sample["name"]),
                    "smiles": case_smiles,
                    "label": int(sample["label"]),
                    "descriptor": descriptor,
                    "attention_weight": float(weight),
                    "is_filtered": descriptor in filtered_keys,
                }
            )
    return rows


def _filtered_attention_vector(
    attention_values: np.ndarray,
    prior_keys: Sequence[str],
    filtered_keys: Sequence[str],
) -> np.ndarray:
    mask = np.array([k not in filtered_keys for k in prior_keys], dtype=bool)
    weights = attention_values[mask]
    total = float(weights.sum())
    if total <= 0:
        return weights
    return weights / total


def _normalized_descriptor_weight(
    attention_values: np.ndarray,
    prior_keys: Sequence[str],
    filtered_keys: Sequence[str],
    descriptor_idx: int,
) -> float:
    mask = np.array([k not in filtered_keys for k in prior_keys], dtype=bool)
    norm_weights = _filtered_attention_vector(attention_values, prior_keys, filtered_keys)
    present_positions = np.where(mask)[0]
    pos_in_norm = int(np.where(present_positions == descriptor_idx)[0][0])
    return float(norm_weights[pos_in_norm])


def _aggregate_stats(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "cv": None,
            "min": None,
            "max": None,
        }
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0 else None,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_prior_stability_rows(
    class_attention: np.ndarray,
    prior_keys: Sequence[str],
    filtered_keys_list: Sequence[Sequence[str]],
    task_id: int,
    n_test: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for d_idx, descriptor in enumerate(prior_keys):
        raw_vals: List[float] = []
        norm_vals: List[float] = []
        for i, flks in enumerate(filtered_keys_list):
            if descriptor in flks:
                continue
            attention_values = class_attention[i, :, task_id]
            raw_vals.append(float(attention_values[d_idx]))
            norm_vals.append(
                _normalized_descriptor_weight(attention_values, prior_keys, flks, d_idx)
            )

        raw_stats = _aggregate_stats(raw_vals)
        norm_stats = _aggregate_stats(norm_vals)
        rows.append(
            {
                "descriptor": descriptor,
                "n_present": len(raw_vals),
                "n_test": n_test,
                "mean_raw": raw_stats["mean"],
                "std_raw": raw_stats["std"],
                "cv_raw": raw_stats["cv"],
                "min_raw": raw_stats["min"],
                "max_raw": raw_stats["max"],
                "mean_norm": norm_stats["mean"],
                "std_norm": norm_stats["std"],
                "cv_norm": norm_stats["cv"],
            }
        )
    return rows


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_test_attention_csv(
    path: Path,
    class_attention: np.ndarray,
    prior_keys: Sequence[str],
    test_smiles: Sequence[str],
    labels_test: np.ndarray,
    filtered_keys_list: Sequence[Sequence[str]],
    task_id: int,
) -> None:
    rows: List[Dict[str, object]] = []
    for i, flks in enumerate(filtered_keys_list):
        attention_values = class_attention[i, :, task_id]
        filtered_set = set(flks)
        present_weights = [
            float(attention_values[d_idx])
            for d_idx, descriptor in enumerate(prior_keys)
            if descriptor not in filtered_set
        ]
        present_average = float(np.mean(present_weights)) if present_weights else None

        for d_idx, descriptor in enumerate(prior_keys):
            weight = float(attention_values[d_idx])
            is_filtered = descriptor in filtered_set
            relative_pct: float | None = None
            if not is_filtered and present_average is not None and present_average != 0:
                relative_pct = (weight - present_average) / present_average * 100.0
            rows.append(
                {
                    "sample_idx": i,
                    "smiles": str(test_smiles[i]),
                    "label": int(labels_test[i]),
                    "descriptor": descriptor,
                    "attention_weight": weight,
                    "is_filtered": is_filtered,
                    "present_descriptor_average": present_average,
                    "relative_importance_pct": relative_pct,
                }
            )
    write_csv_rows(
        path,
        rows,
        fieldnames=[
            "sample_idx",
            "smiles",
            "label",
            "descriptor",
            "attention_weight",
            "is_filtered",
            "present_descriptor_average",
            "relative_importance_pct",
        ],
    )


def write_hit_ratio_summary_csv(
    path: Path,
    hit_ratios: Dict[int, Dict[str, float]],
    hit_ratio_stds: Dict[int, Dict[str, float]],
    prior_keys: Sequence[str],
    top_ks: Sequence[int],
    n_test: int,
) -> None:
    fieldnames = ["descriptor", "n_test"]
    for k in top_ks:
        fieldnames.extend([f"hit_ratio_top{k}", f"hit_ratio_std_top{k}"])
    rows: List[Dict[str, object]] = []
    for descriptor in prior_keys:
        row: Dict[str, object] = {"descriptor": descriptor, "n_test": n_test}
        for k in top_ks:
            row[f"hit_ratio_top{k}"] = hit_ratios[k][descriptor]
            row[f"hit_ratio_std_top{k}"] = hit_ratio_stds[k][descriptor]
        rows.append(row)
    write_csv_rows(path, rows, fieldnames=fieldnames)


def write_run_metadata(
    path: Path,
    dataset: str,
    ckpt_path: Path,
    specific_epoch: int,
    num_samples: int,
    task_id: int,
    prior_keys: Sequence[str],
    axis: str,
    hit_ratio_ks: Sequence[int],
    hit_ratio_k: int,
) -> None:
    payload = {
        "dataset": dataset,
        "ckpt_path": str(ckpt_path),
        "specific_epoch": specific_epoch,
        "num_samples": num_samples,
        "task_id": task_id,
        "prior_keys": list(prior_keys),
        "hit_ratio_k": hit_ratio_k,
        "hit_ratio_ks": list(hit_ratio_ks),
        "split": "test",
        "axis_info": axis,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_stability_summary(
    path: Path,
    dataset: str,
    num_samples: int,
    prior_rows: Sequence[Dict[str, object]],
    hit_ratio_ks: Sequence[int],
    hit_ratio_k: int,
) -> None:
    prior_std_norm = [
        float(row["std_norm"]) for row in prior_rows if row.get("std_norm") is not None
    ]
    n_present_ratios = [
        float(row["n_present"]) / float(row["n_test"])
        for row in prior_rows
        if row.get("n_test")
    ]
    payload = {
        "dataset": dataset,
        "num_samples": num_samples,
        "hit_ratio_k": hit_ratio_k,
        "hit_ratio_ks": list(hit_ratio_ks),
        "prior_std_norm_mean": float(np.mean(prior_std_norm)) if prior_std_norm else None,
        "prior_std_norm_max": float(np.max(prior_std_norm)) if prior_std_norm else None,
        "prior_std_raw_mean": float(np.mean([
            float(row["std_raw"]) for row in prior_rows if row.get("std_raw") is not None
        ])) if prior_rows else None,
        "mean_n_present_ratio": float(np.mean(n_present_ratios)) if n_present_ratios else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_dataset(
    dataset: str,
    batch_size: int,
    num_workers: int,
    task_id: int,
    hit_ratio_ks: Sequence[int],
    hit_ratio_k: int,
    diagnostics: bool,
    output_root: Path | None,
) -> None:
    cfg_path, ckpt_path, dataroot, out_dir = resolve_paths(dataset, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_epoch = get_ckpt_epoch(str(ckpt_path))
    diagnostics_top_k = hit_ratio_k

    print(
        f"[{dataset}] knowledge attention | ckpt={ckpt_path} | epoch={ckpt_epoch} "
        f"| hit_ratio_k={hit_ratio_k} | hit_ratio_ks={list(hit_ratio_ks)} | out={out_dir}"
    )

    test_dataloader, test_smiles, labels_test, _name_test = build_test_dataloader(
        dataset, dataroot, batch_size, num_workers
    )
    model = load_finetuned_model(str(cfg_path), str(ckpt_path), device=device, verbose=False)
    _attention, class_attention, axis = calculate_attention_on_multitask(
        task_type="classification",
        model=model,
        data_loader=test_dataloader,
        device=device,
    )
    prior_keys = list(model.prior_fusion_block.lib.prior_keys)
    filtered_keys_list = compute_filtered_keys(test_smiles, prior_keys)
    n_test = len(test_smiles)

    metadata_path = out_dir / f"{dataset}_run_metadata.json"
    write_run_metadata(
        metadata_path,
        dataset=dataset,
        ckpt_path=ckpt_path,
        specific_epoch=ckpt_epoch,
        num_samples=n_test,
        task_id=task_id,
        prior_keys=prior_keys,
        axis=axis,
        hit_ratio_ks=hit_ratio_ks,
        hit_ratio_k=hit_ratio_k,
    )
    print(f"[{dataset}] wrote {metadata_path}")

    test_attention_path = out_dir / f"{dataset}_test_attention.csv"
    write_test_attention_csv(
        test_attention_path,
        class_attention,
        prior_keys,
        test_smiles,
        labels_test,
        filtered_keys_list,
        task_id,
    )
    print(f"[{dataset}] wrote {test_attention_path}")

    hit_ratios, hit_ratio_stds = compute_hit_ratio_multi_k(
        class_attention, prior_keys, labels_test, filtered_keys_list, task_id, hit_ratio_ks
    )
    hit_ratio_path = out_dir / f"{dataset}_hit_ratio_summary.csv"
    write_hit_ratio_summary_csv(
        hit_ratio_path, hit_ratios, hit_ratio_stds, prior_keys, hit_ratio_ks, len(labels_test)
    )
    print(f"[{dataset}] wrote {hit_ratio_path}")

    domain_rows = compute_case_study_attention(
        model=model,
        device=device,
        dataset=dataset,
        dataroot=dataroot,
        prior_keys=prior_keys,
        task_id=task_id,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    domain_path = out_dir / f"{dataset}_domain_attention.csv"
    write_csv_rows(
        domain_path,
        domain_rows,
        fieldnames=["molecule_name", "smiles", "label", "descriptor", "attention_weight", "is_filtered"],
    )
    print(f"[{dataset}] wrote {domain_path}")

    prior_stability_rows = compute_prior_stability_rows(
        class_attention,
        prior_keys,
        filtered_keys_list,
        task_id,
        n_test,
    )
    prior_stability_path = out_dir / f"{dataset}_prior_attention_stability.csv"
    write_csv_rows(
        prior_stability_path,
        prior_stability_rows,
        fieldnames=[
            "descriptor",
            "n_present",
            "n_test",
            "mean_raw",
            "std_raw",
            "cv_raw",
            "min_raw",
            "max_raw",
            "mean_norm",
            "std_norm",
            "cv_norm",
        ],
    )
    print(f"[{dataset}] wrote {prior_stability_path}")

    stability_summary_path = out_dir / f"{dataset}_stability_summary.json"
    write_stability_summary(
        stability_summary_path,
        dataset=dataset,
        num_samples=n_test,
        prior_rows=prior_stability_rows,
        hit_ratio_ks=hit_ratio_ks,
        hit_ratio_k=hit_ratio_k,
    )
    print(f"[{dataset}] wrote {stability_summary_path}")

    if diagnostics:
        diag_dir = out_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plot_info = {
            "class_attention_info": {0: class_attention},
            "prior_keys": prior_keys,
            "axis_info": axis,
        }
        plot_attention_v2(
            plot_info,
            prior_keys,
            axis,
            title_suffix=f"{dataset}_class_attention",
            reduction="mean_batch",
            save_dir=str(diag_dir),
            topK=30,
            task_id=task_id,
        )
        plot_macro_metric(
            {0: class_attention},
            prior_keys,
            specific_epoch=0,
            save_dir=str(diag_dir),
            topk=min(15, diagnostics_top_k),
            task_id=task_id,
            window_size=1,
        )
        sample_dir = diag_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, (smi, gt, flks) in enumerate(zip(test_smiles, labels_test, filtered_keys_list)):
            attention_values = class_attention[i, :, task_id]
            attention = {k: v for k, v in zip(prior_keys, attention_values)}
            highlight_descriptors_v2(
                str(smi),
                attention,
                filtered_keys=flks,
                gt=gt,
                topK=diagnostics_top_k,
                save_dir=str(sample_dir),
                prefix=f"v2_{i}_",
            )


def main() -> None:
    import sys

    print(
        "Note: bulk knowledge-attention CLI is a legacy maintainer entry.\n"
        "Use: python interpret.py knowledge run --preset analysis/interpret/presets/knowledge_cases.yaml\n",
        file=sys.stderr,
    )
    args = parse_args()
    preset = args.preset_config
    datasets = args.datasets or list(preset["datasets"])
    hit_ratio_k = int(preset["hit_ratio_k"])
    hit_ratio_ks = normalize_hit_ratio_ks(args.hit_ratio_ks)
    for dataset in datasets:
        run_dataset(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            task_id=args.task_id,
            hit_ratio_ks=hit_ratio_ks,
            hit_ratio_k=hit_ratio_k,
            diagnostics=args.diagnostics,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
