"""Single-molecule knowledge-attention inference + lightweight bar plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from analysis.interpret.knowledge_bar_plot import clean_descriptor, plot_sample_knowledge_attention_bar
from analysis.interpret.presets_loader import checkpoint_paths_for_dataset, resolve_data_path
from analysis.interpret.runners.render_gallery import write_manifest
from dataloader.image_dataloader import ImageDataset, get_datasets, load_filenames_and_labels_multitask
from interpretability.knowledge_attention import (
    build_canonical_smiles_index,
    canonical_smiles,
    resolve_case_study_samples,
    resolve_paths,
)
from utils.clip_train_utils import calculate_attention_on_multitask
from utils.interpretability_support.visual_utils import load_finetuned_model
from utils.mol_utils import calculate_descriptors
from utils.train_utils import load_smiles
from utils.transform_utils import get_default_transforms


def _top_k_bar_from_attention(
    attention_values: np.ndarray,
    prior_keys: Sequence[str],
    filtered_keys: Sequence[str],
    top_k: int,
) -> Tuple[List[str], List[float]]:
    filtered_set = set(filtered_keys)
    present_weights = [
        float(attention_values[d_idx])
        for d_idx, descriptor in enumerate(prior_keys)
        if descriptor not in filtered_set
    ]
    present_average = float(np.mean(present_weights)) if present_weights else None
    if present_average is None or present_average == 0:
        raise ValueError("No present descriptors with non-zero average; cannot plot.")

    scored: List[Tuple[str, float]] = []
    for d_idx, descriptor in enumerate(prior_keys):
        if descriptor in filtered_set:
            continue
        weight = float(attention_values[d_idx])
        relative_pct = (weight - present_average) / present_average * 100.0
        scored.append((clean_descriptor(descriptor), relative_pct))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:top_k]
    return [label for label, _ in top], [value for _, value in top]


def _resolve_sample(
    *,
    dataset: str,
    dataroot: Path,
    case_name: Optional[str] = None,
    smiles: Optional[str] = None,
    label: Optional[int] = None,
) -> Dict[str, object]:
    if case_name:
        samples = resolve_case_study_samples(dataset, dataroot)
        matched = [s for s in samples if str(s["name"]) == case_name]
        if not matched:
            names = [str(s["name"]) for s in samples]
            raise KeyError(f"Case {case_name!r} not found for dataset {dataset!r}; known: {names}")
        return matched[0]

    if smiles is None:
        raise ValueError("Either case_name or smiles must be provided")

    canon = canonical_smiles(smiles)
    image_folder, txt_file = get_datasets(dataset, str(dataroot), data_type="processed")
    all_smiles = load_smiles(txt_file)
    names, labels = load_filenames_and_labels_multitask(
        image_folder, txt_file, task_type="classification"
    )
    canon_index = build_canonical_smiles_index(all_smiles)
    if canon not in canon_index:
        raise ValueError(
            f"SMILES not found in {dataset} dataset; use a preset case name or a molecule present in the dataset."
        )
    idx = canon_index[canon]
    label_val = labels[idx]
    label_int = int(label_val[0] if hasattr(label_val, "__len__") else label_val)
    return {
        "name": f"custom_{idx}",
        "smiles": smiles,
        "dataset_smiles": str(all_smiles[idx]),
        "image_path": str(names[idx]),
        "label": label_int,
    }


def _group_knowledge_cases(
    preset: Dict[str, Any],
    cases: List[Dict[str, Any]],
) -> List[Tuple[Path, Path, List[Dict[str, Any]]]]:
    """Group preset cases by (config, checkpoint) so knowledge memory loads once per group."""
    keys_in_order: List[Tuple[str, str]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for case in cases:
        ds = str(case["dataset"])
        cfg_path, default_ckpt = checkpoint_paths_for_dataset(preset, ds)
        ckpt = resolve_data_path(case.get("ckpt")) or default_ckpt
        key = (str(cfg_path), str(ckpt))
        if key not in grouped:
            keys_in_order.append(key)
            grouped[key] = []
        grouped[key].append(case)
    return [(Path(cfg_s), Path(ckpt_s), grouped[(cfg_s, ckpt_s)]) for cfg_s, ckpt_s in keys_in_order]


def _run_single_attention(
    *,
    sample: Mapping[str, object],
    task_id: int,
    batch_size: int,
    num_workers: int,
    device: str,
    model: Any,
) -> Tuple[np.ndarray, List[str], List[str]]:
    prior_keys = list(model.prior_fusion_block.lib.prior_keys)

    names = [str(sample["image_path"])]
    labels = np.array([[int(sample["label"])]], dtype=np.int64)
    smiles = [str(sample["dataset_smiles"])]

    dataset_obj = ImageDataset(
        names,
        labels,
        img_transformer=get_default_transforms(),
        normalize=None,
        smiles=smiles,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=min(batch_size, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    _attention, class_attention, _axis = calculate_attention_on_multitask(
        task_type="classification",
        model=model,
        data_loader=data_loader,
        device=device,
    )
    case_smiles = str(sample["smiles"])
    filtered_keys = [k for k, v in calculate_descriptors(case_smiles, prior_keys).items() if v == 0]
    attention_values = class_attention[0, :, task_id]
    return attention_values, prior_keys, filtered_keys


def run_knowledge_case(
    preset: Dict[str, Any],
    output_dir: Path,
    *,
    case_filter: Optional[str] = None,
    smiles: Optional[str] = None,
    ckpt_path: Optional[Path] = None,
    dataset: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Path]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(preset.get("top_k", 15))
    task_id = int(preset.get("task_id", 0))
    batch_size = int(preset.get("batch_size", 4))
    num_workers = int(preset.get("num_workers", 2))
    figsize_cm = tuple(float(v) for v in preset.get("figsize_cm", (18.0, 6.0)))
    dpi = int(preset.get("dpi", 150))
    formats = list(preset.get("formats") or ["png"])

    cases = list(preset.get("cases") or [])
    if case_filter:
        cases = [c for c in cases if str(c.get("name")) == case_filter or str(c.get("id")) == case_filter]
    if smiles and (ckpt_path is not None or not cases):
        if dataset is None:
            raise ValueError("--dataset is required when using --smiles without a preset case")
        ds = dataset
        _, _, dataroot, _ = resolve_paths(ds, None)
        sample = _resolve_sample(dataset=ds, dataroot=dataroot, smiles=smiles)
        if ckpt_path is None:
            cfg_path, ckpt_path_resolved = checkpoint_paths_for_dataset(
                {**preset, "checkpoint_root": preset.get("checkpoint_root")},
                ds,
            )
        else:
            cfg_path = ckpt_path.parent / "config.json"
            ckpt_path_resolved = ckpt_path
        model = load_finetuned_model(str(cfg_path), str(ckpt_path_resolved), device=device, verbose=False)
        try:
            attention_values, prior_keys, filtered_keys = _run_single_attention(
                sample=sample,
                task_id=task_id,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                model=model,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        bar_labels, bar_values = _top_k_bar_from_attention(
            attention_values, prior_keys, filtered_keys, top_k
        )
        out_stem = output_dir / str(sample["name"])
        plot_sample_knowledge_attention_bar(
            str(sample["smiles"]),
            bar_labels,
            bar_values,
            top_k=top_k,
            save_stem=out_stem,
            export_formats=formats,
            figsize_cm=figsize_cm,
            dpi=dpi,
            label=int(sample["label"]),
        )
        saved = [out_stem.with_suffix(f".{formats[0]}")]
        write_manifest(output_dir, [{"case": sample["name"], "dataset": ds, "image": saved[0].name}])
        return saved

    if not cases:
        raise ValueError("No cases selected")

    saved: List[Path] = []
    manifest: List[Dict[str, object]] = []
    for cfg_path, ckpt_path, group in _group_knowledge_cases(preset, cases):
        model = load_finetuned_model(str(cfg_path), str(ckpt_path), device=device, verbose=False)
        try:
            for case in group:
                ds = str(case["dataset"])
                _, _, dataroot, _ = resolve_paths(ds, None)
                sample = _resolve_sample(dataset=ds, dataroot=dataroot, case_name=str(case["name"]))
                attention_values, prior_keys, filtered_keys = _run_single_attention(
                    sample=sample,
                    task_id=task_id,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    model=model,
                )
                bar_labels, bar_values = _top_k_bar_from_attention(
                    attention_values, prior_keys, filtered_keys, top_k
                )
                out_stem = output_dir / str(case["name"])
                plot_sample_knowledge_attention_bar(
                    str(sample["smiles"]),
                    bar_labels,
                    bar_values,
                    top_k=top_k,
                    save_stem=out_stem,
                    export_formats=formats,
                    figsize_cm=figsize_cm,
                    dpi=dpi,
                    label=int(sample["label"]),
                )
                out_path = out_stem.with_suffix(f".{formats[0]}")
                saved.append(out_path)
                manifest.append({"case": case["name"], "dataset": ds, "image": out_path.name})
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_manifest(output_dir, manifest)
    return saved
