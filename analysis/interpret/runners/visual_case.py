"""Case-level Grad-CAM panels (structure | upstream descriptors | finetuned)."""

from __future__ import annotations

import gc
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms

from analysis.interpret.presets_loader import checkpoint_paths_for_dataset, resolve_data_path
from analysis.interpret.runners.render_gallery import write_html_gallery, write_manifest
from utils.interpretability_support.gradcam_utils import benchmark
from utils.interpretability_support.moleculeace_gradcam_common import load_finetuned_from_log_dir
from utils.interpretability_support.visual_utils import load_finetuned_model, load_pretrained_model
from utils.mol_utils import get_descriptor_value


def _case_stem(case: Mapping[str, Any]) -> str:
    return str(case.get("name") or case.get("id"))


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _park_finetuned_on_cpu(finetuned_model: Any | None) -> Any | None:
    """Free GPU memory for upstream Grad-CAM (high-rank descriptors can peak >16 GB)."""
    if finetuned_model is None:
        return None
    param = next(finetuned_model.parameters(), None)
    if param is None or param.device.type == "cpu":
        return finetuned_model
    finetuned_model.cpu()
    _maybe_empty_cuda_cache()
    return finetuned_model


def _restore_finetuned_device(finetuned_model: Any | None, device: str) -> Any | None:
    if finetuned_model is None:
        return None
    param = next(finetuned_model.parameters(), None)
    if param is None or param.device.type == device:
        return finetuned_model
    return finetuned_model.to(device)


def _prepare_batch(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[List[str], np.ndarray, np.ndarray, List[np.ndarray], torch.Tensor]:
    names: List[str] = []
    smiles_list: List[str] = []
    labels: List[int | float] = []
    mols = []
    for case in cases:
        names.append(_case_stem(case))
        smi = str(case["smiles"])
        smiles_list.append(smi)
        if "gt" in case:
            labels.append(float(case["gt"]))
        else:
            labels.append(int(case["label"]))
        mols.append(Chem.MolFromSmiles(smi))
    images = [Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224)) for mol in mols]
    rgb_images = [(np.array(image) / 255).astype(np.float32) for image in images]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = torch.stack([transform(image) for image in images], dim=0)
    return names, np.array(smiles_list), np.array(labels), rgb_images, input_tensor


def _run_panel_for_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    cfg_path: Path,
    ckpt_path: Path,
    preset: Dict[str, Any],
    device: str,
    pretrained_models: Optional[Dict[str, Any]] = None,
    finetuned_model: Optional[Any] = None,
    regression: bool = False,
) -> np.ndarray:
    names, smiles_arr, targets, rgb_images, input_tensor = _prepare_batch(cases)
    visualization = (np.concatenate(rgb_images, axis=0) * 255).astype(np.uint8)

    descriptors = list(preset.get("descriptors") or [])
    text_template = preset.get("text_template")
    eigen_smooth = bool(preset.get("eigen_smooth", True))
    aug_smooth = bool(preset.get("aug_smooth", True))
    typing = bool(preset.get("typing", False))
    upstream_target_fn = str(preset.get("upstream_target_fn_name", "softmax"))
    downstream_target_fn = str(
        preset.get("downstream_target_fn_name")
        or ("rmse" if regression else preset.get("target_fn_name", "softmax"))
    )
    task_id = int(preset.get("task_id", 0))
    upstream = bool(preset.get("upstream", True))
    downstream = bool(preset.get("downstream", True))
    if regression:
        downstream_targets = [float(t) for t in targets]
    else:
        downstream_targets = [int(t) for t in targets]

    if upstream and descriptors:
        pre_ckpt = resolve_data_path(preset.get("pretrained_ckpt"))
        if pre_ckpt is None or not pre_ckpt.is_file():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {pre_ckpt}")
        if downstream and finetuned_model is not None:
            finetuned_model = _park_finetuned_on_cpu(finetuned_model)
        for descriptor in descriptors:
            descriptor_targets = [
                get_descriptor_value(str(smi), [descriptor])[descriptor] for smi in smiles_arr
            ]
            cached = pretrained_models is not None and descriptor in pretrained_models
            if cached:
                pre_model = pretrained_models[descriptor]
            else:
                pre_model = load_pretrained_model(
                    str(pre_ckpt), descriptor=descriptor, text_template=text_template, device=device
                )
            target_layers = [pre_model.image_encoder.layer4[-1]]
            typing_info = {
                "descriptor": descriptor,
                "text_template": text_template,
                "descriptor_targets": descriptor_targets,
            }
            pre_cam = benchmark(
                pre_model,
                rgb_images,
                input_tensor,
                target_layers,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
                category=descriptor_targets,
                target_fn_name=upstream_target_fn,
                info=typing_info,
                task_id=None,
                typing=typing,
            )
            visualization = np.concatenate([visualization, pre_cam], axis=1)
            if not cached:
                del pre_model
                gc.collect()
                _maybe_empty_cuda_cache()

    if downstream:
        finetuned_model = _restore_finetuned_device(finetuned_model, device)
        if finetuned_model is None:
            if regression:
                finetuned_model, _, _ = load_finetuned_from_log_dir(
                    ckpt_path.parent if ckpt_path.parent.name != "" else cfg_path.parent,
                    ckpt_path,
                    device,
                )
            else:
                finetuned_model = load_finetuned_model(str(cfg_path), str(ckpt_path), device=device, verbose=False)
        base_forward = finetuned_model.forward
        finetuned_model.forward = partial(base_forward, smiles=smiles_arr.tolist())
        try:
            target_layers = [finetuned_model.image_encoder.layer4[-1]]
            typing_info = {"text_template": text_template, "descriptor_targets": downstream_targets}
            fin_cam = benchmark(
                finetuned_model,
                rgb_images,
                input_tensor,
                target_layers,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
                category=downstream_targets,
                target_fn_name=downstream_target_fn,
                info=typing_info,
                typing=typing,
                task_id=task_id,
                task_type="regression" if regression else "classification",
            )
            visualization = np.concatenate([visualization, fin_cam], axis=1)
        finally:
            finetuned_model.forward = base_forward

    return visualization


def load_shared_panel_models(
    preset: Dict[str, Any],
    *,
    cfg_path: Path,
    ckpt_path: Path,
    device: str,
    regression: bool = False,
) -> tuple[Dict[str, Any], Any | None]:
    """Load the finetuned model once; upstream models load per descriptor on demand."""
    downstream = bool(preset.get("downstream", True))

    finetuned_model = None
    if downstream:
        if regression:
            log_dir = ckpt_path.parent
            finetuned_model, _, _ = load_finetuned_from_log_dir(log_dir, ckpt_path, device)
        else:
            finetuned_model = load_finetuned_model(str(cfg_path), str(ckpt_path), device=device, verbose=False)

    return {}, finetuned_model


def _group_case_panel_batches(
    preset: Dict[str, Any],
    cases: List[Dict[str, Any]],
) -> List[Tuple[Path, Path, List[Dict[str, Any]], bool]]:
    """Return (cfg_path, ckpt_path, case_group, regression) batches."""
    if cases and cases[0].get("log_dir"):
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for case in cases:
            log_dir = resolve_data_path(case.get("log_dir"))
            ckpt = resolve_data_path(case.get("ckpt"))
            if log_dir is None or ckpt is None:
                raise ValueError(f"Case {_case_stem(case)} requires log_dir and ckpt")
            key = (str(log_dir), str(ckpt))
            grouped.setdefault(key, []).append(case)
        batches: List[Tuple[Path, Path, List[Dict[str, Any]], bool]] = []
        for (log_dir_s, ckpt_s), group in grouped.items():
            log_dir = Path(log_dir_s)
            ckpt = Path(ckpt_s)
            cfg_path = log_dir / "config.json"
            batches.append((cfg_path, ckpt, group, True))
        return batches

    if preset.get("checkpoint_root"):
        by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for case in cases:
            by_dataset.setdefault(str(case["dataset"]), []).append(case)
        regression = str(preset.get("benchmark", "")).lower() == "moleculeace"
        return [
            (*checkpoint_paths_for_dataset(preset, dataset), group, regression)
            for dataset, group in by_dataset.items()
        ]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        by_dataset.setdefault(str(case["dataset"]), []).append(case)
    return [
        (*checkpoint_paths_for_dataset(preset, dataset), group, False)
        for dataset, group in by_dataset.items()
    ]


def _save_panels_for_group(
    group: List[Dict[str, Any]],
    *,
    cfg_path: Path,
    ckpt_path: Path,
    preset: Dict[str, Any],
    device: str,
    output_dir: Path,
    case_filter: Optional[str],
    regression: bool,
) -> tuple[List[Path], List[Dict[str, object]]]:
    saved: List[Path] = []
    manifest: List[Dict[str, object]] = []
    pretrained_models, finetuned_model = load_shared_panel_models(
        preset,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        device=device,
        regression=regression,
    )
    try:
        dataset = str(group[0].get("dataset", ""))
        if case_filter and len(group) == 1:
            panel = _run_panel_for_cases(
                group,
                cfg_path=cfg_path,
                ckpt_path=ckpt_path,
                preset=preset,
                device=device,
                pretrained_models=pretrained_models,
                finetuned_model=finetuned_model,
                regression=regression,
            )
            stem = _case_stem(group[0])
            out_path = output_dir / f"{stem}_gradcam_panel.png"
            Image.fromarray(panel).save(out_path)
            saved.append(out_path)
            manifest.append({"case": stem, "dataset": dataset, "image": out_path.name})
        else:
            for case in group:
                panel = _run_panel_for_cases(
                    [case],
                    cfg_path=cfg_path,
                    ckpt_path=ckpt_path,
                    preset=preset,
                    device=device,
                    pretrained_models=pretrained_models,
                    finetuned_model=finetuned_model,
                    regression=regression,
                )
                stem = _case_stem(case)
                out_path = output_dir / f"{stem}_gradcam_panel.png"
                Image.fromarray(panel).save(out_path)
                saved.append(out_path)
                manifest.append({"case": stem, "dataset": dataset, "image": out_path.name})
    finally:
        pretrained_models.clear()
        finetuned_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved, manifest


def run_case_panel(
    preset: Dict[str, Any],
    output_dir: Path,
    *,
    case_filter: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Path]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cases = list(preset.get("cases") or [])
    if case_filter:
        cases = [c for c in cases if _case_stem(c) == case_filter or str(c.get("id")) == case_filter]
    if not cases:
        raise ValueError("No cases selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    manifest: List[Dict[str, object]] = []

    for cfg_path, ckpt_path, group, regression in _group_case_panel_batches(preset, cases):
        batch_saved, batch_manifest = _save_panels_for_group(
            group,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            preset=preset,
            device=device,
            output_dir=output_dir,
            case_filter=case_filter,
            regression=regression,
        )
        saved.extend(batch_saved)
        manifest.extend(batch_manifest)

    write_manifest(output_dir, manifest)
    write_html_gallery(output_dir, title=str(preset.get("title", preset["id"])), image_paths=saved)
    return saved
