"""Single-molecule MoleculeACE Grad-CAM inference for high-resolution rendering."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms

from models.clip_model_utils import AdaptedCLIP
from utils.interpretability_support.gradcam_utils import GradCAMForOrdinalCLIP, get_target_fn
from utils.interpretability_support.moleculeace_gradcam_common import (
    DEFAULT_DESCRIPTORS,
    RESOLUTION,
    load_finetuned_from_log_dir,
)
from utils.interpretability_support.visual_utils import load_pretrained_model
from utils.mol_utils import get_descriptor_value
from utils.path_utils import get_data_root


def _prepare_smiles_tensors(smiles: str) -> Tuple[List[np.ndarray], torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    pil = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(RESOLUTION, RESOLUTION))
    rgb = (np.array(pil).astype(np.float32) / 255.0,)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Resize(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = torch.stack([transform(pil)], dim=0)
    return list(rgb), input_tensor


def _gradcam_once(
    model: Any,
    input_tensor: torch.Tensor,
    target_layers: List[Any],
    targets: List[Any],
    *,
    task_id: int,
    eigen_smooth: bool,
) -> np.ndarray:
    cam = GradCAMForOrdinalCLIP(model=model, target_layers=target_layers, task_id=task_id)
    with cam:
        attributions = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=False,
        )
    return np.asarray(attributions[0], dtype=np.float32)


def _run_gradcam_mask(
    model: Any,
    input_tensor: torch.Tensor,
    target_layers: List[Any],
    target_value: Optional[float],
    *,
    target_fn_name: str,
    task_id: int = 0,
    eigen_smooth: bool = True,
    aug_smooth: bool = True,
) -> np.ndarray:
    """Match moleculeace ``benchmark`` flip averaging when ``aug_smooth`` is True."""
    target_fn = get_target_fn(target_fn_name)
    if target_fn_name == "scalar":
        targets = [target_fn()]
    else:
        if target_value is None:
            raise ValueError(f"target_value is required for target_fn_name={target_fn_name!r}")
        targets = [target_fn(target_value)]

    if not aug_smooth:
        return _gradcam_once(
            model,
            input_tensor,
            target_layers,
            targets,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
        )

    h = TF.hflip(input_tensor)
    v = TF.vflip(input_tensor)
    hv = TF.hflip(v)
    m0 = _gradcam_once(model, input_tensor, target_layers, targets, task_id=task_id, eigen_smooth=eigen_smooth)
    m1 = np.flip(_gradcam_once(model, h, target_layers, targets, task_id=task_id, eigen_smooth=eigen_smooth), axis=1).copy()
    m2 = np.flip(_gradcam_once(model, v, target_layers, targets, task_id=task_id, eigen_smooth=eigen_smooth), axis=0).copy()
    m3 = np.flip(
        _gradcam_once(model, hv, target_layers, targets, task_id=task_id, eigen_smooth=eigen_smooth),
        axis=(0, 1),
    ).copy()
    return ((m0 + m1 + m2 + m3) / 4.0).astype(np.float32)


def _predict_regression(
    model: Any,
    input_tensor: torch.Tensor,
    smiles: str,
    task_id: int,
) -> float:
    base_forward = model.forward
    if isinstance(model, AdaptedCLIP):
        model.forward = partial(base_forward, smiles=[smiles])
    try:
        with torch.no_grad():
            out = model(input_tensor)
        if isinstance(out, tuple):
            out = out[0]
        arr = out.detach().cpu().numpy()
        if arr.ndim == 2:
            return float(arr[0, task_id])
        return float(arr.reshape(-1)[0])
    finally:
        model.forward = base_forward


def collect_moleculeace_attributions(
    smiles: str,
    *,
    log_dir: Path,
    ckpt: Path,
    dataset_id: str,
    gt: Optional[float] = None,
    upstream: bool = False,
    descriptors: Optional[List[str]] = None,
    pretrained_ckpt: Optional[Path] = None,
    task_id: int = 0,
    eigen_smooth: bool = True,
    aug_smooth: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Run Grad-CAM for one SMILES and return raw 224px masks.

    Returns:
        attributions: (num_cams, 224, 224)
        cam_labels: label per cam channel
        meta: pred, gt_used, paths, etc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = log_dir.expanduser().resolve()
    ckpt = ckpt.expanduser().resolve()

    rgb_images, input_tensor = _prepare_smiles_tensors(smiles)
    del rgb_images
    input_tensor = input_tensor.to(device)

    finetuned_model, cfg, cfg_path = load_finetuned_from_log_dir(log_dir, ckpt, device)
    target_layers = [finetuned_model.image_encoder.layer4[-1]]

    pred = _predict_regression(finetuned_model, input_tensor, smiles, task_id)
    if gt is None:
        gradcam_target_mode = "scalar"
        gt_used = None
        fin_target_fn = "scalar"
        fin_target_value = None
    else:
        gradcam_target_mode = "rmse"
        gt_used = float(gt)
        fin_target_fn = "rmse"
        fin_target_value = gt_used

    masks: List[np.ndarray] = []
    cam_labels: List[str] = []

    if upstream:
        desc_list = descriptors if descriptors is not None else DEFAULT_DESCRIPTORS
        pre_ckpt = (
            pretrained_ckpt.expanduser().resolve()
            if pretrained_ckpt
            else (get_data_root() / "checkpoints/pretraining/RN50px224.ckpt")
        )
        if not pre_ckpt.is_file():
            raise FileNotFoundError(f"pretrained checkpoint not found: {pre_ckpt}")

        for descriptor in desc_list:
            descriptor_target = float(
                get_descriptor_value(smiles, [descriptor])[descriptor]
            )
            pre_model = load_pretrained_model(str(pre_ckpt), descriptor=descriptor)
            pre_layers = [pre_model.image_encoder.layer4[-1]]
            mask = _run_gradcam_mask(
                pre_model,
                input_tensor,
                pre_layers,
                descriptor_target,
                target_fn_name="softmax",
                task_id=task_id,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )
            masks.append(mask)
            cam_labels.append(descriptor)
            del pre_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    base_forward = finetuned_model.forward
    if isinstance(finetuned_model, AdaptedCLIP):
        finetuned_model.forward = partial(base_forward, smiles=[smiles])
    try:
        fin_mask = _run_gradcam_mask(
            finetuned_model,
            input_tensor,
            target_layers,
            fin_target_value,
            target_fn_name=fin_target_fn,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth,
        )
    finally:
        finetuned_model.forward = base_forward

    masks.append(fin_mask)
    cam_labels.append("finetuned")

    attributions = np.stack(masks, axis=0)
    meta = {
        "smiles": smiles,
        "dataset_id": dataset_id,
        "log_dir": str(log_dir),
        "ckpt": str(ckpt),
        "config": cfg_path,
        "pred": pred,
        "gt_used": gt_used,
        "gt_provided": gt is not None,
        "gradcam_target_mode": gradcam_target_mode,
        "upstream": upstream,
        "cam_labels": cam_labels,
        "task_id": task_id,
        "eigen_smooth": eigen_smooth,
        "aug_smooth": aug_smooth,
    }
    return attributions, cam_labels, meta
