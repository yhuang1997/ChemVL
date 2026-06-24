"""Multiview (4 flip TTA) finetuned Grad-CAM for downstream classification tasks."""

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
from utils.interpretability_support.gradcam_utils import (
    GradCAMForOrdinalCLIP,
    _ssim_per_sample,
    get_target_fn,
)
from utils.interpretability_support.moleculeace_gradcam_common import RESOLUTION
from utils.interpretability_support.visual_utils import load_finetuned_model

VIEW_NAMES = ["identity", "hflip", "vflip", "hvflip", "mean"]


def _prepare_input_tensor(smiles: str, device: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    pil = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(RESOLUTION, RESOLUTION))
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Resize(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil).unsqueeze(0).to(device)


def _run_gradcam_mask(
    model: Any,
    input_tensor: torch.Tensor,
    target_layers: List[Any],
    label: int,
    *,
    task_id: int = 0,
    eigen_smooth: bool = True,
) -> np.ndarray:
    target_fn = get_target_fn("softmax")
    targets = [target_fn(label)]
    cam = GradCAMForOrdinalCLIP(model=model, target_layers=target_layers, task_id=task_id)
    with cam:
        attributions = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=False,
        )
    return np.asarray(attributions[0], dtype=np.float32)


def _predict_classification(
    model: Any,
    input_tensor: torch.Tensor,
    smiles: str,
    task_id: int,
) -> Tuple[int, float]:
    base_forward = model.forward
    if isinstance(model, AdaptedCLIP):
        model.forward = partial(base_forward, smiles=[smiles])
    try:
        with torch.no_grad():
            out = model(input_tensor)
        if isinstance(out, tuple):
            out = out[0]
        logits = out.detach().cpu().numpy()
        if logits.ndim >= 2:
            task_logits = logits[0, :, task_id] if logits.ndim == 3 else logits[0]
        else:
            task_logits = logits.reshape(-1)
        pred_label = int(np.argmax(task_logits))
        score = float(torch.softmax(torch.tensor(task_logits), dim=0)[pred_label])
        return pred_label, score
    finally:
        model.forward = base_forward


def collect_multiview_finetuned_gradcam(
    smiles: str,
    label: int,
    cfg_path: Path,
    ckpt_path: Path,
    *,
    task_id: int = 0,
    eigen_smooth: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Run 4 deterministic flip-view Grad-CAMs plus their mean for one molecule.

    Returns:
        masks: (5, 224, 224) — [identity, hflip_back, vflip_back, hvflip_back, mean]
        ssim: flip-SSIM dict from ``_ssim_per_sample``
        meta: prediction and checkpoint metadata
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = cfg_path.expanduser().resolve()
    ckpt_path = ckpt_path.expanduser().resolve()

    model = load_finetuned_model(str(cfg_path), str(ckpt_path), device=device, verbose=False)
    target_layers = [model.image_encoder.layer4[-1]]
    input_tensor = _prepare_input_tensor(smiles, device)

    base_forward = model.forward
    if isinstance(model, AdaptedCLIP):
        model.forward = partial(base_forward, smiles=[smiles])

    pred_label, pred_score = _predict_classification(model, input_tensor, smiles, task_id)

    try:
        identity = _run_gradcam_mask(
            model,
            input_tensor,
            target_layers,
            label,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
        )

        input_hflip = TF.hflip(input_tensor)
        hflip = _run_gradcam_mask(
            model,
            input_hflip,
            target_layers,
            label,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
        )
        hflip_back = TF.hflip(torch.from_numpy(hflip)).numpy()

        input_vflip = TF.vflip(input_tensor)
        vflip = _run_gradcam_mask(
            model,
            input_vflip,
            target_layers,
            label,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
        )
        vflip_back = TF.vflip(torch.from_numpy(vflip)).numpy()

        input_hvflip = TF.hflip(input_vflip)
        hvflip = _run_gradcam_mask(
            model,
            input_hvflip,
            target_layers,
            label,
            task_id=task_id,
            eigen_smooth=eigen_smooth,
        )
        hvflip_back = TF.vflip(TF.hflip(torch.from_numpy(hvflip))).numpy()

        mean_mask = (identity + hflip_back + vflip_back + hvflip_back) / 4.0
        masks = np.stack([identity, hflip_back, vflip_back, hvflip_back, mean_mask], axis=0)
        ssim_stats = _ssim_per_sample(identity, hflip_back, vflip_back, hvflip_back)
    finally:
        model.forward = base_forward

    meta = {
        "smiles": smiles,
        "label": int(label),
        "pred_label": pred_label,
        "pred_score": pred_score,
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "task_id": task_id,
        "eigen_smooth": eigen_smooth,
        "view_names": list(VIEW_NAMES),
    }
    return masks, ssim_stats, meta
