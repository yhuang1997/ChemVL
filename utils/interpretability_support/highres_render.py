"""Render 1024px Grad-CAM overlays from 224px attribution masks."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from utils.interpretability_support.gradcam_utils import (
    custom_show_cam_on_image,
    normalize_attribution_for_display,
    parse_normalize_percentile,
    resolve_normalize_mode,
    upscale_attribution,
)


def _prepare_mask_for_render(
    attribution_mask: np.ndarray,
    *,
    normalize: Union[bool, str] = False,
    normalize_percentile: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    perc = normalize_percentile if normalize_percentile is not None else (2.0, 98.0)
    mode = resolve_normalize_mode(normalize)
    return normalize_attribution_for_display(
        attribution_mask,
        mode=mode,
        percentile_low=perc[0],
        percentile_high=perc[1],
    )


def smiles_to_rgb_image(smiles: str, size: int = 1024) -> np.ndarray:
    """RDKit 2D structure image as float32 RGB in [0, 1], shape (H, W, 3)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    pil_img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    return (np.array(pil_img) / 255).astype(np.float32)


def save_rgb_image(rgb: np.ndarray, output_path: str) -> None:
    """Save float32 RGB [0, 1] or uint8 RGB image to PNG."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if rgb.dtype == np.uint8:
        Image.fromarray(rgb).save(output_path)
    else:
        Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(output_path)


def render_molecule_highres(
    smiles: str,
    output_path: Optional[str] = None,
    target_resolution: int = 1024,
) -> np.ndarray:
    """Render a high-resolution RDKit molecule image (no Grad-CAM overlay)."""
    mol_img = smiles_to_rgb_image(smiles, size=target_resolution)
    mol_vis = np.uint8(np.clip(mol_img, 0, 1) * 255)
    if output_path is not None:
        save_rgb_image(mol_vis, output_path)
    return mol_vis


def render_heatmap_highres(
    attribution_mask: np.ndarray,
    output_path: Optional[str] = None,
    target_resolution: int = 1024,
    cmap_style: str = "jet_white",
    upscale_method: str = "lanczos",
    normalize: Union[bool, str] = False,
    normalize_percentile: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Render upscaled Grad-CAM heatmap on a white background (no molecule)."""
    white_bg = np.ones((target_resolution, target_resolution, 3), dtype=np.float32)
    display_mask = _prepare_mask_for_render(
        attribution_mask,
        normalize=normalize,
        normalize_percentile=normalize_percentile,
    )
    up_mask = upscale_attribution(
        display_mask,
        size=(target_resolution, target_resolution),
        method=upscale_method,
    )
    heatmap_vis = custom_show_cam_on_image(
        white_bg,
        up_mask,
        use_rgb=True,
        image_weight=0.0,
        cmap_style=cmap_style,
    )
    if output_path is not None:
        save_rgb_image(heatmap_vis, output_path)
    return heatmap_vis


def render_gradcam_highres(
    smiles: str,
    attribution_mask: np.ndarray,
    output_path: Optional[str] = None,
    target_resolution: int = 1024,
    cmap_style: str = "jet_white",
    image_weight: float = 0.5,
    upscale_method: str = "lanczos",
    normalize: Union[bool, str] = False,
    normalize_percentile: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Upscale a Grad-CAM mask and overlay on a high-resolution RDKit molecule image.

    :param attribution_mask: 2D array (H, W), usually 224x224 from Step1
    :returns: uint8 RGB image (target_resolution, target_resolution, 3)
    """
    mol_img = smiles_to_rgb_image(smiles, size=target_resolution)
    display_mask = _prepare_mask_for_render(
        attribution_mask,
        normalize=normalize,
        normalize_percentile=normalize_percentile,
    )
    up_mask = upscale_attribution(
        display_mask,
        size=(target_resolution, target_resolution),
        method=upscale_method,
    )
    cam_vis = custom_show_cam_on_image(
        mol_img,
        up_mask,
        use_rgb=True,
        image_weight=image_weight,
        cmap_style=cmap_style,
    )
    if output_path is not None:
        save_rgb_image(cam_vis, output_path)
    return cam_vis


def render_batch_from_attributions(
    smiles_list: list[str],
    attributions: np.ndarray,
    save_dir: str,
    target_resolution: int = 1024,
    cmap_style: str = "jet_white",
    image_weight: float = 0.5,
    normalize: Union[bool, str] = False,
    normalize_percentile: Optional[Tuple[float, float]] = None,
) -> list[str]:
    """
    Batch render from ``attributions`` with shape (num_samples, num_cams, H, W).

    Output files: ``cam_{idx:02d}_{cam_id:02d}_px{resolution}.png``
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples, num_cams, _, _ = attributions.shape
    paths: list[str] = []

    for idx in range(num_samples):
        smiles = smiles_list[idx]
        for cam_id in range(num_cams):
            mask = attributions[idx, cam_id]
            save_path = os.path.join(
                save_dir,
                f"cam_{idx:02d}_{cam_id:02d}_px{target_resolution}.png",
            )
            render_gradcam_highres(
                smiles,
                mask,
                output_path=save_path,
                target_resolution=target_resolution,
                cmap_style=cmap_style,
                image_weight=image_weight,
                normalize=normalize,
                normalize_percentile=normalize_percentile,
            )
            paths.append(save_path)
    return paths
