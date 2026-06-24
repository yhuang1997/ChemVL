"""Canonical 224px RDKit depiction for ChemVL pretraining and default downstream PNGs.

Matches the legacy ``Smiles2Img`` / OrdinalCLIP image-pretraining corpus:
``MolFromSmiles`` + ``MolsToGridImage`` at fixed square canvas (default 224).

Use ``utils.rdkit_depiction_utils`` presets (layout_var, zoom_*, style_var) only for
depiction ablations — not for pretraining or default finetuning PNGs.
"""
from __future__ import annotations

from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw

DEFAULT_CANVAS_PX = 224


def smiles_to_pretrain_pil(smiles: str, canvas_px: int = DEFAULT_CANVAS_PX) -> Image.Image:
    """SMILES → RGB PIL image (white canvas on parse failure)."""
    size = int(canvas_px)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", (size, size), color=(255, 255, 255))
    try:
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (size, size), color=(255, 255, 255))


__all__ = ["DEFAULT_CANVAS_PX", "smiles_to_pretrain_pil"]
