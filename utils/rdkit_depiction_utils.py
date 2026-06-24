"""
RDKit 2D depiction helpers for ablation presets (default / layout_var / style_var).

Used by ``tools/datasets/render_depiction_dataset.py``. Training reads pre-rendered PNGs
(e.g. ``zoom_50``: smaller subject baked into 224×224 PNGs).

Style ablation levers (RDKit MolDrawOptions summary):
- Atom palette via ``updateAtomPalette`` or ``useBWAtomLabels``
- Bond lines: ``bondLineWidth``, ``multipleBondOffset``
- Aromatic depiction: Kekulize when supported (fallback to default)
- Font sizes: ``minFontSize``, ``maxFontSize``, ``fixedFontSize`` when available
- Highlights: ``setHighlightColour`` (usually avoided in ablations)

``style_var`` uses white background, a distinct CPK-like palette, heavier bond/font weight,
and Kekulé aromatic rings when RDKit allows.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple, Union

from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from utils.depiction_constants import VALID_RENDER_PRESETS

RenderPreset = Literal["default", "layout_var", "style_var", "zoom_25", "zoom_50", "zoom_75"]

_ZOOM_LINEAR_SCALE: dict[str, float] = {
    "zoom_25": 0.25,
    "zoom_50": 0.50,
    "zoom_75": 0.75,
}

_PRESET_COMPARISON_ORDER: Tuple[str, ...] = ("default", "layout_var", "style_var")

_depictor_determinism_initialized = False


def _ensure_depiction_determinism() -> None:
    """Prefer RDKit built-in 2D depictor for reproducible coords (same SMILES → same layout)."""
    global _depictor_determinism_initialized
    if _depictor_determinism_initialized:
        return
    _depictor_determinism_initialized = True
    try:
        rdDepictor.SetPreferCoordGen(False)
    except Exception:
        pass


def _require_valid_preset(preset: str) -> RenderPreset:
    p = (preset or "default").strip()
    if p not in VALID_RENDER_PRESETS:
        raise ValueError(f"Unknown render_preset {p!r}; expected one of {VALID_RENDER_PRESETS}")
    return p  # type: ignore[return-value]


def prepare_mol_2d(mol: Chem.Mol, preset: RenderPreset) -> None:
    """Compute 2D coordinates in-place on ``mol`` according to ``preset``."""
    _ensure_depiction_determinism()
    if preset == "layout_var":
        try:
            from rdkit.Chem import rdCoordGen

            rdCoordGen.AddCoords(mol)
            return
        except Exception:
            rdDepictor.Compute2DCoords(mol, canonOrient=True)
            return
    if preset == "style_var":
        rdDepictor.Compute2DCoords(mol, canonOrient=True)
        return
    if preset in _ZOOM_LINEAR_SCALE:
        # Same 2D layout as default; only rasterization differs (scaled subject on canvas).
        rdDepictor.Compute2DCoords(mol, canonOrient=True)
        return
    # default
    rdDepictor.Compute2DCoords(mol, canonOrient=True)


def _style_draw_options() -> Draw.MolDrawOptions:
    """White background; style signal = palette + bond/aromatic/font tuning (not canvas tint)."""
    opts = Draw.MolDrawOptions()
    opts.bondLineWidth = 2
    opts.multipleBondOffset = 0.15
    opts.padding = 0.06
    # Explicit white background (same as default depiction intent).
    for setter in ("SetBackgroundColour", "setBackgroundColour"):
        fn = getattr(opts, setter, None)
        if callable(fn):
            try:
                fn((1.0, 1.0, 1.0))
                break
            except Exception:
                pass
    # Alternate CPK-like but hue-shifted / higher-chroma palette (atoms; bond tint follows in default drawer).
    palette = {
        1: (0.5, 0.5, 0.52),
        6: (0.22, 0.22, 0.28),
        7: (0.12, 0.35, 0.72),
        8: (0.78, 0.22, 0.08),
        9: (0.62, 0.78, 0.18),
        15: (0.75, 0.42, 0.02),
        16: (0.68, 0.58, 0.12),
        17: (0.08, 0.62, 0.38),
        35: (0.52, 0.2, 0.06),
        53: (0.42, 0.06, 0.52),
    }
    try:
        opts.updateAtomPalette(palette)
    except Exception:
        try:
            opts.useBWAtomLabels()
        except AttributeError:
            pass
    # Font: slightly larger fixed label size when supported (vs fully automatic default).
    for attr, val in (("fixedFontSize", 14), ("minFontSize", 11), ("maxFontSize", 20)):
        if hasattr(opts, attr):
            try:
                setattr(opts, attr, val)
            except Exception:
                pass
    return opts


def mol_to_pil(mol: Chem.Mol, canvas_px: int, preset: RenderPreset) -> Image.Image:
    """Rasterize a single molecule with fixed square canvas."""
    if mol is None:
        return Image.new("RGB", (canvas_px, canvas_px), color=(255, 255, 255))
    mol = Chem.Mol(mol)
    prepare_mol_2d(mol, preset)
    if preset == "style_var":
        # Aromatic style: Kekule single/double bonds when possible (vs default aromatic circles).
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        opts = _style_draw_options()
        pil = Draw.MolToImage(mol, size=(canvas_px, canvas_px), options=opts)
    elif preset in _ZOOM_LINEAR_SCALE:
        # Default RDKit draw at full canvas, then linear scale centered on white.
        base = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(canvas_px, canvas_px)).convert("RGB")
        side = max(1, round(int(canvas_px) * _ZOOM_LINEAR_SCALE[preset]))
        small = base.resize((side, side), Image.Resampling.LANCZOS)
        out = Image.new("RGB", (int(canvas_px), int(canvas_px)), (255, 255, 255))
        ox = (int(canvas_px) - side) // 2
        oy = (int(canvas_px) - side) // 2
        out.paste(small, (ox, oy))
        pil = out
    else:
        pil = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(canvas_px, canvas_px))
    return pil.convert("RGB")


def smiles_to_pil(
    smiles: str,
    canvas_px: int = 224,
    preset: Union[str, RenderPreset] = "default",
) -> Image.Image:
    """SMILES → RGB PIL image (white canvas on parse failure)."""
    pr = _require_valid_preset(str(preset))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", (canvas_px, canvas_px), color=(255, 255, 255))
    return mol_to_pil(mol, canvas_px, pr)


def render_smiles_grid(
    smiles_list: Sequence[str],
    canvas_px: int = 224,
    preset: Union[str, RenderPreset] = "default",
    titles: Optional[Sequence[str]] = None,
    max_per_row: int = 4,
) -> Image.Image:
    """Simple grid of molecules (one preset) for quick inspection."""
    pr = _require_valid_preset(str(preset))
    cells = [smiles_to_pil(s, canvas_px, pr) for s in smiles_list]
    if not cells:
        return Image.new("RGB", (canvas_px, canvas_px), color=(255, 255, 255))
    mpr = max(1, min(int(max_per_row), len(cells)))
    rows: List[Image.Image] = []
    for i in range(0, len(cells), mpr):
        rows.append(_hstack_images(cells[i : i + mpr]))
    _ = titles  # reserved for future captions
    return _vstack_images(rows)


def build_preset_comparison_grid(
    smiles_list: Sequence[str],
    canvas_px: int = 224,
    max_per_row: int = 4,
) -> Image.Image:
    """
    Rows: presets (default, layout_var, style_var); columns: molecules.

    Useful for choosing a single layout_var / style_var variant.
    """
    rows: List[Image.Image] = []
    for preset in _PRESET_COMPARISON_ORDER:
        row_img = render_smiles_grid(
            smiles_list,
            canvas_px=canvas_px,
            preset=preset,
            titles=list(smiles_list),
            max_per_row=max_per_row,
        )
        rows.append(row_img.convert("RGB"))
    return _vstack_images(rows)


def _hstack_images(images: Sequence[Image.Image]) -> Image.Image:
    w = sum(im.size[0] for im in images)
    h = max(im.size[1] for im in images)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.size[0]
    return out


def _vstack_images(images: Sequence[Image.Image]) -> Image.Image:
    w = max(im.size[0] for im in images)
    h = sum(im.size[1] for im in images)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.size[1]
    return out


def save_comparison_grids(
    smiles_list: Sequence[str],
    out_path: str,
    canvas_px: int = 224,
) -> None:
    """Write preset comparison grid to ``out_path`` (PNG)."""
    grid = build_preset_comparison_grid(smiles_list, canvas_px=canvas_px)
    grid.save(out_path)


__all__ = [
    "VALID_RENDER_PRESETS",  # re-exported from utils.depiction_constants
    "RenderPreset",
    "prepare_mol_2d",
    "mol_to_pil",
    "smiles_to_pil",
    "render_smiles_grid",
    "build_preset_comparison_grid",
    "save_comparison_grids",
]
