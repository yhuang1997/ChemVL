"""Write zero-coordinate 2D SDF from SMILES (topology only, no 3D embed)."""

from __future__ import annotations

from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import SDWriter

    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


def write_sdf_2d(smiles: str, path: Path) -> bool:
    if not _RDKIT_OK:
        return False
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    w = SDWriter(str(path))
    try:
        mol.SetProp("_Name", path.stem)
        w.write(mol)
    finally:
        w.close()
    return True
