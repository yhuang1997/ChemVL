"""Generate multi-conformer 3D SDF from SMILES (ETKDGv3 + MMFF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def write_sdf_3d_multi(
    smiles: str,
    out_path: Path,
    *,
    export_id: str,
    num_confs: int = 20,
) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    energies: List[Tuple[int, float]] = []
    for conf_id in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            props_mmff = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props_mmff, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        except Exception:
            continue

    if not energies:
        raise RuntimeError("no conformers after embed/MMFF")

    energies.sort(key=lambda x: x[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    try:
        for conf_id, energy in energies:
            mol.SetProp("_Name", f"{export_id}_conf{conf_id}_E{energy:.2f}")
            writer.write(mol, confId=conf_id)
    finally:
        writer.close()

    return {
        "num_conformers": len(energies),
        "best_energy": energies[0][1],
        "out_path": str(out_path),
    }
