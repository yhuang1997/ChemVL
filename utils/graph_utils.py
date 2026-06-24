from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

try:
    from torch_geometric.data import Batch, Data

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    Batch = None
    Data = None
    TORCH_GEOMETRIC_AVAILABLE = False

# Same as MolCLR dataset.py / dataset_test.py: Z in 1..118 -> index 0..117; index 118 is mask token
# (see external/MolCLR/dataset/dataset.py masked aug: x[...,0] = len(ATOM_LIST)).
ATOM_LIST = list(range(1, 119))
MASK_ATOM_FEAT_INDEX = len(ATOM_LIST)  # 118


def _atomic_num_to_feat_index(atomic_num: int) -> int:
    """Match MolCLR's ATOM_LIST.index(Z) when possible; OOD Z uses mask index like contrastive masking."""
    try:
        return ATOM_LIST.index(int(atomic_num))
    except ValueError:
        return MASK_ATOM_FEAT_INDEX


# Official MolCLR GIN/GCN use Embedding(3) while CHIRALITY_LIST has 4 RDKit enums — upstream inconsistency.
# Map CHI_OTHER (index 3) to CHI_UNSPECIFIED (0): do not pretend OTHER is CW/CCW; keep indices in [0,2].
MOLCLR_NUM_CHIRALITY_TAG = 3


def _chirality_index_molclr(atom) -> int:
    raw = CHIRALITY_LIST.index(atom.GetChiralTag())
    if raw >= MOLCLR_NUM_CHIRALITY_TAG:
        return 0
    return raw


CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def build_graph_from_smiles(smiles: str, add_hs: Optional[bool] = None) -> Data:
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required for graph operations.")
    if add_hs is None:
        add_hs = os.environ.get("CHEMVL_GRAPH_ADD_HS", "").lower() in ("1", "true", "yes")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    if add_hs:
        mol = Chem.AddHs(mol)

    num_atoms = mol.GetNumAtoms()
    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(_atomic_num_to_feat_index(atom.GetAtomicNum()))
        chirality_idx.append(_chirality_index_molclr(atom))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row = []
    col = []
    edge_feat = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type = BOND_LIST.index(bond.GetBondType())
        bond_dir = BONDDIR_LIST.index(bond.GetBondDir())
        edge_feat.append((bond_type, bond_dir))
        edge_feat.append((bond_type, bond_dir))

    if len(row) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_atoms
    data.smiles = smiles
    return data


def build_graph_batch(smiles_list, add_hs: Optional[bool] = None):
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required for graph operations.")
    graphs = [build_graph_from_smiles(smi, add_hs=add_hs) for smi in smiles_list]
    return Batch.from_data_list(graphs)
