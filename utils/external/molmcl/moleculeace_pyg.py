# SPDX-License-Identifier: MIT
"""In-memory PyG dataset for MoleculeACE rows (SMILES + labels), MolMCL featurization."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import Dataset
import torch_geometric.transforms as T

RDLogger.DisableLog("rdApp.*")


def _mol_to_graph_fn(feat_type: str) -> Callable:
    import importlib

    mod = importlib.import_module("molmcl.utils.data")
    fn = getattr(mod, f"mol_to_graph_data_obj_{feat_type}", None)
    if fn is None:
        raise ValueError(f"Unknown feat_type {feat_type!r}; expected basic/rich/super_rich.")
    return fn


class MoleculeAceSmilesGraphDataset(Dataset):
    """One PyG ``Data`` per row; invalid SMILES raise ``ValueError``."""

    def __init__(self, smiles: list[str], labels: np.ndarray, feat_type: str = "super_rich"):
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if len(smiles) != labels.shape[0]:
            raise ValueError("smiles and labels length mismatch.")

        self.feat_type = feat_type
        self._mol_to_graph = _mol_to_graph_fn(feat_type)
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")

        self.smiles: list[str] = []
        self.labels: list[np.ndarray] = []
        self.mol_data = []

        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES at row {i}: {smi!r}")
            data = self._mol_to_graph(mol)
            data = self.transform(data)
            self.smiles.append(smi)
            self.labels.append(labels[i])
            self.mol_data.append(data)

        self.num_task = labels.shape[1]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        graph = self.mol_data[idx]
        graph.label = torch.tensor(self.labels[idx], dtype=torch.float32)
        graph.smi = self.smiles[idx]
        return graph
