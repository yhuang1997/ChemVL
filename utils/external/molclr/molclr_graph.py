# SPDX-License-Identifier: MIT
"""MolCLR-style graph construction (always AddHs during finetune)."""

from __future__ import annotations

from torch_geometric.data import Batch, Data

from utils.graph_utils import build_graph_from_smiles


def build_molclr_graph_from_smiles(smiles: str) -> Data:
    """Match ``external/MolCLR/dataset/dataset_test.py`` (AddHs enabled)."""
    return build_graph_from_smiles(smiles, add_hs=True)


def attach_labels_to_graph(graph: Data, label) -> Data:
    import torch

    g = graph.clone()
    y = torch.as_tensor(label, dtype=torch.float32).view(1, -1)
    g.y = y
    return g


def molclr_graph_collate(batch):
    graphs, labels = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    import torch

    batch_graph.y = torch.stack([torch.as_tensor(l, dtype=torch.float32).view(-1) for l in labels])
    return batch_graph
