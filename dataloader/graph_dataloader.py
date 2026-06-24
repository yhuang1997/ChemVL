from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from utils.finetune_ablation_attacks import apply_graph_structure_mask, graph_mask_rng_seed
from utils.graph_utils import build_graph_from_smiles


class GraphDataset(Dataset):
    def __init__(
        self,
        filenames,
        labels,
        smiles=None,
        structure_mask_graph_spec: Optional[Dict[str, Any]] = None,
        cfg_for_graph_mask: Optional[Dict[str, Any]] = None,
        add_hs: Optional[bool] = None,
    ):
        self.filenames = filenames
        self.labels = labels
        self.smiles = smiles
        self.structure_mask_graph_spec = structure_mask_graph_spec
        self.cfg_for_graph_mask = cfg_for_graph_mask
        self.add_hs = add_hs

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.smiles is None:
            raise ValueError("GraphDataset requires SMILES strings for each sample.")
        smi = self.smiles[idx]
        graph = build_graph_from_smiles(smi, add_hs=self.add_hs)
        if self.structure_mask_graph_spec is not None and self.cfg_for_graph_mask is not None:
            rng = np.random.default_rng(graph_mask_rng_seed(self.cfg_for_graph_mask, idx))
            graph = apply_graph_structure_mask(
                graph,
                float(self.structure_mask_graph_spec.get("node_mask_fraction", 0.0)),
                float(self.structure_mask_graph_spec.get("edge_drop_fraction", 0.0)),
                rng,
            )
        label = self.labels[idx]
        return graph, label, smi


def graph_collate_fn(batch):
    graphs, labels, smiles = [], [], []
    for graph, label, smi in batch:
        graphs.append(graph)
        labels.append(label)
        smiles.append(smi)

    batch_graph = Batch.from_data_list(graphs)
    labels = torch.as_tensor(np.stack(labels))
    if any(smi is not None for smi in smiles):
        return batch_graph, labels, smiles
    return batch_graph, labels


def build_graph_dataloader(dataset, batch_size, shuffle, num_workers, worker_init_fn=None):
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )
    if worker_init_fn is not None:
        kw["worker_init_fn"] = worker_init_fn
    return DataLoader(**kw)
