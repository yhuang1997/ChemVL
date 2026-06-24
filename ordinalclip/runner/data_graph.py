import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader


from ordinalclip.utils.logging import get_logger, print_log
from utils.graph_utils import (
    build_graph_from_smiles,
    ATOM_LIST,
    CHIRALITY_LIST,
    BOND_LIST,
    BONDDIR_LIST,
)

logger = get_logger(__name__)
print = lambda x: print_log(x, logger=logger)

def _normalize_identifier(name: str) -> str:
    stem = Path(name).stem
    return stem


def _load_smiles_table(metadata_file: str) -> Dict[str, str]:
    smiles_lookup: Dict[str, str] = {}
    metadata_file = os.path.expanduser(metadata_file)
    with open(metadata_file, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "index" not in reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError(f"Metadata file {metadata_file} must contain 'index' and 'smiles' columns.")
        for row in reader:
            identifier = str(row["index"]).strip()
            smiles = row["smiles"].strip()
            smiles_lookup[identifier] = smiles
    logger.info(f"Loaded {len(smiles_lookup)} entries from {metadata_file}.")
    return smiles_lookup


class GraphDatasetBase(Dataset):
    def __init__(self, data_file: str, metadata_file: str):
        self.data_file = data_file
        self.metadata = _load_smiles_table(metadata_file)
        self.sample_ids: List[str] = []
        self.labels: List[List[int]] = []

        with open(self.data_file) as fin:
            for line in fin:
                splits = line.split()
                if len(splits) <= 1:
                    continue
                sample_id = _normalize_identifier(splits[0])
                label_values = [int(label) for label in splits[1:]]
                self.sample_ids.append(sample_id)
                self.labels.append(label_values)

        self.name = Path(data_file).stem.lower()
        if "val" in self.name or "test" in self.name:
            print(f"GraphDataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"GraphDataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"GraphDataset prepare: len of labels: {len(self.labels[0])}")
        print(f"GraphDataset prepare: len of dataset: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def _lookup_smiles(self, sample_id: str) -> str:
        key = str(sample_id)
        smiles = self.metadata.get(key, None)
        if smiles is None and key.isdigit():
            smiles = self.metadata.get(str(int(key)), None)
        if smiles is None:
            raise KeyError(f"Cannot find SMILES for id={sample_id} in metadata.")
        return smiles

    def _build_graph(self, sample_id: str) -> Data:
        smiles = self._lookup_smiles(sample_id)
        graph = build_graph_from_smiles(smiles)
        graph.sample_id = sample_id
        return graph

    def _select_target(self, target_list: List[int]):
        if "val" in self.name or "test" in self.name:
            return target_list[len(target_list) // 2]
        return random.choice(target_list)

    def split_dataset_by_label(self):
        output = defaultdict(list)
        for sample_id, label in zip(self.sample_ids, self.labels):
            target = label[len(label) // 2]
            output[target].append(sample_id)
        return output

    def generate_fewshot_dataset(self, num_shots=-1, repeat=False):
        if num_shots <= 0:
            print("GraphDataset not generate few-shot dataset: num_shots<=0")
            return

        output = self.split_dataset_by_label()

        print("GraphDataset generate few-shot dataset")
        print("GraphDataset clear full dataset: sample_ids & labels")
        self._sample_ids = self.sample_ids
        self._labels = self.labels

        self.sample_ids = []
        self.labels = []
        print(
            f"GraphDataset build few_shot: num labels: {len(output.keys())}, "
            f"{list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )
        for label, sample_ids in output.items():
            if len(sample_ids) >= num_shots:
                sampled_ids = random.sample(sample_ids, num_shots)
            else:
                print(f"GraphDataset not enough: class-{label}: {len(sample_ids)}")
                if repeat:
                    sampled_ids = random.choices(sample_ids, k=num_shots)
                else:
                    sampled_ids = sample_ids

            self.sample_ids.extend(sampled_ids)
            self.labels.extend([[label]] * len(sampled_ids))
        assert len(self.sample_ids) == len(self.labels), f"{len(self.sample_ids)} != {len(self.labels)}"
        print(f"GraphDataset len of few shot dataset: {len(self.sample_ids)}")

    def generate_distribution_shifted_dataset(self, num_topk_scaled_class=-1, scale_factor=0.3):
        if num_topk_scaled_class <= 0:
            print("GraphDataset not generate distribution shifted dataset: num_topk_scaled_class<=1")
            return
        if scale_factor == 1.0:
            print("GraphDataset not generate distribution shifted dataset: scale_factor=1.0")
            return
        assert 0 < scale_factor < 1.0

        output = self.split_dataset_by_label()

        print("GraphDataset generate distribution shifted dataset")
        print("GraphDataset clear full dataset: sample_ids & labels")
        self._sample_ids = self.sample_ids
        self._labels = self.labels

        self.sample_ids = []
        self.labels = []

        print(
            f"GraphDataset build distribution shifted: num labels: {len(output.keys())}, "
            f"{list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )

        num_samples_per_label = [[k, len(v)] for k, v in output.items()]
        num_samples_per_label.sort(key=lambda x: x[1], reverse=True)

        for idx, label_cnt in enumerate(num_samples_per_label):
            if idx < num_topk_scaled_class:
                sample_ids = output[label_cnt[0]]
                sampled_ids = random.sample(sample_ids, max(int(len(sample_ids) * scale_factor), 1))
            else:
                sampled_ids = output[label_cnt[0]]

            self.sample_ids.extend(sampled_ids)
            self.labels.extend([[label_cnt[0]]] * len(sampled_ids))

        assert len(self.sample_ids) == len(self.labels), f"{len(self.sample_ids)} != {len(self.labels)}"
        print(f"GraphDataset len of distribution shifted dataset: {len(self.sample_ids)}")

    def generate_long_tail(self):
        sample_ids_new, labels_new = [], []
        len_before = len(self.labels)
        for index in range(len_before):
            sample_id, target_list = self.sample_ids[index], self.labels[index]
            if "val" in self.name or "test" in self.name:
                target = target_list[len(target_list) // 2]
            else:
                target = random.choice(target_list)
            if target >= 50:
                sample_ids_new.append(sample_id)
                labels_new.append(target_list)

        self.sample_ids = sample_ids_new
        self.labels = labels_new
        len_after = len(self.labels)
        logger.info(f"GraphDataset generate long tail dataset, the change of # of samples: {len_before} -> {len_after}.")

    @staticmethod
    def normal_sampling(mean, label_k, std=2):
        return math.exp(-((label_k - mean) ** 2) / (2 * std**2)) / (math.sqrt(2 * math.pi) * std)


class GraphRegressionDataset(GraphDatasetBase):
    def __getitem__(self, index):
        sample_id, target_list = self.sample_ids[index], self.labels[index]
        target = self._select_target(target_list)
        graph = self._build_graph(sample_id)
        return graph, target


class GraphMultiRegressionDataset(GraphDatasetBase):
    def __getitem__(self, index):
        sample_id, target_list = self.sample_ids[index], self.labels[index]
        graph = self._build_graph(sample_id)
        return graph, target_list


class GraphRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_file,
        val_data_file,
        test_data_file,
        graph_metadata_file,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        few_shot=None,
        label_distributed_shift=None,
        use_long_tail=False,
    ):
        super().__init__()
        self.train_set = GraphRegressionDataset(train_data_file, graph_metadata_file)
        self.val_set = GraphRegressionDataset(val_data_file, graph_metadata_file)
        self.test_set = GraphRegressionDataset(test_data_file, graph_metadata_file)

        few_shot = few_shot or {}
        label_distributed_shift = label_distributed_shift or {}

        self.train_set.generate_fewshot_dataset(**few_shot)
        self.train_set.generate_distribution_shifted_dataset(**label_distributed_shift)
        if use_long_tail:
            self.val_set.generate_long_tail()
            self.test_set.generate_long_tail()

        self.train_dataloder_cfg = train_dataloder_cfg or {}
        self.eval_dataloder_cfg = eval_dataloder_cfg or {}

    def train_dataloader(self):
        return GraphDataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return GraphDataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return GraphDataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)


class GraphMultiRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_file,
        val_data_file,
        test_data_file,
        graph_metadata_file,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        few_shot=None,
        label_distributed_shift=None,
        use_long_tail=False,
    ):
        super().__init__()
        self.train_set = GraphMultiRegressionDataset(train_data_file, graph_metadata_file)
        self.val_set = GraphMultiRegressionDataset(val_data_file, graph_metadata_file)
        self.test_set = GraphMultiRegressionDataset(test_data_file, graph_metadata_file)

        few_shot = few_shot or {}
        label_distributed_shift = label_distributed_shift or {}

        self.train_set.generate_fewshot_dataset(**few_shot)
        self.train_set.generate_distribution_shifted_dataset(**label_distributed_shift)
        if use_long_tail:
            self.val_set.generate_long_tail()
            self.test_set.generate_long_tail()

        self.train_dataloder_cfg = train_dataloder_cfg or {}
        self.eval_dataloder_cfg = eval_dataloder_cfg or {}

    def train_dataloader(self):
        return GraphDataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return GraphDataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return GraphDataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)
