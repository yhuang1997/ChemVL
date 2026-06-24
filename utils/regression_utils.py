import pandas as pd
import numpy as np
import os


class RegressionDataScheduler(object):
    def __init__(self, task_name, num_cls=1000, bin_mode="min-max", min_max_percentile=None, labels_csv_path=None):
        self.task = task_name
        self.num_tasks = 1
        self.num_cls = num_cls
        self.num_bins = num_cls - 1
        self.bin_mode = bin_mode
        self.min_max_percentile = min_max_percentile
        # If set, read labels from this CSV (e.g. MoleculeACE under cfg dataroot); else legacy MPP layout.
        self.labels_csv_path = labels_csv_path
        self.bins = self._create_bins()

    def _get_and_parse_data(self):
        if self.labels_csv_path is not None:
            data_file = self.labels_csv_path
        else:
            data_root = "datasets/downstream/MPP/regression"
            data_file = os.path.join(data_root, self.task, "processed", f"{self.task}_processed_ac.csv")

        df = pd.read_csv(data_file)
        labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
        labels = labels.astype(float)

        self.num_tasks = labels.shape[1]
        return labels  # [num_samples, num_tasks]

    def _create_bins(self):
        # create bins for each task
        labels = self._get_and_parse_data()
        bins = []
        if self.bin_mode == "min-max":
            if self.min_max_percentile is not None:
                min_ = np.percentile(labels, self.min_max_percentile[0], axis=0)
                max_ = np.percentile(labels, self.min_max_percentile[1], axis=0)
            else:
                min_ = labels.min(axis=0)
                max_ = labels.max(axis=0)
            for i in range(self.num_tasks):
                bins.append(np.linspace(min_[i], max_[i], self.num_bins))
        elif self.bin_mode == "quantile":
            for i in range(self.num_tasks):
                bins.append(np.quantile(labels[:, i], np.linspace(0, 1, self.num_bins)))
        else:
            raise ValueError(f"Unknown bin mode {self.bin_mode}")

        return bins

    def get_bin_index(self, task_id, value):
        return np.digitize(value, self.bins[task_id], right=False)

    def get_bin_edges(self, task_id, value=None, cls_index=None):
        assert (value is None) != (cls_index is None), "Either value or cls_index should be provided"
        if value is not None:
            upper_bin_index = self.get_bin_index(task_id, value)
        else:
            upper_bin_index = cls_index
        if upper_bin_index == 0:
            return -np.inf, self.bins[task_id][0]
        elif upper_bin_index == self.num_bins:
            return self.bins[task_id][self.num_bins - 1], np.inf
        else:
            return self.bins[task_id][upper_bin_index - 1], self.bins[task_id][upper_bin_index]

    # Given cls index, return the representative value
    def get_regression_value(self, task_id, cls_index):
        assert 0 <= cls_index < self.num_cls, f"Invalid cls index {cls_index}"
        if cls_index == 0:
            return self.bins[task_id][0]
        elif cls_index == self.num_cls - 1:
            return self.bins[task_id][self.num_bins - 1]
        else:
            return (self.bins[task_id][cls_index - 1] + self.bins[task_id][cls_index]) / 2

    def generate_texts(self, task_id=None, prompt=None):
        if task_id is None:
            task_id = np.arange(self.num_tasks)
        elif not isinstance(task_id, (list, tuple)):
            task_id = [task_id]
        if prompt is None:
            prompt = "Task {:d}: Value is in range {:.3f} to {:.3f}."
        all_texts = []

        for tid in task_id:
            texts = []
            for c in range(self.num_cls):
                bin_edges = self.get_bin_edges(tid, cls_index=c)
                texts.append(prompt.format(tid, bin_edges[0], bin_edges[1]))
            all_texts.append(texts)
        return np.array(all_texts)  # [num_tasks, num_cls]

    def generate_class_names(self, task_id=None):
        if task_id is None:
            task_id = np.arange(self.num_tasks)
        elif not isinstance(task_id, (list, tuple)):
            task_id = [task_id]
        all_class_names = []
        for tid in task_id:
            class_names = []
            for c in range(self.num_cls):
                bin_edges = self.get_bin_edges(tid, cls_index=c)
                class_names.append(f"Value is in range {bin_edges[0]:.3f} to {bin_edges[1]:.3f}")
            all_class_names.append(class_names)
        return np.array(all_class_names)