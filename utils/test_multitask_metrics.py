"""Unit tests for multitask per-task test metric helpers."""

from __future__ import annotations

import unittest

from utils.finetune_utils import extract_per_task_test_metrics, multitask_column_names


class MultitaskColumnNamesTests(unittest.TestCase):
    def test_clintox_ignores_class_names(self) -> None:
        cfg = {
            "dataset": {
                "dataset": "clintox",
                "num_tasks": 2,
                "class_names": ["class1", "class2"],
            }
        }
        self.assertEqual(
            multitask_column_names(cfg, 2),
            ["FDA_APPROVED", "CT_TOX"],
        )

    def test_tox21_ignores_class_names(self) -> None:
        cfg = {
            "dataset": {
                "dataset": "tox21",
                "num_tasks": 12,
                "class_names": ["class1", "class2"],
            }
        }
        names = multitask_column_names(cfg, 12)
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "NR-AR")
        self.assertEqual(names[-1], "SR-p53")

    def test_sider_multitask_fallback(self) -> None:
        cfg = {
            "dataset": {
                "dataset": "sider",
                "num_tasks": 27,
                "class_names": ["class1", "class2"],
            }
        }
        self.assertEqual(
            multitask_column_names(cfg, 27),
            [f"task_{i}" for i in range(27)],
        )

    def test_single_task_fallback(self) -> None:
        cfg = {"dataset": {"dataset": "bbbp", "num_tasks": 1}}
        self.assertEqual(multitask_column_names(cfg, 1), ["task_0"])


class ExtractPerTaskTestMetricsTests(unittest.TestCase):
    def test_extracts_rocauc_per_task(self) -> None:
        test_results = {
            "result_list_dict_each_task": [
                {"ROCAUC": 0.91},
                {"ROCAUC": 0.82},
            ]
        }
        out = extract_per_task_test_metrics(
            test_results,
            ["FDA_APPROVED", "CT_TOX"],
            "ROCAUC",
        )
        self.assertEqual(out, {"FDA_APPROVED": 0.91, "CT_TOX": 0.82})

    def test_extracts_rmse_per_task(self) -> None:
        test_results = {
            "result_list_dict_each_task": [
                {"RMSE": 1.2},
                {"RMSE": 0.8},
            ]
        }
        out = extract_per_task_test_metrics(
            test_results,
            ["task_0", "task_1"],
            "rmse",
        )
        self.assertEqual(out, {"task_0": 1.2, "task_1": 0.8})

    def test_molclr_style_task_list(self) -> None:
        test_results = {
            "ROCAUC": 0.865,
            "result_list_dict_each_task": [
                {"ROCAUC": 0.91},
                {"ROCAUC": 0.82},
            ],
        }
        out = extract_per_task_test_metrics(
            test_results,
            ["FDA_APPROVED", "CT_TOX"],
            "ROCAUC",
        )
        self.assertEqual(out, {"FDA_APPROVED": 0.91, "CT_TOX": 0.82})
        test_results = {"result_list_dict_each_task": [{"ROCAUC": 0.5}]}
        out = extract_per_task_test_metrics(
            test_results,
            ["FDA_APPROVED", "CT_TOX"],
            "ROCAUC",
        )
        self.assertEqual(out, {"FDA_APPROVED": 0.5})


if __name__ == "__main__":
    unittest.main()
