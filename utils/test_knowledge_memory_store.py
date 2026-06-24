"""Tests for sharded knowledge memory store."""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from utils.knowledge_memory_store import (
    KM_KEYS_PER_SHARD,
    KM_SHARD_THRESHOLD,
    km_shard_dir,
    load_knowledge_memory_store,
    save_knowledge_memory_store,
)


class KnowledgeMemoryStoreTest(unittest.TestCase):
    def test_small_monolithic_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "toy_knowledge_memory_all.pkl")
            mem = {f"smi_{i}": torch.randn(208, 512) for i in range(10)}
            save_knowledge_memory_store(path, mem)
            loaded = load_knowledge_memory_store(path)
            self.assertEqual(set(loaded.keys()), set(mem.keys()))

    def test_large_sharded_roundtrip(self) -> None:
        n = KM_SHARD_THRESHOLD + KM_KEYS_PER_SHARD + 3
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "big_knowledge_memory_all.pkl")
            mem = {f"smi_{i}": torch.randn(208, 512) for i in range(n)}
            save_knowledge_memory_store(path, mem)
            self.assertTrue(Path(km_shard_dir(path)).is_dir())
            self.assertFalse(Path(path).is_file())
            loaded = load_knowledge_memory_store(path)
            self.assertEqual(len(loaded), n)

    def test_legacy_monolithic_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy_knowledge_memory_all.pkl"
            mem = {"a": torch.zeros(2, 3)}
            with open(path, "wb") as f:
                pickle.dump(mem, f)
            loaded = load_knowledge_memory_store(str(path))
            self.assertIn("a", loaded)


if __name__ == "__main__":
    unittest.main()
