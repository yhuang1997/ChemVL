"""Smoke tests for canonical pretraining / default depiction rendering."""
from __future__ import annotations

import unittest

import numpy as np

from utils.pretrain_image_render import DEFAULT_CANVAS_PX, smiles_to_pretrain_pil


class TestPretrainImageRender(unittest.TestCase):
    def test_output_shape_rgb(self):
        img = smiles_to_pretrain_pil("c1ccccc1")
        arr = np.asarray(img)
        self.assertEqual(arr.shape, (DEFAULT_CANVAS_PX, DEFAULT_CANVAS_PX, 3))
        self.assertEqual(arr.dtype, np.uint8)

    def test_invalid_smiles_white_canvas(self):
        img = smiles_to_pretrain_pil("not_a_smiles")
        arr = np.asarray(img)
        self.assertEqual(arr.shape, (DEFAULT_CANVAS_PX, DEFAULT_CANVAS_PX, 3))
        self.assertTrue(np.all(arr == 255))

    def test_cco_regression_digest(self):
        """Stable depiction for ethanol under rdkit-pypi==2022.9.5."""
        import hashlib

        arr = np.asarray(smiles_to_pretrain_pil("CCO"))
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        self.assertEqual(digest, "7c89cb9fb8497fe2bb40522df1609fe79ae2d39d0fc87edc53d795f247d35096")


if __name__ == "__main__":
    unittest.main()
