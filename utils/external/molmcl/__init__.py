# SPDX-License-Identifier: MIT
"""ChemVL-side MolMCL integration (PyG backend, in-process finetune).

Official training code stays under ``external/MolMCL``. Entry:
``python utils/external/finetune_external.py --config ...``.

RDKit / ``torch_geometric`` load only when accessing graph symbols (PEP 562 ``__getattr__``).
"""

from __future__ import annotations

from utils.external.molmcl.molmcl_external_config import (
    load_merged_molmcl_config,
    molmcl_root_from_cfg,
    preview_molmcl_epochs_batch,
)

__all__ = [
    "MolMCLMoleculeACEBackend",
    "MoleculeAceSmilesGraphDataset",
    "get_moleculeace_processed_ac_csv",
    "load_moleculeace_tabular_multitask",
    "load_merged_molmcl_config",
    "molmcl_root_from_cfg",
    "preview_molmcl_epochs_batch",
]


def __getattr__(name: str):
    if name == "MolMCLMoleculeACEBackend":
        from utils.external.molmcl.molmcl_moleculeace_backend import MolMCLMoleculeACEBackend

        return MolMCLMoleculeACEBackend
    if name == "MoleculeAceSmilesGraphDataset":
        from utils.external.molmcl.moleculeace_pyg import MoleculeAceSmilesGraphDataset

        return MoleculeAceSmilesGraphDataset
    if name == "get_moleculeace_processed_ac_csv":
        from utils.external.molmcl.moleculeace_tabular import get_moleculeace_processed_ac_csv

        return get_moleculeace_processed_ac_csv
    if name == "load_moleculeace_tabular_multitask":
        from utils.external.molmcl.moleculeace_tabular import load_moleculeace_tabular_multitask

        return load_moleculeace_tabular_multitask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
