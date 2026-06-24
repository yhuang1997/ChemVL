# MolMCL GNN inside the ChemVL fine-tuning shell

Entry: [`utils/external/finetune_external.py`](../../utils/external/finetune_external.py).

- **MoleculeACE splits:** Selected by `dataset.protocol` (e.g. `MolMCL`) via [`utils/moleculeace_molmcl.py`](../../utils/moleculeace_molmcl.py); independent of ChemVL `dataset.split` used for MoleculeNet.
- **Data:** Reads ChemVL MoleculeACE `*_processed_ac.csv` and builds PyG graphs in memory.
- **Optimization:** Loads `GNNPredictor` from `external/MolMCL`; hyperparameters come from the MolMCL yaml referenced by `model.molmcl.external_config`, with optional `model.molmcl.molmcl_overrides`. ChemVL `training.epochs` / `training.batch_size` do not override MolMCL yaml.
- **Logs:** ChemVL-style `train_val_test_history.csv`, `result.json`, and loss plots.

**Dependencies:** `torch_geometric`, `external/MolMCL` on `PYTHONPATH`, and RDKit (`rdkit-pypi==2022.9.5` recommended). See [`scripts/external/molmcl_under_chemvl/README.md`](../../scripts/external/molmcl_under_chemvl/README.md).

**Configs:** `configs/external/molmcl/`; `model.molmcl.external_config` may use placeholders such as `configs/external/molmcl/base_config/moleculenet/{dataset}.yaml`.

**MoleculeNet:** `dataset.benchmark == "moleculenet"`, backend `molmcl_moleculenet`. Data under `{dataset.dataroot}/MPP/classification|regression/<task>/processed/`. Splits follow `dataset.split` via [`utils/finetune_utils.py`](../../utils/finetune_utils.py) `get_split`. Classification metric: ROC-AUC; regression metrics follow `get_metric` (default `rmse`, QM sets use `mae`).
