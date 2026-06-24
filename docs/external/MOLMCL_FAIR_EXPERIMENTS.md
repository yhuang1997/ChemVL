# MolMCL fair-comparison notes (under ChemVL)

This document defines what is held constant when comparing MolMCL baselines inside the ChemVL shell.

- **In-process fine-tuning (recommended):** [`utils/external/finetune_external.py`](../../utils/external/finetune_external.py) and [MOLMCL_CHEMVL_FINETUNE.md](MOLMCL_CHEMVL_FINETUNE.md) — same MoleculeACE MolMCL splits, ChemVL logs, and `result.json`.
- **Batch entry points:** [`scripts/external/molmcl_under_chemvl/README.md`](../../scripts/external/molmcl_under_chemvl/README.md).

## What is matched

- **Data layout:** MoleculeACE uses ChemVL `*_processed_ac.csv` and image trees under `CHEMVL_DATA_ROOT`.
- **MoleculeACE splits:** `protocol=MolMCL` uses [`utils/moleculeace_molmcl.py`](../../utils/moleculeace_molmcl.py) (`moleculeace_split`), aligned with MolMCL official CHEMBL logic.
- **Reporting:** ChemVL `result.json` / `train_val_test_history.csv` fields; MolMCL GNN backend regression keys match [`models/evaluate.py`](../../models/evaluate.py).

## What is not matched

- **Hyperparameters:** MolMCL yaml defaults may differ from ChemVL CLIP fine-tuning unless an experiment explicitly aligns them.
- **Official subprocess reproduction:** This repo documents the ChemVL-shell `finetune_external.py` path only.

## Randomness

- ChemVL: `training.runseed` and `fix_train_random_seed`.
- MoleculeACE MolMCL splits: fixed `random_state` inside `moleculeace_molmcl`.

## Data paths

- MoleculeACE: `{CHEMVL_DATA_ROOT}/finetuning_datasets/MoleculeACE/{TASK}/processed/{TASK}_processed_ac.csv`.
