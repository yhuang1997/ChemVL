# MolCLR GNN inside the ChemVL fine-tuning shell

Entry: [`utils/external/finetune_external.py`](../../utils/external/finetune_external.py) with `model.finetune_backend == "molclr_moleculenet"`.

- **Splits:** Same as ChemVL MoleculeNet — `dataset.split` → [`utils/finetune_utils.py`](../../utils/finetune_utils.py) `get_split`.
- **Data:** ChemVL MPP `*_processed_ac.csv`; graphs built with AddHs ([`molclr_graph.py`](../../utils/external/molclr/molclr_graph.py)).
- **Model:** In-process `GINet` / `GCN` from `external/MolCLR`; hyperparameters from `model.molclr.external_config` → `config_finetune.yaml`.
- **Multi-task classification:** ClinTox / Tox21 / Sider use per-target fine-tuning; report mean ROC-AUC.
- **QM7:** Train-set z-score normalization + L1Loss; denormalize at inference.
- **Logs:** ChemVL `train_val_test_history.csv` and `result.json`.

Batch scripts: [`scripts/external/molclr_under_chemvl/README.md`](../../scripts/external/molclr_under_chemvl/README.md).

Configs: `configs/external/molclr/*.external.json`.
