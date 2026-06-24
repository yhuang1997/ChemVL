# ImageMol inside the ChemVL fine-tuning shell

Entry: [`utils/external/finetune_external.py`](../../utils/external/finetune_external.py) with `model.finetune_backend == "imagemol_moleculenet"`.

- **Splits:** Same as ChemVL MoleculeNet (`dataset.split` → `get_split`).
- **Model:** ResNet18 + single `fc(num_tasks)`; BCEWithLogitsLoss for classification.
- **Pretraining:** `model.imagemol.resume` → ImageMol checkpoint (first 120 keys).
- **Augmentation:** Official CenterCrop pipeline ([`imagemol_transforms.py`](../../utils/external/imagemol/imagemol_transforms.py)).
- **Hyperparameters:** `configs/external/imagemol/params_imagemol.yaml`; ChemVL JSON `training` block does not override lr/batch/epochs (see `imagemol_external_config.py`).
- **config.json:** ChemVL shell fields plus `external_effective_hparams` written by the backend.

Batch scripts: [`scripts/external/imagemol_under_chemvl/README.md`](../../scripts/external/imagemol_under_chemvl/README.md).
