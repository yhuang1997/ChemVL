# MolCLR under ChemVL (MoleculeNet baseline)

In-process MolCLR GIN/GCN fine-tuning with ChemVL splits, metrics, and logging.

## Prerequisites

```bash
git submodule update --init external/MolCLR
export CHEMVL_DATA_ROOT=/path/to/your/data
python tools/hf_download.py download
python tools/hf_download.py unpack   # checkpoints/external/MolCLR-{GIN,GCN}.ckpt
```

## Single task

```bash
python utils/external/finetune_external.py \
  --config configs/external/molclr/molclr_gin_molclr_moleculenet_esol.external.json
```

## Batch (MoleculeNet 10 × 4 methods × runseed)

```bash
export CHEMVL_DATA_ROOT=/path/to/your/data

bash scripts/external/molclr_under_chemvl/run_scaffold.sh
bash scripts/external/molclr_under_chemvl/run_random_scaffold.sh
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHEMVL_DATA_ROOT` | (required) | Data and checkpoint root |
| `EXP_NAME` | `baseline-under-chemvl-scaffold` | Results subdirectory |
| `DATASET_LIST` | `configs/external/moleculenet/dataset_list_moleculenet10.txt` | Task list file |
| `DATASETS` | (empty = all) | Comma-separated subset |
| `RUNSEED_START/END` | `1` / `3` | Runseed range |
| `DRY_RUN` | `0` | `1` = print plan only |
| `NO_SKIP` | `0` | `1` = do not skip existing result.json |

## Config highlights

| Field | Description |
|-------|-------------|
| `model.finetune_backend` | `molclr_moleculenet` |
| `model.molclr.external_config` | `external/MolCLR/config_finetune.yaml` |
| `model.molclr.model_type` | `gin` / `gcn` |
| `model.molclr.resume` | `null` (scratch) or MolCLR checkpoint name |

Results: `{CHEMVL_DATA_ROOT}/results/baseline_methods/{EXP_NAME}/molclr_*_moleculenet/<dataset>/`.

See [`docs/external/MOLCLR_CHEMVL_FINETUNE.md`](../../../docs/external/MOLCLR_CHEMVL_FINETUNE.md).
