# MoleculeACE variants

Protocol templates for MoleculeACE fine-tuning (MolMCL split). Naming aligns with [`../moleculenet_variants/`](../moleculenet_variants/): `{modality}_chemvl_baseline_{strategy}.json` and `graph_gin_baseline_scratch.json`.

## Config files

| File | Strategy |
|------|----------|
| `image_chemvl_baseline_scratch.json` | Image RN50, train from scratch |
| `image_chemvl_baseline_pt.json` | Image, prompt tuning (PT) |
| `image_chemvl_baseline_ft.json` | Image, full fine-tune (FT) |
| `image_chemvl_baseline_kgpt.json` | Image, knowledge-guided prompt tuning (KGPT) |
| `graph_gin_baseline_scratch.json` | Graph GIN, train from scratch |
| `graph_gin_chemvl_baseline_ft.json` | Graph GIN ChemVL, FT |
| `graph_gin_chemvl_baseline_pt.json` | Graph GIN ChemVL, PT |
| `graph_gin_chemvl_baseline_kgpt.json` | Graph GIN ChemVL, KGPT |

`basic.version` matches the filename stem. Outputs go under `{basic.log_dir_base}/{version}/{CHEMBL_TARGET}/`.

## Registry

[`group_registry.json`](group_registry.json) maps group id → config path for batch drivers (same layout as moleculenet_variants).

## Usage

```bash
python finetune_moleculeace.py \
  --config configs/moleculeace_variants/image_chemvl_baseline_kgpt.json
```

Override `dataset.dataset` (target id) and paths for your `CHEMVL_DATA_ROOT`. See `DEFAULT_PROTOCOL_NOTE.txt` for MolMCL vs default protocol.

## Legacy naming

Old `*@molmcl_protocol.json` names live under [`../moleculeace/`](../moleculeace/) (Tier-1 manifest path). Existing runs may appear under `results/moleculeace/chemvl_kgpt@molmcl_protocol/`; new runs use directories named after `basic.version` above.

## Optimal hparams (planned)

Per-target **KGPT** configs with promoted hyperparameters and Hub checkpoint paths will be added here (similar to MoleculeNet-4 under `configs/moleculenet/`).

**Seeds:** formal per-target configs will set `training.runseed` and `training.pl_init_seed`. These templates omit them; `finetune_moleculeace.py` assigns random seeds when missing.
