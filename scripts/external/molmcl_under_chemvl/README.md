# MolMCL under ChemVL (in-process)

Batch entry points for MolMCL GNN inside the ChemVL shell (`utils/external/finetune_external.py`).

Configs: `configs/external/molmcl/`

## Prerequisites

```bash
git submodule update --init external/MolMCL
conda activate molmcl
export CHEMVL_DATA_ROOT=/path/to/your/data
python tools/hf_download.py download
python tools/hf_download.py unpack
# checkpoints/external/: zinc-{gps,gnn}_best.pt (see root README if not in Hub snapshot)
```

## MoleculeNet batch

```bash
conda activate molmcl
export CHEMVL_DATA_ROOT=/path/to/your/data

bash scripts/external/molmcl_under_chemvl/run_scaffold.sh
bash scripts/external/molmcl_under_chemvl/run_random_scaffold.sh

RUN_GPS=0 bash scripts/external/molmcl_under_chemvl/run_scaffold.sh
DATASETS=esol,freesolv,lipo,qm7 bash scripts/external/molmcl_under_chemvl/run_scaffold.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_GIN` / `RUN_GPS` | `1` | Backbone toggles |
| `DATASETS` | (empty) | Comma-separated subset |
| `EXP_NAME` | `baseline-under-chemvl-scaffold` | Results subdirectory |
| `SMOKE` | `0` | Smoke runs use a separate results folder |
| `DRY_RUN` / `NO_SKIP` | `0` | Batch control |

## MoleculeACE

```bash
conda activate molmcl
export CHEMVL_DATA_ROOT=/path/to/your/data

bash scripts/external/molmcl_under_chemvl/run_moleculeace_gin.sh
bash scripts/external/molmcl_under_chemvl/run_moleculeace_gps.sh
```

## Dependencies

Use conda env **`molmcl`** (PyG stack). RDKit **`rdkit-pypi==2022.9.5`** recommended — see [`requirements-rdkit-molmcl.txt`](./requirements-rdkit-molmcl.txt).

Docs: [`docs/external/MOLMCL_CHEMVL_FINETUNE.md`](../../../docs/external/MOLMCL_CHEMVL_FINETUNE.md)
