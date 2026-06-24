# ImageMol under ChemVL (MoleculeNet baseline)

In-process ImageMol fine-tuning with ChemVL splits, metrics, and logging.

## Prerequisites

```bash
export CHEMVL_DATA_ROOT=/path/to/your/data
python tools/hf_download.py download
python tools/hf_download.py unpack   # checkpoints/external/ImageMol.pth.tar
```

## Single task

```bash
python utils/external/finetune_external.py \
  --config configs/external/imagemol/imagemol_moleculenet_bbbp.external.json
```

## Batch (MoleculeNet 10 × runseed)

```bash
export CHEMVL_DATA_ROOT=/path/to/your/data

bash scripts/external/imagemol_under_chemvl/run_scaffold.sh
bash scripts/external/imagemol_under_chemvl/run_random_scaffold.sh
```

Environment variables: same as [`molclr_under_chemvl/README.md`](../molclr_under_chemvl/README.md).

Batch configs: `configs/external/imagemol/imagemol_moleculenet_{scaffold,random_scaffold}.external.json`

Results: `{CHEMVL_DATA_ROOT}/results/baseline_methods/{EXP_NAME}/imagemol_moleculenet/<dataset>/`.

See [`docs/external/IMAGEMOL_CHEMVL_FINETUNE.md`](../../../docs/external/IMAGEMOL_CHEMVL_FINETUNE.md).
