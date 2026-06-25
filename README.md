## Requirements

### Operating System

The code has been tested under **Ubuntu 24.04 LTS (WSL2)**.

### Hardware and Expected Runtime

- **Pre-training:** The full pre-training procedure was conducted on NVIDIA A100 GPUs.
- **Fine-tuning and inference:** A **GPU is required** for fine-tuning and downstream experiments (e.g. a single NVIDIA GeForce RTX 4090). A typical downstream fine-tuning run completes within **several hours** on one GPU, depending on the task configuration.

## Quick start

```bash
conda create -n chemvl -y python=3.9
conda activate chemvl
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install rdkit-pypi==2022.9.5
pip install -r requirements.txt --no-deps

export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data
python tools/hf_download.py download
python tools/hf_download.py unpack

python finetune_moleculenet.py \
  --config configs/tutorials/moleculenet_bbbp_classification_scaffold_PT.json

python interpret.py list
```

See the sections below for optional PyG setup, external baselines, notebooks, and ablation configs.

## Environment Installation Guide

1. Create a clean conda environment
  ```
    conda create -n chemvl -y python=3.9
    conda activate chemvl
    python -m pip install --upgrade pip
  ```
2. Install PyTorch (GPU version, CUDA 11.7)
  ```
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
    Verify the installation:
3. Install RDKit
  ```
    pip install rdkit-pypi==2022.9.5
  ```
4. Install remaining Python dependencies
  ```
    pip install -r requirements.txt --no-deps
  ```
5. (Optional) Enable MolCLR-style graph pretraining (PyG stack)
  ```bash
    # MolCLR graph encoder depends on the PyTorch Geometric ecosystem.
    # The commands below are for torch==1.13.1+cu117.

    # (a) install prebuilt PyG operator wheels (MUST match torch/cu version)
    pip install --no-cache-dir --only-binary=:all: \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

    # (b) install torch-geometric
    pip install --no-cache-dir torch-geometric==2.3.1
  ```

## Data root, layout, and Hugging Face downloads

Large artifacts (datasets, checkpoints, logs, prior-knowledge caches) should live **outside** the git clone, under a single root directory.

### 0. `CHEMVL_DATA_ROOT` — where large files live

Pick **one directory** on your machine for datasets, checkpoints, logs, and optional caches. Training code reads this root through **`get_data_root()`** in [`utils/path_utils.py`](utils/path_utils.py).

- **Set `CHEMVL_DATA_ROOT` before running experiments** (required on any machine that is not using the default `~/chemvl-data` fallback).
- **Prior-knowledge `.pkl` caches** default to `{CHEMVL_DATA_ROOT}/cache_for_knowledge/` when `model.knowledge_memory_path` is not set in JSON (see `get_knowledge_cache_dir()` in the same module).

```bash
export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data
```

**Important:** JSON configs still store **literal paths** for `dataset.dataroot`, `log_dir_base`, and `model.resume`. Setting `CHEMVL_DATA_ROOT` does **not** auto-rewrite those fields — it affects `get_data_root()`, the default knowledge-cache directory, and legacy path remapping at config load time.

### 1. Recommended directory layout under `CHEMVL_DATA_ROOT`

Keep **the same relative paths** locally so you can mirror one snapshot into `CHEMVL_DATA_ROOT`:


| Path (relative to `CHEMVL_DATA_ROOT`) | Typical use                                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `archives/`                           | `.tar.zst` bundles after Hub download; run `python tools/hf_download.py unpack` to restore sibling directories                 |
| `checkpoints/`                        | Pretrained ChemVL / CLIP backbones and finetuned weights (`model.resume`)                                                      |
| `finetuning_datasets/`                | MPP, MoleculeACE, and other downstream CSV + image trees (`dataset.dataroot`)                                                  |
| `pretraining_datasets/`               | Graph pretraining metadata (`10M-106mds/` on Hub); image PNGs generated locally via `tools/datasets/render_pretrain_images.py` |
| `cache_for_knowledge/`                | Prior-knowledge memory pickles (created on demand)                                                                             |
| `results/`                            | Training logs and outputs (`log_dir_base` in configs)                                                                          |
| `descriptor_info.pkl`                 | Descriptor metadata at the **root** of `CHEMVL_DATA_ROOT` (used by descriptor workflows)                                       |


### 2. Download assets from Hugging Face

Install the Hub client if needed (`pip install huggingface_hub` or use `huggingface-cli` from the same ecosystem).

**One-shot download of the published snapshot into your data root** (from the repository root, with `CHEMVL_DATA_ROOT` set):

```bash
export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data
python tools/hf_download.py download
```

This runs `snapshot_download` from the Hub dataset
[yzhuang1997/chemvl-data](https://huggingface.co/datasets/yzhuang1997/chemvl-data)
into `CHEMVL_DATA_ROOT`, preserving the directory layout above.

Then extract packaged archives (multi-file dataset trees ship as `.tar.zst`):

```bash
python tools/hf_download.py unpack
```

See `tools/hf_download.py` for environment-variable overrides.

**Public Hugging Face dataset card (English):** [`docs/data/HF_DATASET_CARD.md`](docs/data/HF_DATASET_CARD.md)

**Image pretraining PNGs** (`pretraining_datasets/images-10M@224px/`) are not on the Hub; after download + unpack, generate from metadata:

```bash
python tools/datasets/render_pretrain_images.py render --split train --skip-existing
python tools/datasets/render_pretrain_images.py render --split test --skip-existing
```

See [`docs/data/HF_DATASET_CARD.md`](docs/data/HF_DATASET_CARD.md) for the full image-pretraining workflow.

## Fine-tuning on downstream tasks

Main entry points:

- `finetune_moleculenet.py` — MoleculeNet / MPP benchmarks (`utils/finetune_utils.py` pipeline).
- `finetune_moleculeace.py` — MoleculeACE (MolMCL split protocol).

Training uses **validation only** for checkpoint selection; **test** is evaluated once after training on `train_best.pth` and `valid_best.pth` (see `result.json` fields `final_test_train_best` / `final_test_valid_best`).

### Run MoleculeNet Benchmark (single config)

```bash
# Fine-tune MoleculeNet BBBP (binary classification, scaffold split)
python finetune_moleculenet.py --config configs/tutorials/moleculenet_bbbp_classification_scaffold_PT.json
```

### Run MoleculeNet Benchmark (compose configs; later overrides earlier)

You can pass several `--config` paths; they are merged in order, and **later files win** on overlapping keys (e.g. the tutorial BBBP bundle plus a strategy JSON such as linear probing).

```bash
python finetune_moleculenet.py \
  --config configs/tutorials/moleculenet_bbbp_classification_scaffold_PT.json \
  --config configs/fragments/finetuning_strategy/linear_probing.json
```

### Run MoleculeACE Benchmark

```bash
python finetune_moleculeace.py --config configs/tutorials/moleculeace_chembl2047_ec50_FT.json
```

### External baselines (MolCLR / ImageMol / MolMCL under ChemVL)

Fair-comparison baselines run **inside the ChemVL shell** (`utils/external/finetune_external.py`): same splits and evaluation as the main fine-tuning CLIs, with method code from `external/MolCLR` and `external/MolMCL`. See [`docs/external/SUBMODULES.md`](docs/external/SUBMODULES.md) and per-method READMEs under `scripts/external/`.

**Prerequisites** (once per machine):

```bash
git submodule update --init external/MolMCL external/MolCLR
export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data
python tools/hf_download.py download
python tools/hf_download.py unpack
# checkpoints/external/: MolCLR-{GIN,GCN}.ckpt, ImageMol.pth.tar
# MolMCL zinc-{gps,gnn}_best.pt: obtain from MolMCL upstream or your fork if not in the Hub snapshot
```

**MolCLR & ImageMol** (chemvl env; GPU required):

```bash
bash scripts/external/molclr_under_chemvl/run_scaffold.sh
bash scripts/external/molclr_under_chemvl/run_random_scaffold.sh
bash scripts/external/imagemol_under_chemvl/run_scaffold.sh
bash scripts/external/imagemol_under_chemvl/run_random_scaffold.sh
```

**MolMCL** (separate `molmcl` conda env with PyG; see [`scripts/external/molmcl_under_chemvl/README.md`](scripts/external/molmcl_under_chemvl/README.md)):

```bash
conda activate molmcl
bash scripts/external/molmcl_under_chemvl/run_scaffold.sh
bash scripts/external/molmcl_under_chemvl/run_moleculeace_gin.sh
```

Dry-run without training: prefix any command with `DRY_RUN=1`. Results land under `{CHEMVL_DATA_ROOT}/results/baseline_methods/` (MoleculeNet) or `{CHEMVL_DATA_ROOT}/results/moleculeace/` (MoleculeACE MolMCL runs).

### Interpretability (preset showcase)

```bash
python interpret.py list
python interpret.py visual run --preset analysis/interpret/presets/case_moleculenet_curated.yaml
python interpret.py visual run --preset analysis/interpret/presets/case_moleculeace_curated.yaml --case CHEMBL1871_Ki_ac_delta3_0000_0000
python interpret.py visual run --preset analysis/interpret/presets/testset_moleculenet_bbbp.yaml --max-molecules 10
python interpret.py knowledge run --preset analysis/interpret/presets/knowledge_cases.yaml --case Cebaracetam
```

See [`analysis/interpret/README.md`](analysis/interpret/README.md) for all presets and checkpoint paths.

### Upstream descriptor inference

```bash
python pretrain_inference.py \
  -c ordinalclip/configs/default.yaml \
  -c ordinalclip/configs/base_cfgs/data_cfg/datasets/mol-10M-106mds/local.yaml
```

### Common config locations

- Compose fragments: `configs/fragments/` (`base/`, `splitting/`, `finetuning_strategy/`)
- Ablation studies: `configs/ablation_study/` (see topic `group_registry.json` under each subfolder)
- Tutorials (monolithic): `configs/tutorials/`

## Quick demo (Jupyter notebooks)

Optional demos with **pre-executed outputs** for browsing on GitHub. To re-run locally:

1. Set `export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data` (or write that path to `notebooks/.chemvl_data_root`).
2. The first code cell calls `configure_chemvl_data_root()` to auto-detect a data root with the pretraining checkpoint when the variable is unset.
3. Open and run from the repo root, or from `notebooks/` (both layouts are supported).

Prefer the CLI commands above for headless reproducibility.

- `notebooks/a_molecular_descriptor_inference.ipynb`
  - Descriptor-conditioned inference with the pretrained ChemVL model (`pretrain_inference.py` workflow).
- `notebooks/b_prompt-guided_visual_attention.ipynb`
  - Grad-CAM style attention under descriptor prompts (`interpret.py` visual showcase).
- `notebooks/c_context-aware_molecular_representation.ipynb`
  - t-SNE views of molecular representations under different descriptor prompts (pre-training and downstream BACE / CHEMBL1871).

## License

This project is released under the [MIT License](LICENSE.txt).

## Acknowledgments

ChemVL builds on and adapts code from several open-source projects. See each upstream repository for its license terms:

- [OpenAI CLIP](https://github.com/openai/CLIP) (MIT)
- [OrdinalCLIP](https://github.com/xk-huang/OrdinalCLIP) (MIT)
- [CoOp](https://github.com/KaiyangZhou/CoOp) (MIT)
- [MolCLR](https://github.com/yuyangw/MolCLR)
- [MolMCL](https://github.com/yuewan2/MolMCL)
- [MoleculeACE](https://github.com/molML/MoleculeACE)
- [ImageMol](https://github.com/HongxinXiang/ImageMol)
- [OpenMMLab mmcv](https://github.com/open-mmlab/mmcv) registry utilities (Apache-2.0)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (MIT)

External baseline reproduction uses Git submodules under `external/`; initialize them with `git submodule update --init --recursive`.
