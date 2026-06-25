# ChemVL Data

Dataset repository for the ChemVL chemistry vision–language model ([code on GitHub](https://github.com/yhuang1997/ChemVL)).  
This snapshot provides **pretrained weights**, **downstream datasets** (MoleculeNet / MoleculeACE as 2D structure images), **finetuned checkpoints**, descriptor metadata, and knowledge-memory caches needed to reproduce the paper and run the official tutorials.

**License & citation:** see the ChemVL paper and GitHub repository (update before public release).

---

## Download

```bash
pip install huggingface_hub
export CHEMVL_DATA_ROOT=/path/to/chemvl-data

git clone https://github.com/yhuang1997/ChemVL.git
cd ChemVL
python tools/hf_download.py download
python tools/hf_download.py unpack
```

Set `CHEMVL_DATA_ROOT` to the directory where the snapshot was downloaded. Large directory trees are shipped as `**.tar.zst` archives** under `archives/`; `hf_download.py unpack` restores the layout expected by the training code.

---

## Layout (after unpack)

```text
CHEMVL_DATA_ROOT/
├── descriptor_info.pkl
├── archives/                              # download artifacts (optional to delete after unpack)
├── cache_for_knowledge/*.pkl
├── checkpoints/
│   ├── pretraining/RN50px224.ckpt         # image backbone
│   ├── pretraining/GIN.ckpt, GCN.ckpt     # graph backbones
│   ├── external/                          # MolCLR, ImageMol baselines
│   └── finetuning/                        # from archives/checkpoints_finetuning.tar.zst
├── finetuning_datasets/                   # from archives/finetuning_datasets.tar.zst
│   ├── MPP/classification/                # MoleculeNet (6 cls tasks)
│   ├── MPP/regression/                    # MoleculeNet (4 reg tasks)
│   └── MoleculeACE/                       # 30 ChEMBL targets
└── pretraining_datasets/
    ├── 10M-106mds/                        # graph + image pretrain metadata (on Hub)
    └── images-10M@224px/                  # generate locally (see below)
```

Update literal paths in JSON configs (`dataset.dataroot`, `model.resume`) to match your `CHEMVL_DATA_ROOT`.

---

## Image pretraining corpus (generate locally)

The ~10M PNG corpus for ChemVL-Image pretraining is **not** stored on the Hub (too large). The Hub snapshot includes `pretraining_datasets/10M-106mds/` (`mds.csv`, `train.txt`, `test.txt`). Generate 224×224 PNGs with the ChemVL repo (requires `rdkit-pypi==2022.9.5`, same as the main README):

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data

python tools/datasets/render_pretrain_images.py render --split train --skip-existing
python tools/datasets/render_pretrain_images.py render --split test --skip-existing
```

Output layout (matches `ordinalclip/configs/base_cfgs/data_cfg/datasets/mol-10M-106mds/local.yaml`):

```text
pretraining_datasets/images-10M@224px/train_data/{index}.png
pretraining_datasets/images-10M@224px/test_data/{index}.png
```

Rendering uses `utils/pretrain_image_render.py` (`MolFromSmiles` + RDKit `MolsToGridImage` at 224px), the same recipe as default downstream PNGs in ChemVL.

---

## Contents


| Category           | What you get                                                             |
| ------------------ | ------------------------------------------------------------------------ |
| Pretrained weights | ChemVL-Image (RN50), ChemVL-Graph (GIN/GCN)                              |
| Finetuned weights  | Preset checkpoints under `checkpoints/finetuning/presets/` (config + weights) |
| Downstream data    | Full MoleculeNet-10 and MoleculeACE-30 processed splits + 224×224 PNGs   |
| Pretraining        | `10M-106mds` metadata on Hub; image PNGs via `render_pretrain_images.py` |
| Other              | `descriptor_info.pkl`; MoleculeNet `cache_for_knowledge/*.pkl` (no MoleculeACE caches) |


---

## Quick start (after download)

```bash
# MoleculeNet BBBP fine-tuning
python finetune_moleculenet.py \
  --config configs/tutorials/moleculenet_bbbp_classification_scaffold_PT.json

# MoleculeACE fine-tuning
python finetune_moleculeace.py \
  --config configs/tutorials/moleculeace_chembl2047_ec50_FT.json

# Knowledge-attention showcase (single case)
python interpret.py knowledge run \
  --preset analysis/interpret/presets/knowledge_cases.yaml --case Cebaracetam
```

See the GitHub `README.md` for environment setup (Python 3.9, PyTorch 1.13.1+cu117, RDKit). **A GPU is required** for fine-tuning and inference.