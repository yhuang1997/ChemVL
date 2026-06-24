# External repositories (Git submodules)

Baseline and reference implementations are pinned under `external/` as **Git submodules**. Remotes point to maintained forks for reproducibility.


| Directory | Fork (origin) | Upstream |
| --------- | ------------- | -------- |
| `external/MolMCL` | [yhuang1997/MolMCL](https://github.com/yhuang1997/MolMCL) | [yuewan2/MolMCL](https://github.com/yuewan2/MolMCL) |
| `external/MoleculeACE` | [yhuang1997/MoleculeACE](https://github.com/yhuang1997/MoleculeACE) | [molML/MoleculeACE](https://github.com/molML/MoleculeACE) |
| `external/MolCLR` | [yhuang1997/MolCLR](https://github.com/yhuang1997/MolCLR) | [yuyangw/MolCLR](https://github.com/yuyangw/MolCLR) (default branch `master`) |

## First clone

```bash
git clone --recurse-submodules <ChemVL repo URL>
# If you already cloned without submodules:
git submodule update --init --recursive
```

## Bump a submodule pointer after fork changes

```bash
cd external/MolMCL   # or MoleculeACE / MolCLR
git checkout main    # or master for MolCLR
git pull origin main
# … develop, commit, push to your fork …
cd ../..
git add external/MolMCL
git commit -m "chore(external): bump MolMCL submodule"
```

## Baselines under ChemVL (MolMCL / MolCLR / ImageMol)

Before first run:

```bash
git submodule update --init external/MolMCL external/MolCLR
export CHEMVL_DATA_ROOT=/path/to/your/data
python tools/hf_download.py download
python tools/hf_download.py unpack
# checkpoints/external/: MolCLR-{GIN,GCN}.ckpt, ImageMol.pth.tar, zinc-{gps,gnn}_best.pt (see root README)
```

ChemVL-side integration lives in [`utils/external/`](../../utils/external/) and [`scripts/external/`](../../scripts/external/).

- MolMCL: [`MOLMCL_CHEMVL_FINETUNE.md`](MOLMCL_CHEMVL_FINETUNE.md), [`MOLMCL_FAIR_EXPERIMENTS.md`](MOLMCL_FAIR_EXPERIMENTS.md)
- MolCLR / ImageMol: READMEs under `scripts/external/molclr_under_chemvl/` and `imagemol_under_chemvl/`
