# Interpret showcase presets

Reader-facing presets for reproducing case studies and lightweight interpretability figures. Internal bundle provenance appears only in YAML `# Provenance:` comments—not in CLI examples or preset `id` fields.

## Presets


| File                                    | Mode              | Description                                                 |
| --------------------------------------- | ----------------- | ----------------------------------------------------------- |
| `presets/case_moleculenet_curated.yaml` | `case_panel`      | BBBP/BACE curated cases: structure                          |
| `presets/case_moleculeace_curated.yaml` | `case_panel`      | 11 MoleculeACE cases: structure                             |
| `presets/testset_moleculenet_bbbp.yaml` | `testset_gallery` | Random sample from BBBP **test** split (panel per molecule) |
| `presets/knowledge_cases.yaml`          | `knowledge_cases` | BBBP/BACE knowledge-attention bar plots                     |


List presets:

```bash
python interpret.py list
```

## Requirements

- Set `CHEMVL_DATA_ROOT` and download data/checkpoints (`python tools/hf_download.py`).
- **MoleculeNet** presets default to `{CHEMVL_DATA_ROOT}/checkpoints/finetuning/presets/knowledge_prompt_tuning/{dataset}/`.
- **MoleculeACE** cases need per-target finetune runs under `{CHEMVL_DATA_ROOT}/checkpoints/finetuning/moleculeace/{target}/{run_id}/` (see preset `log_dir` / `ckpt`; Hub may not ship all runs).
- GPU recommended for inference.

## Examples

```bash
# MoleculeNet case panels (structure | upstream descriptors | finetuned)
python interpret.py visual run --preset analysis/interpret/presets/case_moleculenet_curated.yaml

# One MoleculeNet case
python interpret.py visual run --preset analysis/interpret/presets/case_moleculenet_curated.yaml --case Cebaracetam

# MoleculeACE high-res Grad-CAM (all 11 cases, or one by id)
python interpret.py visual run --preset analysis/interpret/presets/case_moleculeace_curated.yaml
python interpret.py visual run --preset analysis/interpret/presets/case_moleculeace_curated.yaml \
  --case CHEMBL1871_Ki_ac_delta3_0000_0000

# BBBP test-split gallery (sample N molecules from test split)
python interpret.py visual run --preset analysis/interpret/presets/testset_moleculenet_bbbp.yaml --max-molecules 10

# Knowledge-attention bar plot (preset case)
python interpret.py knowledge run --preset analysis/interpret/presets/knowledge_cases.yaml --case Cebaracetam

# Knowledge-attention (direct SMILES + checkpoint)
python interpret.py knowledge run \
  --smiles 'C1=CC(=CC=C1C3CN(CC(N2CC(NCC2)=O)=O)C(C3)=O)Cl' \
  --ckpt "$CHEMVL_DATA_ROOT/checkpoints/finetuning/presets/knowledge_prompt_tuning/bbbp/ckpt.pth" \
  --dataset bbbp
```

## Output layout

Default output root: `results/interpret/<preset_id>/`

Each run writes PNG figures and `manifest.json`. Case-panel runs also emit `index.html`; testset gallery does not.

## Notes

- **Testset gallery** (`testset_moleculenet_bbbp.yaml`): uses the finetune checkpoint’s `config.json` split; adjust `split`, `seed`, `max_molecules`, or `molecule_ids` in the YAML to match your local sampling.
- **MoleculeACE**: each case uses `{CHEMVL_DATA_ROOT}/checkpoints/finetuning/moleculeace/{TARGET}/{run_id}/best.pth` (see Hub dataset layout in `docs/data/HF_DATASET_CARD.md`).
- Knowledge bar plots: `analysis/interpret/knowledge_bar_plot.py` (showcase DPI, default png).

