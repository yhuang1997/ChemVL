# Ablation study configs

Unified layout for paper ablation experiments. Each **topic** is a subdirectory with its own `group_registry.json`.

## Layout

| Topic | Directory | Notes |
|-------|-----------|-------|
| Shared task JSON | `shared/datasets/` | MoleculeNet-10 task fields (`dataset`, `class_names`, …) |
| Pretraining source | `pretraining/` | Checkpoint / backbone grid |
| Misplacement mask | `misplacement_mask/` | FT + KGPT **ablation** variants |
| Dummy prompt learner | `dummy_prompt_learner/` | KGPT prompt-learner ablations |
| Descriptor-only | `descriptor_only/` | RDKit / text MLP (FT); public `group_registry.json` has **two** groups (`all`, `v2_107`); MLP sweep overlays live in `overlays/` for maintainer `--group` only |
| Depiction (image FT) | `depiction_image_ft/` | Layout / style / zoom variants |

## Driver

```bash
python scripts/ablation_study_run.py \
  --registry configs/ablation_study/pretraining/group_registry.json \
  --task-dir configs/ablation_study/shared/datasets \
  --dataset bbbp --group image_chemvl
```

Topic-specific shells live under `scripts/experiments/ablation_*/`.

## Public `group_registry` policy

- **FT baselines** (`*_ft_baseline`) are allowed.
- **KGPT baselines** (`*_kgpt_baseline`) are **omitted** from public registries — they correspond to main-table KGPT runs and are maintained separately under `configs/moleculenet/` (maintainer-only).
- KGPT **ablation** groups (e.g. misplacement, dummy prompt variants) remain in registries.

Overlay JSON under `misplacement_mask/overlays/kgpt/*_baseline.json` may exist for private reference but must not be wired in public `group_registry.json`.
