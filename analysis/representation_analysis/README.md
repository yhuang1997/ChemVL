# Representation analysis (downstream t-SNE)

Preset-driven CLI for **notebook c** (`notebooks/c_context-aware_molecular_representation.ipynb`). Supports **`mode: downstream`** only: extract image/text features from a fine-tuned checkpoint, fuse with α, run t-SNE, and export traceable artifacts.

Pretraining t-SNE stays in the notebook via `utils/tsne_utils.py` and does **not** use this package.

## Quick start

```bash
python -m analysis.representation_analysis \
  --preset analysis/representation_analysis/presets/downstream_tsne_bace.json

# Notebook workflow: render template first
# notebooks/presets/c_downstream_bace_notebook.json → results/notebooks/c_bace_preset_resolved.json
python -m analysis.representation_analysis \
  --preset results/notebooks/c_bace_preset_resolved.json
```

CHEMBL1871 regression (α montage + activity-cliff highlights): template `notebooks/presets/c_downstream_chembl1871_notebook.json`.

## Preset fields

- **`common.model_ckpt`:** Downstream fine-tuned weights.
- **`common.pretraining_resume`:** Backbone checkpoint for `load_model(cfg)`.
- **`downstream.combined_alpha_list`:** Image/text fusion coefficients.
- **`downstream.task_type`:** `classification` or `regression`.
- **Regression optional:** `tsne_alpha_montage`, `overlay_ac_mols`, `tsne_metrics` (LS@k / NP@k).

## Outputs

Under `outputs.out_dir`:

- `plots/tsne/*.png|*.svg` — main panels; regression may include montage / activity-cliff overlays
- `tsne_source_data.csv` — point-level audit data
- `downstream_tsne_metrics.csv` — regression metrics (optional)
- `manifest.json` — preset snapshot and artifact paths

## Modules

| Path | Role |
|------|------|
| `cli.py` / `__main__.py` | CLI entry |
| `core/pipeline.py` | Downstream orchestration |
| `features/feature_builder.py` | Feature extraction |
| `clustering/reducer.py` | t-SNE / UMAP |
| `viz/plots.py` | Classification / regression plots |
| `metrics/embedding_neighbor_metrics.py` | LS@k, NP@k |
| `metrics/moleculeace_cliff.py` | MoleculeACE cliff parsing |
