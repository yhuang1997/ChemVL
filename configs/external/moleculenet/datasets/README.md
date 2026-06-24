# Per-task MoleculeNet metadata for batch overlays

Each `<key>.json` supplies `task_type`, `num_tasks`, and on-disk `dataset` folder name
(e.g. key `lipo` → `"dataset": "lipophilicity"`).

**`dataroot` in these files** is the ImageMol layout root (`.../MPP/classification` or
`.../MPP/regression`). Graph backends (MolCLR) ignore it and use the parent
`finetuning_datasets` from the base external JSON.
