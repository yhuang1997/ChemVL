MolMCL-style YAML templates used by ChemVL `finetune_external` (`model.molmcl.external_config`).

- **Paths**: reference from repo root, e.g. `configs/external/molmcl/base_config/moleculeace/chembl_gps.yaml`, or use the `{dataset}` placeholder in JSON (e.g. `.../moleculenet/{dataset}.yaml`) so one ChemVL config can sweep MoleculeNet tasks via batch overlays.
- **Upstream**: derived from `external/MolMCL/config/`; edit here for ChemVL-side defaults without changing the submodule.
