# Config fragments (composable JSON pieces)

These JSON files are **not** standalone training entrypoints. Pass them **after** a task or tutorial config so later files override earlier keys:

```bash
python finetune_moleculenet.py \
  --config configs/fragments/base/classification/bbbp.json \
  --config configs/fragments/splitting/scaffold.json \
  --config configs/fragments/finetuning_strategy/prompt_tuning.json
```

For quick starts, prefer monolithic files under `configs/tutorials/`.

## `finetuning_strategy/` filename vs runtime field

| Fragment file | `training.finetune_strategy` in JSON |
|---------------|-------------------------------------|
| `linear_probing.json` | `linear_probing` |
| `prompt_tuning.json` | `text_prompt_tuning` |
| `knowledge-enhanced_prompt_tuning.json` | `prior_guided_tuning` |
| `fully_tuning.json` | `fully_tuning` |

Only the **filename** was renamed for `prompt_tuning` and `knowledge-enhanced_prompt_tuning`; Python code still uses the original strategy strings.
