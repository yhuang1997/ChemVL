#!/usr/bin/env bash
# Build a minimal CHEMVL_DATA_ROOT mirror simulating Hub post-unpack layout (YHU-45 path validation).
#
#   export CHEMVL_SRC=/mnt/d/wsl-data/chemvl
#   export CHEMVL_PUBLIC=/mnt/d/wsl-data/chemvl-public
#   bash dev/public_export/bootstrap_chemvl_public.sh
set -euo pipefail

CHEMVL_SRC="${CHEMVL_SRC:-/mnt/d/wsl-data/chemvl}"
CHEMVL_PUBLIC="${CHEMVL_PUBLIC:-/mnt/d/wsl-data/chemvl-public}"

if [[ ! -d "$CHEMVL_SRC" ]]; then
  echo "ERROR: CHEMVL_SRC not found: $CHEMVL_SRC" >&2
  exit 1
fi

mkdir -p "$CHEMVL_PUBLIC"
MISSING=()

copy_file() {
  local rel="$1"
  local src="$CHEMVL_SRC/$rel"
  local dst="$CHEMVL_PUBLIC/$rel"
  if [[ ! -e "$src" ]]; then
    MISSING+=("$rel")
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  rsync -a "$src" "$dst"
  echo "[ok] $rel"
}

copy_tree() {
  local rel="$1"
  local src="$CHEMVL_SRC/$rel"
  local dst="$CHEMVL_PUBLIC/$rel"
  if [[ ! -d "$src" ]]; then
    MISSING+=("$rel/")
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  rsync -a "$src/" "$dst/"
  echo "[ok] $rel/"
}

copy_glob() {
  local dir="$1"
  local pattern="$2"
  local found=0
  shopt -s nullglob
  for src in "$CHEMVL_SRC/$dir"/$pattern; do
    found=1
    rel="${src#"$CHEMVL_SRC/"}"
    copy_file "$rel" || true
  done
  shopt -u nullglob
  if [[ "$found" -eq 0 ]]; then
    MISSING+=("$dir/$pattern")
  fi
}

# --- singles (Hub root) ---
copy_file "descriptor_info.pkl" || true

# --- pretraining backbone ---
copy_file "checkpoints/pretraining/RN50px224.ckpt" || true

# --- MoleculeNet interpret presets (Hub: knowledge_prompt_tuning) ---
for ds in bbbp bace; do
  copy_tree "checkpoints/finetuning/presets/knowledge_prompt_tuning/$ds" || true
done

# --- MoleculeACE interpret runs (Hub: moleculeace/{TARGET}/{run_id}/best.pth + config.json) ---
declare -A MOLECULEACE_CKPT_SRC=(
  [CHEMBL204_Ki]="checkpoints/finetuning/moleculeace/CHEMBL204_Ki/2026_04_19_02_09/train_best.pth"
  [CHEMBL219_Ki]="checkpoints/finetuning/moleculeace/CHEMBL219_Ki/2026_04_19_07_45/valid_best.pth"
  [CHEMBL1871_Ki]="checkpoints/finetuning/moleculeace/CHEMBL1871_Ki/2026_05_18_23_06/train_best.pth"
  [CHEMBL2835_Ki]="checkpoints/finetuning/moleculeace/CHEMBL2835_Ki/2026_04_20_03_06/valid_best.pth"
)
declare -A MOLECULEACE_RUN_REL=(
  [CHEMBL204_Ki]="checkpoints/finetuning/moleculeace/CHEMBL204_Ki/2026_04_19_02_09"
  [CHEMBL219_Ki]="checkpoints/finetuning/moleculeace/CHEMBL219_Ki/2026_04_19_07_45"
  [CHEMBL1871_Ki]="checkpoints/finetuning/moleculeace/CHEMBL1871_Ki/2026_05_18_23_06"
  [CHEMBL2835_Ki]="checkpoints/finetuning/moleculeace/CHEMBL2835_Ki/2026_04_20_03_06"
)

for target in "${!MOLECULEACE_RUN_REL[@]}"; do
  run_rel="${MOLECULEACE_RUN_REL[$target]}"
  ckpt_src_rel="${MOLECULEACE_CKPT_SRC[$target]}"
  cfg_src="$CHEMVL_SRC/$run_rel/config.json"
  ckpt_src="$CHEMVL_SRC/$ckpt_src_rel"
  out_dir="$CHEMVL_PUBLIC/$run_rel"
  if [[ ! -f "$cfg_src" ]]; then
    MISSING+=("$run_rel/config.json")
    continue
  fi
  if [[ ! -f "$ckpt_src" ]]; then
    MISSING+=("$ckpt_src_rel")
    continue
  fi
  mkdir -p "$out_dir"
  cp -a "$cfg_src" "$out_dir/config.json"
  cp -a "$ckpt_src" "$out_dir/best.pth"
  echo "[ok] $run_rel/{config.json,best.pth}"
done

# --- downstream data (tutorials + interpret) ---
copy_tree "finetuning_datasets/MPP/classification/bbbp" || true
copy_tree "finetuning_datasets/MoleculeACE/CHEMBL2047_EC50" || true
for target in CHEMBL204_Ki CHEMBL219_Ki CHEMBL1871_Ki CHEMBL2835_Ki; do
  copy_tree "finetuning_datasets/MoleculeACE/$target" || true
done

# --- knowledge cache (MoleculeNet only, Hub scope) ---
copy_glob "cache_for_knowledge" "bbbp_*.pkl" || true
copy_glob "cache_for_knowledge" "bace_*.pkl" || true

# --- pretrain metadata for local-mini upstream inference ---
if [[ -d "$CHEMVL_SRC/pretraining_datasets/10M-106mds" ]]; then
  copy_tree "pretraining_datasets/10M-106mds" || true
else
  MISSING+=("pretraining_datasets/10M-106mds/")
fi

echo ""
echo "Mirror root: $CHEMVL_PUBLIC"
echo "Test with:   export CHEMVL_DATA_ROOT=$CHEMVL_PUBLIC"
if ((${#MISSING[@]} > 0)); then
  echo ""
  echo "Missing sources (${#MISSING[@]}):"
  printf '  - %s\n' "${MISSING[@]}"
  exit 1
fi
