#!/usr/bin/env bash
# Aggregate test metrics for MolMCL-under-ChemVL batch runs (layout: <RESULT_ROOT>/<version>/<dataset>/<timestamp>/).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${_SCRIPT_DIR}/../_common.sh"
external_baseline_repo_root "$_SCRIPT_DIR"
external_baseline_export_data_root
external_baseline_resolve_default_python

LOG_DIR_BASE="${LOG_DIR_BASE:-${CHEMVL_DATA_ROOT}/results/moleculeace}"
EXP_NAME="${EXP_NAME:-molmcl_under_chemvl}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

"$PY" scripts/moleculeace_batch_analyze.py \
  --root "${RESULT_ROOT}" \
  --out-stem molmcl_under_chemvl \
  "$@"
