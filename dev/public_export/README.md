# Public export tooling (YHU-45, private-only)

Tier-1：[`TIER1_ENTRY.md`](../docs/public_export/TIER1_ENTRY.md) · Tier-2：[`TIER2_CANDIDATES.md`](../docs/public_export/TIER2_CANDIDATES.md) · [`manifest.yaml`](../docs/public_export/manifest.yaml)

**Playground**：仅在 `ChemVL-public-staging` / `public/export-prep` 上迭代；**禁止** Agent 自行 `git push public`（见 [`ORPHAN_PUSH.md`](../docs/public_export/ORPHAN_PUSH.md)）。

## `sync.py`

从 git ref 把 manifest 路径 checkout 到当前 worktree：

```bash
cd /path/to/ChemVL-public-staging

# Tier 1 only (default)
python -m dev.public_export sync --dry-run
python -m dev.public_export sync --apply --source origin/main

# Tier 2: packages with status: included in manifest.yaml
python -m dev.public_export sync --dry-run --tier2-only
python -m dev.public_export sync --apply --tier2-only --source origin/main

# Refresh Tier 1 + included Tier 2 together
python -m dev.public_export sync --apply --tier2 --source origin/main
```

`--source` 默认为 `origin/main`（private 最新）；checkout 前请在 private 主库 `git fetch origin`。

**Tier 2 当前 included**（以 manifest 为准）：`chemvl_baselines_graph`、`external_baselines`、`representation_analysis`。

`external/**` 仅 checkout submodule **gitlinks**；克隆后需：

```bash
git submodule update --init --recursive external/MolMCL external/MoleculeACE external/MolCLR
```

## Path validation（YHU-45）

在 maintainer 机器上构建 Hub 布局镜像并跑 Tier-1 smoke：

```bash
export CHEMVL_SRC=/mnt/d/wsl-data/chemvl
export CHEMVL_PUBLIC=/mnt/d/wsl-data/chemvl-public
bash dev/public_export/bootstrap_chemvl_public.sh

export CHEMVL_DATA_ROOT=/mnt/d/wsl-data/chemvl-public
python dev/public_export/audit_paths.py
```

结果与 smoke 矩阵见 [`docs/public_export/PATH_VALIDATION.md`](../docs/public_export/PATH_VALIDATION.md)。
