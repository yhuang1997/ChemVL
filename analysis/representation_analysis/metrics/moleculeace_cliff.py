"""MoleculeACE ``cliff_mol`` flags for downstream regression t-SNE AC overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def resolve_moleculeace_raw_csv_from_processed(processed_csv: str) -> Optional[Path]:
    """``.../MoleculeACE/{TARGET}/processed/*_processed_ac.csv`` → ``.../MoleculeACE/raw/{TARGET}.csv``."""
    p = Path(processed_csv).expanduser().resolve()
    if p.parent.name != "processed":
        return None
    target_dir = p.parent.parent
    ace_root = target_dir.parent
    cand = ace_root / "raw" / f"{target_dir.name}.csv"
    return cand if cand.is_file() else None


def load_cliff_mol_flags(
    raw_csv: Path,
    smiles: List[str],
    *,
    smiles_col: str = "smiles",
    cliff_col: str = "cliff_mol",
) -> Optional[np.ndarray]:
    """Return length-``n`` int vector in ``{0,1}`` aligned to ``smiles``; ``None`` if unavailable."""
    if not raw_csv.is_file():
        return None
    df = pd.read_csv(raw_csv)
    if smiles_col not in df.columns or cliff_col not in df.columns:
        return None
    mapping: Dict[str, int] = {}
    for s, v in zip(df[smiles_col].astype(str), pd.to_numeric(df[cliff_col], errors="coerce")):
        if np.isfinite(v):
            mapping[s] = int(v)
    out = np.zeros(len(smiles), dtype=np.int32)
    for i, s in enumerate(smiles):
        out[i] = int(mapping.get(str(s), 0))
    return out
