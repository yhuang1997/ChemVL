"""Shared case-study molecules for visual / knowledge interpretability pipelines."""

from __future__ import annotations

from typing import Dict, List, Tuple

CASE_STUDY_MOLECULES: Dict[str, Dict[str, Dict[str, object]]] = {
    "bbbp": {
        "Cebaracetam": {
            "smiles": "C1=CC(=CC=C1C3CN(CC(N2CC(NCC2)=O)=O)C(C3)=O)Cl",
            "label": 1,
        },
        "Dinazafone": {
            "smiles": "C1=C(C(=CC=C1Cl)N(C(CNCC(C)=C)=O)C)C(C2=CC=CC=C2)=O",
            "label": 1,
        },
    },
    "bace_domain": {
        "CHEMBL1092788": {
            "smiles": "Fc1ccc(F)cc1-c1cc(ccc1)[C@]1(N=C(N)N(C)C1=O)c1ccncc1",
            "label": 1,
        },
        "3ivi": {
            "smiles": "Fc1cc(cc(F)c1)C[C@H](NC(=O)C)[C@H](O)C[NH2+][C@]1(CCc2n[nH]cc2C1)c1cc(ccc1)C(C)(C)C",
            "label": 1,
        },
    },
    "clintox": {
        "c1nc2c(nc(nc2n1[C@@H]3C[C@@H](C=C3)CO)N)NC4CC4": {
            "smiles": "c1nc2c(nc(nc2n1[C@@H]3C[C@@H](C=C3)CO)N)NC4CC4",
            "label": 1,
        },
        "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)CO)O)NC=[NH+]C[C@H]2O": {
            "smiles": "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)CO)O)NC=[NH+]C[C@H]2O",
            "label": 1,
        },
    },
}

# Backward-compatible alias used by 03_downstream_inference_and_interpretion.py
specific_molecules = CASE_STUDY_MOLECULES


def get_curated_case_molecules() -> List[Tuple[str, str, str, int]]:
    """Return (checkpoint_dataset, name, smiles, label) for curated interpret cases."""
    rows: List[Tuple[str, str, str, int]] = []
    for name, info in CASE_STUDY_MOLECULES["bbbp"].items():
        rows.append(("bbbp", name, str(info["smiles"]), int(info["label"])))
    for name, info in CASE_STUDY_MOLECULES["bace_domain"].items():
        rows.append(("bace", name, str(info["smiles"]), int(info["label"])))
    return rows


def get_in_domain_fig4_molecules() -> List[Dict[str, object]]:
    """Return fig4b in-domain molecule records: dataset, name, smiles, label."""
    records: List[Dict[str, object]] = []
    for name, info in CASE_STUDY_MOLECULES["bbbp"].items():
        records.append(
            {"dataset": "bbbp", "name": name, "smiles": info["smiles"], "label": info["label"]}
        )
    for name, info in CASE_STUDY_MOLECULES["bace_domain"].items():
        records.append(
            {"dataset": "bace", "name": name, "smiles": info["smiles"], "label": info["label"]}
        )
    return records
