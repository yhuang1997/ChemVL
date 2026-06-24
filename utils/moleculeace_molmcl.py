# SPDX-License-Identifier: MIT
# Protocol logic adapted from MolMCL (https://github.com/yuewan2/MolMCL):
#   molmcl/splitters.py (moleculeace_split)
#   molmcl/utils/moleculeace.py (ActivityCliffs, similarity, Tanimoto, etc.)
# Original MoleculeACE / MolMCL use ``from Levenshtein import distance`` (PyPI: ``Levenshtein`` /
# legacy ``python-Levenshtein``). If that package is installed, this module uses it; otherwise
# a pure-Python fallback is used (slower on large sets).

from __future__ import annotations

from typing import Callable, List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric as GraphFramework
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

try:
    from Levenshtein import distance as _levenshtein_distance  # type: ignore[import-not-found]
except ImportError:

    def _levenshtein_distance(a: str, b: str) -> int:
        """Edit distance (Wagner–Fischer), same semantics as ``Levenshtein.distance``."""
        na, nb = len(a), len(b)
        if na == 0:
            return nb
        if nb == 0:
            return na
        dp = list(range(nb + 1))
        for i in range(1, na + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, nb + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        return dp[nb]


def find_fc(a: float, b: float) -> float:
    return max(a, b) / min(a, b)


def get_fc(bioactivity: List[float], in_log10: bool = True) -> np.ndarray:
    bioactivity = 10 ** abs(np.array(bioactivity)) if in_log10 else np.array(bioactivity)
    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    for i in range(act_len):
        for j in range(i, act_len):
            m[i, j] = find_fc(bioactivity[i], bioactivity[j])
    m = m + m.T - np.diag(np.diag(m))
    np.fill_diagonal(m, 0)
    return m


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, n_bits: int = 1024, use_scaffold: bool = False) -> np.ndarray:
    db_fp = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise ValueError(f"Invalid SMILES for Tanimoto matrix: {smi!r}")
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
        db_fp[smi] = fp
    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    for i in range(smi_len):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]], db_fp[smiles[j]])
    m = m + m.T - np.diag(np.diag(m))
    np.fill_diagonal(m, 0)
    return m


def get_scaffold(smi: str, generic: bool = False) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi!r}")
    if generic:
        return Chem.MolToSmiles(GraphFramework(mol))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def get_scaffold_matrix(smiles: List[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    db_scaf = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise ValueError(f"Invalid SMILES: {smi!r}")
        try:
            skeleton = GraphFramework(m)
        except Exception:
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=n_bits)
        db_scaf[smi] = skeleton_fp
    smi_len = len(smiles)
    mat = np.zeros([smi_len, smi_len])
    for i in range(smi_len):
        for j in range(i, smi_len):
            mat[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]], db_scaf[smiles[j]])
    mat = mat + mat.T - np.diag(np.diag(mat))
    np.fill_diagonal(mat, 0)
    return mat


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True) -> np.ndarray:
    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    for i in range(smi_len):
        for j in range(i, smi_len):
            d = _levenshtein_distance(smiles[i], smiles[j])
            if normalize:
                m[i, j] = d / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = d
    m = m + m.T - np.diag(np.diag(m))
    m = 1 - m
    np.fill_diagonal(m, 0)
    return m


def moleculeace_similarity(smiles: List[str], similarity: float = 0.9) -> np.ndarray:
    m_tani = get_tanimoto_matrix(smiles) >= similarity
    m_scaff = get_scaffold_matrix(smiles) >= similarity
    m_leve = get_levenshtein_matrix(smiles) >= similarity
    return (m_tani + m_scaff + m_leve).astype(int)


class ActivityCliffs:
    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.ndarray]):
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if not isinstance(bioactivity, list) else bioactivity
        self.cliffs = None

    def find_cliffs(
        self,
        similarity: float = 0.9,
        potency_fold: float = 10,
        in_log10: bool = True,
        custom_cliff_function: Callable = None,
        mmp: bool = False,
        reverse: bool = False,
    ):
        if mmp:
            raise NotImplementedError("MMP similarity is not ported in ChemVL moleculeace_molmcl.")
        sim = moleculeace_similarity(self.smiles, similarity)
        if custom_cliff_function is not None:
            sim = custom_cliff_function(self.smiles, similarity)
        self.sim = sim
        fc = (get_fc(self.bioactivity, in_log10=in_log10) > potency_fold).astype(int)
        if not reverse:
            self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)
        else:
            self.cliffs = np.logical_and(sim == 1, fc == 0).astype(int)
        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        if self.cliffs is None:
            self.find_cliffs(**kwargs)
        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        return list((sum(self.cliffs) > 0).astype(int))


def find_stereochemical_siblings(smiles: List[str]) -> List[str]:
    """Same idea as MoleculeACE MoleculeACE/benchmark/data_prep.py (Tanimoto == 1 pairs)."""
    lower = np.tril(get_tanimoto_matrix(smiles, radius=4, n_bits=4096), k=0)
    identical = np.where(lower == 1)
    identical_pairs = [
        [smiles[identical[0][i]], smiles[identical[1][i]]] for i in range(len(identical[0]))
    ]
    return list(set(sum(identical_pairs, [])))


def moleculeace_split(
    smiles: List[str],
    bioactivity: List[float],
    in_log10: bool = True,
    n_clusters: int = 5,
    val_size: float = 0.1,
    test_size: float = 0.1,
    similarity: float = 0.9,
    potency_fold: int = 10,
    remove_stereo: bool = False,
):
    """
    MolMCL-compatible split indices (train, val, test).
    Index order follows MolMCL: concatenation over spectral clusters.
    """
    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    if not in_log10:
        bioactivity = (-np.log10(np.asarray(bioactivity, dtype=float))).tolist()

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    spectral = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, affinity="precomputed")
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx, val_idx = [], [], []
    for cluster in range(n_clusters):
        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(
                cluster_idx,
                test_size=test_size + val_size,
                random_state=RANDOM_SEED,
                stratify=clust_cliff_mols,
                shuffle=True,
            )
        else:
            clust_train_idx, clust_test_idx = train_test_split(
                cluster_idx,
                test_size=test_size + val_size,
                random_state=RANDOM_SEED,
                shuffle=True,
            )

        clust_test_idx, clust_val_idx = train_test_split(
            clust_test_idx,
            test_size=test_size / (test_size + val_size),
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        train_idx.extend(clust_train_idx.tolist())
        val_idx.extend(clust_val_idx.tolist())
        test_idx.extend(clust_test_idx.tolist())

    return train_idx, val_idx, test_idx
