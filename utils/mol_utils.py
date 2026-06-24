import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import tqdm
import os
from rdkit.Chem import Draw

keys = [
    'NHOHCount',
    'NOCount',
    'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles',
    'NumAliphaticRings',
    'NumAromaticCarbocycles',
    'NumAromaticHeterocycles',
    'NumAromaticRings',
    'NumHAcceptors',
    'NumHDonors',
    'NumHeteroatoms',
    'NumRadicalElectrons',
    'NumRotatableBonds',
    'NumSaturatedCarbocycles',
    'NumSaturatedHeterocycles',
    'NumSaturatedRings',
    'RingCount',
    'fr_Al_COO',
    'fr_Al_OH',
    'fr_Al_OH_noTert',
    'fr_ArN',
    'fr_Ar_COO',
    'fr_Ar_N',
    'fr_Ar_NH',
    'fr_Ar_OH',
    'fr_COO',
    'fr_COO2',
    'fr_C_O',
    'fr_C_O_noCOO',
    'fr_C_S',
    'fr_HOCCN',
    'fr_Imine',
    'fr_NH0',
    'fr_NH1',
    'fr_NH2',
    'fr_N_O',
    'fr_Ndealkylation1',
    'fr_Ndealkylation2',
    'fr_Nhpyrrole',
    'fr_SH',
    'fr_aldehyde',
    'fr_alkyl_carbamate',
    'fr_alkyl_halide',
    'fr_allylic_oxid',
    'fr_amide',
    'fr_amidine',
    'fr_aniline',
    'fr_aryl_methyl',
    'fr_azide',
    'fr_azo',
    'fr_barbitur',
    'fr_benzene',
    'fr_benzodiazepine',
    'fr_bicyclic',
    'fr_diazo',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_ester',
    'fr_ether',
    'fr_furan',
    'fr_guanido',
    'fr_halogen',
    'fr_hdrzine',
    'fr_hdrzone',
    'fr_imidazole',
    'fr_imide',
    'fr_isocyan',
    'fr_isothiocyan',
    'fr_ketone',
    'fr_ketone_Topliss',
    'fr_lactam',
    'fr_lactone',
    'fr_methoxy',
    'fr_morpholine',
    'fr_nitrile',
    'fr_nitro',
    'fr_nitro_arom',
    'fr_nitro_arom_nonortho',
    'fr_nitroso',
    'fr_oxazole',
    'fr_oxime',
    'fr_para_hydroxylation',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_phos_acid',
    'fr_phos_ester',
    'fr_piperdine',
    'fr_piperzine',
    'fr_priamide',
    'fr_prisulfonamd',
    'fr_pyridine',
    'fr_quatN',
    'fr_sulfide',
    'fr_sulfonamd',
    'fr_sulfone',
    'fr_term_acetylene',
    'fr_tetrazole',
    'fr_thiazole',
    'fr_thiocyan',
    'fr_thiophene',
    'fr_unbrch_alkane',
    'fr_urea',
    "SINGLE_BOND",
    "DOUBLE_BOND",
    "TRIPLE_BOND",
    "AROMATIC_BOND",
]

functional_groups_SMARTS = {
    'NHOHCount': '[N;H2,H1;!$(N~[!#6]);!$(N~[!#6])]',
    'NOCount': '[N;H0;$(N~[!#6]);!$(N~[!#6]~[!#6])]',
    'NumAliphaticCarbocycles': '[r;R][r;R][C;R1][C;R1][r;R][r;R]',
    'NumAliphaticHeterocycles': '[r;R][r;R][C;R1][C;R1][r;R;H1;!$(a)]',
    'NumAliphaticRings': '[r;R][r;R]',
    'NumAromaticCarbocycles': '[a;r6]',
    'NumAromaticHeterocycles': '[a;r6][!$([a;r6][N,O,S;R1])]',
    'NumAromaticRings': '[a;r]',
    'NumHAcceptors': '[#6;+0;!$([#6](~[O,N,S]));!$([#6][~;+0])]([!#6;!H0;!$([#7]);!$([#8]);!$([#16])])',
    'NumHDonors': '[N;H1,H2;!$(N~[!#6]);!$(N~[!#6]~[!#6])]([!#6;!H0;!$(C(=O)N);!$(C(=O)N-C);!$(C(=O)N-N)])',
    'NumHeteroatoms': '[!#6;!#1;!H0]',
    'NumRadicalElectrons': '[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#34;!#35;!#53]',
    'NumRotatableBonds': '[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#34;!#35;!#53]',
    'NumSaturatedCarbocycles': '[r;R][r;R][C;R1][C;R1][r;R][r;R]',
    'NumSaturatedHeterocycles': '[r;R][r;R][C;R1][C;R1][r;R;H1;!$(a)]',
    'NumSaturatedRings': '[r;R][r;R]',
    'RingCount': '[r]',
    'fr_Al_COO': '[C;$(C(=O)[O,N,S])]([O,N,S;H1;!$(C=O)])=[O,S]',
    'fr_Al_OH': '[CX4][OX2H]',
    'fr_Al_OH_noTert': '[CX4;H1,H0;!$(C(C)(C)C)][OX2H]',
    'fr_ArN': '[NX3][#6]~[#6]',
    'fr_Ar_COO': '[C;$(C(=O)[O,N,S])](=[O,S])[#6;R]',
    'fr_Ar_N': '[#6][N;H1,H2;!$(N~[!#6]);!$(N~[!#6]~[!#6])][#6]',
    'fr_Ar_NH': '[nH][#6](=[N;H1,H2;!$(N~[!#6]);!$(N~[!#6]~[!#6])])[#6]',
    'fr_Ar_OH': '[OX2H][#6;R]',
    'fr_COO': '[CX3](=O)[OX2H1]',
    'fr_COO2': '[CX3](=O)[OX2H0,OX1]',
    'fr_C_O': '[CX3](=O)',
    'fr_C_O_noCOO': '[C;$(C(=O)[O,N,S]);!$(C=O)]',
    'fr_C_S': '[CX3](=S)',
    'fr_HOCCN': '[CH1](=[O,S])[CH2][NH2]',
    'fr_Imine': '[!#6][NX2]=!@[!#6]',
    'fr_NH0': '[NH0]',
    'fr_NH1': '[NH1]',
    'fr_NH2': '[NH2]',
    'fr_N_O': '[N,O]',
    'fr_Ndealkylation1': '[NX3;H2;!$(NC=O)]',
    'fr_Ndealkylation2': '[NX3;H0;!$(NC=O)][C;!$(C(=[O,N])[N,O])]',
    'fr_Nhpyrrole': '[nH][c,nX3](:[c,nX3])[c,nX3]',
    'fr_SH': '[#16][H]',
    'fr_aldehyde': '[CX3H1](=O)',
    'fr_alkyl_carbamate': '[NX3;H0][$([CX4]),$([CX4][CX4])](=O)[OX2H1]',
    'fr_alkyl_halide': '[Cl,Br,I][CX4]',
    'fr_allylic_oxid': '[CH2]=[CH][OX2H]',
    'fr_amide': '[NX3;H2][$([CX3](=[O])[#6]),$([CX3](~[OH])[#6])](=[O])[#6]',
    'fr_amidine': '[NX3H1,NX4+][#6]=[N+]=[N-]',
    'fr_aniline': '[#6][NH2]',
    'fr_aryl_methyl': '[#6][CH2][#6]',
    'fr_azide': '[NX2][NX2][NX1]',
    'fr_azo': '[#6]N=N[#6]',
    'fr_barbitur': '[#6]1(=O)NC(=O)NC1=O',
    'fr_benzene': '[cR1]',
    'fr_benzodiazepine': '[#7]1[#6]2[#6][#6][#6][#6][#6]1[#7]2',
    'fr_bicyclic': '[R2;r3,r4]',
    'fr_diazo': '[N]=[N+]=[N-]',
    'fr_dihydropyridine': '[#6][NH]1[#6][#6][#6][#6][#6]1',
    'fr_epoxide': '[O;R1][CX3](=[O])[CX3]',
    'fr_ester': '[CX3](=O)[OX2H0][#6]',
    'fr_ether': '[OD2]([#6])[#6]',
    'fr_furan': 'o1[cR1][cR1][cR1][cR1][cR1]1',
    'fr_guanido': '[#7;!R][N;!R][NH2]',
    'fr_halogen': '[F,Cl,Br,I]',
    'fr_hdrzine': '[CH1](=[O,S])[NH]([CH3])[CH3]',
    'fr_hdrzone': '[C;R0]=[C;R0][NH;R0][CH2;R0][NH;R0]',
    'fr_imidazole': '[nX3][cR1][nX3]',
    'fr_imide': '[CX3](=[O])[NX3]=[CX3](=[O])',
    'fr_isocyan': '[NX1]=[C;R0]=[O;R0]',
    'fr_isothiocyan': '[NX1]=[C;R0]=[S;R0]',
    'fr_ketone': '[#6][CX3](=O)[#6]',
    'fr_ketone_Topliss': '[#6][C;R0](=O)[#6]',
    'fr_lactam': '[#6][CX3](=[O,S])[NH1]',
    'fr_lactone': '[#6][CX3](=O)[OX2H1]',
    'fr_methoxy': '[O;X2H0][CX4]',
    'fr_morpholine': '[$([O;H1][CH2][CH2][N;H1]),$([O;H1][CH2][CH2][CH2][N;H1])]',
    'fr_nitrile': '[NX1]#[CX2]',
    'fr_nitro': '[NX3](=O)=O',
    'fr_nitro_arom': '[N+](=O)([O-])[#6]',
    'fr_nitro_arom_nonortho': '[N+](=O)([O-])[#6]([#6])[!#6]',
    'fr_nitroso': '[N;X2]=O',
    'fr_oxazole': '[#6][o,nX2]1[#6][#6][#6][#6][#6]1',
    'fr_oxime': '[CX3](=[OX1])[NX2][OH1]',
    'fr_para_hydroxylation': '[#6][CH2][OH]',
    'fr_phenol': '[OX2H][cR1]',
    'fr_phenol_noOrthoHbond': '[OH][cR1;!$([cR1]([OH])[OH])]',
    'fr_phos_acid': '[PH](=[OX1])(=[OX1])[OH1]',
    'fr_phos_ester': '[PH](=[OX1])([OX2H0])[OX2H1]',
    'fr_piperdine': '[NX3]1CCCCC1',
    'fr_piperzine': '[NH1;R0][CH2][CH2][NH1;R0]',
    'fr_priamide': '[NX3;H2][CX3](=[O])[#6]',
    'fr_prisulfonamd': '[N;H1;R0][S;R0](=[O;R0])(=[O;R0])[N;H1;R0]',
    'fr_pyridine': '[nX2]1cccc1',
    'fr_quatN': '[N+;H4]',
    'fr_sulfide': '[SX2][!#8]',
    'fr_sulfonamd': '[S;R0](=[O;R0])(=[O;R0])[NH;R0]',
    'fr_sulfone': '[S;R0](=[O;R0])(=[O;R0])',
    'fr_term_acetylene': '[CX2]#[CX2]',
    'fr_tetrazole': '[nX3]1nnnn1',
    'fr_thiazole': '[#6][sX2][cR1][cR1][#6]',
    'fr_thiocyan': '[SX1](=[CX2])[NX2][#6]',
    'fr_thiophene': '[sX2]1cccc1',
    'fr_unbrch_alkane': '[CX4;R0]',
    'fr_urea': '[NX3][CX3](=[O,S])[NX3]',
}


def get_descriptor_value(smiles, descriptor_names):
    info = {}
    bond_types = count_bond_types(smiles)
    for descriptor in descriptor_names:
        assert descriptor in keys, f"Descriptor {descriptor} not found in available descriptors."
        if descriptor in bond_types:
            info[descriptor] = bond_types[descriptor]
        else:
            info.update(calculate_descriptors(smiles, keys=[descriptor]))
    return info


def draw_with_descriptors(smiles, descriptor_name):
    mol = Chem.MolFromSmiles(smiles)

    if descriptor_name in ["SINGLE_BOND", "DOUBLE_BOND", "TRIPLE_BOND", "AROMATIC_BOND"]:
        highlighted_bonds = []
        for bond in mol.GetBonds():
            if descriptor_name == "SINGLE_BOND" and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                highlighted_bonds.append(bond.GetIdx())
            elif descriptor_name == "DOUBLE_BOND" and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                highlighted_bonds.append(bond.GetIdx())
            elif descriptor_name == "TRIPLE_BOND" and bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                highlighted_bonds.append(bond.GetIdx())
            elif descriptor_name == "AROMATIC_BOND" and bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                highlighted_bonds.append(bond.GetIdx())
    else:
        highlighted_bonds = []

    highlighted_image = Draw.MolToImage(mol,
                                        highlightBonds=highlighted_bonds,
                                        size=(224, 224))

    return highlighted_image


# Function to calculate descriptors for a single molecule
def calculate_descriptors(smiles, keys=None):
    molecule = Chem.MolFromSmiles(smiles)
    descriptors = {name: func(molecule) for name, func in Descriptors.descList if keys is None or name in keys}
    return descriptors


# # Function to calculate bond type for a single molecule
def count_bond_types(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    bond_types = {
        "SINGLE_BOND": 0,
        "DOUBLE_BOND": 0,
        "TRIPLE_BOND": 0,
        "AROMATIC_BOND": 0
    }

    for bond in molecule.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            bond_types["SINGLE_BOND"] += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            bond_types["DOUBLE_BOND"] += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            bond_types["TRIPLE_BOND"] += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            bond_types["AROMATIC_BOND"] += 1

    return bond_types