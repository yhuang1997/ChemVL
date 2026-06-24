from pathlib import Path
import numpy as np
import os
import pandas as pd
import io
from PIL import Image as PILImage
import torch
import pickle
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from omegaconf import DictConfig, OmegaConf
from ordinalclip.runner.runner import Runner, MultiTaskRunner


task2id_106 = {
    0: 'NumRadicalElectrons',
    1: 'NHOHCount',
    2: 'NOCount',
    3: 'NumAliphaticCarbocycles',
    4: 'NumAliphaticHeterocycles',
    5: 'NumAliphaticRings',
    6: 'NumAromaticCarbocycles',
    7: 'NumAromaticHeterocycles',
    8: 'NumAromaticRings',
    9: 'NumHAcceptors',
    10: 'NumHDonors',
    11: 'NumHeteroatoms',
    12: 'NumRotatableBonds',
    13: 'NumSaturatedCarbocycles',
    14: 'NumSaturatedHeterocycles',
    15: 'NumSaturatedRings',
    16: 'RingCount',
    17: 'fr_Al_COO',
    18: 'fr_Al_OH',
    19: 'fr_Al_OH_noTert',
    20: 'fr_ArN',
    21: 'fr_Ar_COO',
    22: 'fr_Ar_N',
    23: 'fr_Ar_NH',
    24: 'fr_Ar_OH',
    25: 'fr_COO',
    26: 'fr_COO2',
    27: 'fr_C_O',
    28: 'fr_C_O_noCOO',
    29: 'fr_C_S',
    30: 'fr_HOCCN',
    31: 'fr_Imine',
    32: 'fr_NH0',
    33: 'fr_NH1',
    34: 'fr_NH2',
    35: 'fr_N_O',
    36: 'fr_Ndealkylation1',
    37: 'fr_Ndealkylation2',
    38: 'fr_Nhpyrrole',
    39: 'fr_SH',
    40: 'fr_aldehyde',
    41: 'fr_alkyl_carbamate',
    42: 'fr_alkyl_halide',
    43: 'fr_allylic_oxid',
    44: 'fr_amide',
    45: 'fr_amidine',
    46: 'fr_aniline',
    47: 'fr_aryl_methyl',
    48: 'fr_azide',
    49: 'fr_azo',
    50: 'fr_barbitur',
    51: 'fr_benzene',
    52: 'fr_benzodiazepine',
    53: 'fr_bicyclic',
    54: 'fr_diazo',
    55: 'fr_dihydropyridine',
    56: 'fr_epoxide',
    57: 'fr_ester',
    58: 'fr_ether',
    59: 'fr_furan',
    60: 'fr_guanido',
    61: 'fr_halogen',
    62: 'fr_hdrzine',
    63: 'fr_hdrzone',
    64: 'fr_imidazole',
    65: 'fr_imide',
    66: 'fr_isocyan',
    67: 'fr_isothiocyan',
    68: 'fr_ketone',
    69: 'fr_ketone_Topliss',
    70: 'fr_lactam',
    71: 'fr_lactone',
    72: 'fr_methoxy',
    73: 'fr_morpholine',
    74: 'fr_nitrile',
    75: 'fr_nitro',
    76: 'fr_nitro_arom',
    77: 'fr_nitro_arom_nonortho',
    78: 'fr_nitroso',
    79: 'fr_oxazole',
    80: 'fr_oxime',
    81: 'fr_para_hydroxylation',
    82: 'fr_phenol',
    83: 'fr_phenol_noOrthoHbond',
    84: 'fr_phos_acid',
    85: 'fr_phos_ester',
    86: 'fr_piperdine',
    87: 'fr_piperzine',
    88: 'fr_priamide',
    89: 'fr_prisulfonamd',
    90: 'fr_pyridine',
    91: 'fr_quatN',
    92: 'fr_sulfide',
    93: 'fr_sulfonamd',
    94: 'fr_sulfone',
    95: 'fr_term_acetylene',
    96: 'fr_tetrazole',
    97: 'fr_thiazole',
    98: 'fr_thiocyan',
    99: 'fr_thiophene',
    100: 'fr_unbrch_alkane',
    101: 'fr_urea',
    102: 'SINGLE_BOND',
    103: 'DOUBLE_BOND',
    104: 'TRIPLE_BOND',
    105: 'AROMATIC_BOND'}


id2task_106 = {
    'NumRadicalElectrons': 0,
    'NHOHCount': 1,
    'NOCount': 2,
    'NumAliphaticCarbocycles': 3,
    'NumAliphaticHeterocycles': 4,
    'NumAliphaticRings': 5,
    'NumAromaticCarbocycles': 6,
    'NumAromaticHeterocycles': 7,
    'NumAromaticRings': 8,
    'NumHAcceptors': 9,
    'NumHDonors': 10,
    'NumHeteroatoms': 11,
    'NumRotatableBonds': 12,
    'NumSaturatedCarbocycles': 13,
    'NumSaturatedHeterocycles': 14,
    'NumSaturatedRings': 15,
    'RingCount': 16,
    'fr_Al_COO': 17,
    'fr_Al_OH': 18,
    'fr_Al_OH_noTert': 19,
    'fr_ArN': 20,
    'fr_Ar_COO': 21,
    'fr_Ar_N': 22,
    'fr_Ar_NH': 23,
    'fr_Ar_OH': 24,
    'fr_COO': 25,
    'fr_COO2': 26,
    'fr_C_O': 27,
    'fr_C_O_noCOO': 28,
    'fr_C_S': 29,
    'fr_HOCCN': 30,
    'fr_Imine': 31,
    'fr_NH0': 32,
    'fr_NH1': 33,
    'fr_NH2': 34,
    'fr_N_O': 35,
    'fr_Ndealkylation1': 36,
    'fr_Ndealkylation2': 37,
    'fr_Nhpyrrole': 38,
    'fr_SH': 39,
    'fr_aldehyde': 40,
    'fr_alkyl_carbamate': 41,
    'fr_alkyl_halide': 42,
    'fr_allylic_oxid': 43,
    'fr_amide': 44,
    'fr_amidine': 45,
    'fr_aniline': 46,
    'fr_aryl_methyl': 47,
    'fr_azide': 48,
    'fr_azo': 49,
    'fr_barbitur': 50,
    'fr_benzene': 51,
    'fr_benzodiazepine': 52,
    'fr_bicyclic': 53,
    'fr_diazo': 54,
    'fr_dihydropyridine': 55,
    'fr_epoxide': 56,
    'fr_ester': 57,
    'fr_ether': 58,
    'fr_furan': 59,
    'fr_guanido': 60,
    'fr_halogen': 61,
    'fr_hdrzine': 62,
    'fr_hdrzone': 63,
    'fr_imidazole': 64,
    'fr_imide': 65,
    'fr_isocyan': 66,
    'fr_isothiocyan': 67,
    'fr_ketone': 68,
    'fr_ketone_Topliss': 69,
    'fr_lactam': 70,
    'fr_lactone': 71,
    'fr_methoxy': 72,
    'fr_morpholine': 73,
    'fr_nitrile': 74,
    'fr_nitro': 75,
    'fr_nitro_arom': 76,
    'fr_nitro_arom_nonortho': 77,
    'fr_nitroso': 78,
    'fr_oxazole': 79,
    'fr_oxime': 80,
    'fr_para_hydroxylation': 81,
    'fr_phenol': 82,
    'fr_phenol_noOrthoHbond': 83,
    'fr_phos_acid': 84,
    'fr_phos_ester': 85,
    'fr_piperdine': 86,
    'fr_piperzine': 87,
    'fr_priamide': 88,
    'fr_prisulfonamd': 89,
    'fr_pyridine': 90,
    'fr_quatN': 91,
    'fr_sulfide': 92,
    'fr_sulfonamd': 93,
    'fr_sulfone': 94,
    'fr_term_acetylene': 95,
    'fr_tetrazole': 96,
    'fr_thiazole': 97,
    'fr_thiocyan': 98,
    'fr_thiophene': 99,
    'fr_unbrch_alkane': 100,
    'fr_urea': 101,
    'SINGLE_BOND': 102,
    'DOUBLE_BOND': 103,
    'TRIPLE_BOND': 104,
    'AROMATIC_BOND': 105}


task2id_7 = {
    0: 'HeavyAtomCount',
    1: 'NumAromaticRings',
    2: 'fr_halogen',
    3: 'SINGLE_BOND',
    4: 'DOUBLE_BOND',
    5: 'TRIPLE_BOND',
    6: 'AROMATIC_BOND',
}

id2task_7 = {
    'HeavyAtomCount': 0,
    'NumAromaticRings': 1,
    'fr_halogen': 2,
    'SINGLE_BOND': 3,
    'DOUBLE_BOND': 4,
    'TRIPLE_BOND': 5,
    'AROMATIC_BOND': 6,
}


def parse_cfg(args):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    cfg = OmegaConf.merge(cfg, OmegaConf.create())
    from utils.path_utils import expand_data_root_strings

    cfg = OmegaConf.create(expand_data_root_strings(OmegaConf.to_container(cfg, resolve=True)))
    # Add Checkpoint
    cfg.ckpt = args.ckpt
    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    # Set test_only & debug & verbose
    args.test_only = True
    args.debug = False
    args.verbose = False

    seed = cfg.runner_cfg.seed
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def load_checkpoint(cfg: DictConfig):
    if cfg.multitask:
        runner = MultiTaskRunner(**OmegaConf.to_container(cfg.runner_cfg))
    else:
        runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
    return runner.load_from_checkpoint(cfg.ckpt, **OmegaConf.to_container(cfg.runner_cfg))


def _ensure_pil(img):
    # In Notebook, RDKit may return IPython.display.Image (with .data bytes)
    if hasattr(img, "data") and isinstance(img.data, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(img.data)).convert("RGB")

    # In normal cases, RDKit returns PIL.Image.Image
    if hasattr(img, "convert"):
        return img.convert("RGB")

    raise TypeError(f"Unexpected type {type(img)}")


def get_image_and_transform(smiles):
    # This is the default transformation for CLIP
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    from utils.pretrain_image_render import smiles_to_pretrain_pil

    images = [smiles_to_pretrain_pil(smi, canvas_px=224) for smi in smiles]
    input_image_tensor = torch.stack([transform(image) for image in images], dim=0)
    return input_image_tensor, images


def get_descriptor_values(smiles, keys=None, cache_dict=None, update_cache=False, cache_path=None):
    descriptors_list = []

    if cache_dict is None:
        cache_dict = {}
    if keys is None:
        keys = [name for name, _ in Descriptors.descList]
    for smi in smiles:
        ret = cache_dict.get(smi, None)
        if ret is None:
            mol = Chem.MolFromSmiles(smi)
            descriptors = {name: func(mol) for name, func in Descriptors.descList if name in keys}
            bonds = count_bond_types(smi)
            descriptors.update(bonds)
            descriptors_list.append(descriptors)
            if update_cache:
                cache_dict[smi] = descriptors
        else:
            try:
                ret = {name: ret[name] for name in keys}
            except KeyError:
                mol = Chem.MolFromSmiles(smi)
                ret = {name: func(mol) for name, func in Descriptors.descList if
                       keys is None or name in keys}
                bonds = count_bond_types(smi)
                ret.update(bonds)
                if update_cache:
                    cache_dict[smi] = ret
            descriptors_list.append(ret)
    if update_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_dict, f)
    return descriptors_list


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


def inference(runner, smiles, batch_size=32, keys=None, labels=None, save_original_images=True):
    runner.cuda()
    runner.eval()

    task_ids = []  # num_tasks
    for key in keys:
        task_ids.append(id2task_106[key])

    # get the descriptor values
    if labels is None:
        labels = get_descriptor_values(smiles, keys=keys)  # num_samples * dict_info
    gts = []  # num_tasks, num_samples
    for key in keys:
        gts.append([label[key] for label in labels])
    gts = torch.tensor(gts).float().t()  # num_samples, num_tasks

    num_samples = len(smiles)
    logits = []  # num_tasks, num_samples, num_classes
    for _ in task_ids:
        logits.append([])
    if save_original_images:
        non_transformed_images_list = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_images, batch_non_transformed_images = get_image_and_transform(smiles[i:i + batch_size])
            batch_images = batch_images.cuda()
            # batch_gts = gts[i:i + batch_size].cuda()
            batch_gts = None  # we don't need ground truth when doing inference
            # batch_gts = torch.tensor([[1, 0, 2, 3] for _ in range(106)]).cuda()
            batch_data = (batch_images, batch_gts)
            # _ = runner.training_step(batch_data, batch_idx=None)
            _, batch_logits, _ = runner.predict_step(batch_data, batch_idx=None, task_id=task_ids)
            for j in range(len(task_ids)):
                normed_batch_logits = torch.nn.functional.softmax(batch_logits[j], dim=1).cpu().numpy()
                logits[j].extend(normed_batch_logits)
            if save_original_images:
                non_transformed_images_list.extend(batch_non_transformed_images)

    # num_samples first
    logits = [[logit[i] for logit in logits] for i in range(num_samples)]

    if save_original_images:
        non_transformed_images = np.stack(non_transformed_images_list)
    else:
        non_transformed_images = None

    return non_transformed_images, logits, gts.cpu().numpy().astype(int)


def save_as_csv(molecule_names, smiles, logits_per_descriptor, md_gts, mds, output_dir=None):
    if output_dir is None:
        output_dir = '.'

    df = pd.DataFrame({
        'molecule_name': molecule_names,
        'smiles': smiles
    })

    for i, md in enumerate(mds):
        for j in range(len(logits_per_descriptor[i])):
            pred = np.argmax(logits_per_descriptor[i][j])
            logit = logits_per_descriptor[i][j][pred]
            gt_logit = logits_per_descriptor[i][j][md_gts[i][j]]
            df.loc[j, f'{md}_logit'] = logit
            df.loc[j, f'{md}_gt_logit'] = gt_logit
            df.loc[j, f'{md}_pred'] = pred
            df.loc[j, f'{md}_gt'] = md_gts[i][j]

    df.to_csv(os.path.join(output_dir, f'results.csv'), index=False)