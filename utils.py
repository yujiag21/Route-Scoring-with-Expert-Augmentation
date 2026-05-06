

import torch.nn as nn
from aizynthfinder.utils.image import RouteImageFactory
import os
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    return smiles


def canonicalize_reactions(rxn_smiles):
    reactants_smiles, products_smiles = rxn_smiles.split('>>')
    # Process reactants and products separately
    canonical_reactants = [canonicalize_smiles(sm) for sm in reactants_smiles.split('.')]
    canonical_products = [canonicalize_smiles(sm) for sm in products_smiles.split('.')]

    canonical_reactants = '.'.join(sorted(canonical_reactants))
    canonical_products = '.'.join(sorted(canonical_products))

    # Combine back into a canonical reaction SMILES
    return canonical_reactants + '>>' + canonical_products


def canonicalize_route(tree):
    trees = [tree]

    while len(trees) != 0:
        curr_reaction = trees.pop()
        if ('reaction' in curr_reaction['type']):

            # Get reaction_class
            try:
                curr_reaction['metadata']['mapped_reaction_smiles'] = canonicalize_reactions(
                    curr_reaction['metadata']['mapped_reaction_smiles'])
            except:
                curr_reaction['metadata']['mapped_reaction_smiles'] = canonicalize_reactions(
                    curr_reaction['metadata']['mapped_smiles'])

            for achild in curr_reaction['children']:
                trees.append(achild)

        else:
            if 'children' in curr_reaction:
                [trees.append(child) for child in curr_reaction['children']]

    return tree

def save_picture(df_test, res_dir):
    for index, row in df_test.iterrows():
        molecule_path = os.path.join(res_dir, f"Molecule_{row['molecule_index']}-{row['SMILES']}")
        if not (os.path.exists(molecule_path)):
            os.makedirs(molecule_path)

        image = RouteImageFactory(row['reactants']).image
        image.save(os.path.join(molecule_path, f"{row['image_index']}.png"))

    return


def add_indices(df_test_):
    df_test = df_test_.copy()

    image_index = 0
    molecule_index = 1
    molecule_indices = []
    image_indices = []
    prev_smiles = df_test['SMILES'][0]

    for index, row in df_test.iterrows():
        if row['SMILES'] == prev_smiles:
            image_index += 1
        else:
            molecule_index += 1
            image_index = 1
        image_indices.append(image_index)
        molecule_indices.append(molecule_index)
        prev_smiles = row['SMILES']

    df_test['molecule_index'] = molecule_indices
    df_test['image_index'] = image_indices
    return df_test