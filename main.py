import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distance_regression.data_processing import data_process, get_embedding, input_process
import sys
import warnings
import torch.nn as nn
import torch
import os
import shutil
import argparse
from rdkit import Chem
from aizynthfinder.utils.image import RouteImageFactory


class DeepSetEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_encoder, max_encoder, dropout_rate=0):
        super(DeepSetEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, encoding_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        encoded = torch.sum(x, dim=0)  # Aggregating the set using sum
        return encoded


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_main, max_main, dropout_rate=0):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


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


def read_data(file_name):
    os.path.dirname(os.path.abspath(__file__))
    file_list = []

    distances_dict = {}
    with open(file_name, "r") as fp:
        distance_dict = json.load(fp)
        distances_dict.update(distance_dict)
        mols = list(distances_dict.keys())

        for mol in mols:
            for j in range(len(distances_dict[mol])):
                file_list.append(distances_dict[mol][j][0])
                distances_dict[mol][j].pop(0)
                tree = distances_dict[mol][j][6]  # [Molecule name][The route index][The route]
                distances_dict[mol][j][6] = canonicalize_route(tree)

    # print(len(distances_dict))
    return distances_dict, file_list


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


def main(args):
    encoding_size = 256
    n_main = 2
    n_encoder = 2
    max_encoder =1024
    max_main = 1024
    mode = args.mode
    print(args.mode)
    # model_dir = f"results/{args.mode}/batch{batch_size}_encoding{encoding_size}_epochs{num_epochs}_{task_id}/final_models"   ## Fix: Change the seed number
    model_dir = f"model/"  ## Fix: Change the seed number

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # distances_dict = read_new_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read route dictionary and canonicalize
    distance_dict, mol_list = read_data(f'data/{args.input_file}')
    # add route embedding
    df_test = data_process(distance_dict, mode)

    df_test = add_indices(df_test).copy()

    X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, other_feature = input_process(
        df_test, device)
    input_size = len(X_train_list_inputs[0][0])
    # print(input_size)
    # input_size = 64
    other_feature_size = 4

    encoder = DeepSetEncoder(input_size, encoding_size, n_encoder, max_encoder).to(device).float()
    main_network = NeuralNetwork(encoding_size + other_feature_size, n_main, max_main).to(device).float()

    encoder.load_state_dict(torch.load(os.path.join(model_dir, f"encoder_{args.mode}_prediction.pt")))
    main_network.load_state_dict(torch.load(os.path.join(model_dir, f"main_network_{args.mode}_prediction.pt")))

    X_test_mol_list_tensor = np.array(df_test['feasibility_input'])
    y_test_mol_tensor = torch.tensor(df_test['distance'].values, dtype=torch.float32).to(device)
    X_test_mol_tensor = torch.stack([torch.tensor(lst) for lst in np.array(df_test['inputs'])]).to(device)

    predictions_list = []
    with torch.no_grad():
        for i, (inputs, list_inputs, label) in enumerate(
                zip(X_test_mol_tensor, X_test_mol_list_tensor, y_test_mol_tensor),
                0):
            if len(inputs) > 0 and len(list_inputs) > 0:
                # Forward pass
                list_inputs_tensor = torch.tensor(np.array(list_inputs), dtype=torch.float32).to(device)
                encoded = encoder(list_inputs_tensor)

                predictions = main_network(encoded.float(), inputs.float())
                predictions_list.append(predictions.cpu().item())

    df_test['predicted_distance'] = predictions_list
    df_output = df_test[
        ['molecule_index', 'image_index', 'SMILES', 'cost', 'stability', 'feasibility', 'reaction_list', 'distance',
         'predicted_distance', 'reactants']]
    df_output.rename({"feasibility_input": 'reaction_class'})

    output_dir = f"{args.output_path}/{args.mode}"
    if not os.path.exists(output_dir):
    #     shutil.rmtree(res_dir)
        os.makedirs(output_dir)
    save_picture(df_test, output_dir)
    df_output.to_csv(os.path.join(output_dir, 'prediction_test_route.csv'), index=False)
    df_output.to_excel(os.path.join(output_dir, 'prediction_test_route.xlsx'), index=False)
    print(f'The route scores are saved in {output_dir}')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding_size", type=int, default=256, help="Size of encoding")
    parser.add_argument("--output_path", type=str, default='route_score', help="Output save path")
    parser.add_argument("--input_file", type=str, default='route_10.json', help="Input file name")
    parser.add_argument("--mode", type=str, default='sdf', help="mode")
    parser.add_argument("--n_layers_encoder", type=int, default=2,
                        help='Number of layers with max_neuron_encoder neurons in the encoder')
    parser.add_argument("--n_layers_main", type=int, default=2,
                        help='Number of layers with max_neuron_main neurons in the main network')
    args = parser.parse_args()
    main(args)
