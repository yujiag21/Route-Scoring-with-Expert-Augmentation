import torch
import numpy as np
import pandas as pd
from fingerprint import DrfpEncoder
from aizynthfinder.chem.mol import UniqueMolecule
from sklearn.model_selection import train_test_split
from datetime import datetime
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
)

def get_feasibility_values(curr_reaction, reaction_class, i):
    try:
        r_class = str(curr_reaction['metadata']['classification']).split(' ')[0]
        r_class = min(r_class.split(';'))
        feasibility = reaction_class.loc[reaction_class['reaction_class']==r_class]['rank_score'].iloc[0]
    except:
#         print('Cannot process reaction class. Class: ' + str(curr_reaction['metadata']['classification']))
        r_class =  '0.0.0'
        feasibility = 5
        
    r_class=list(map(int, r_class.split('.')))
    if len(r_class)==2:
        r_class.append(0)

    return r_class, feasibility
    
def get_embedding(curr_reaction, mode='sdf', generator=None):
    if mode == 'sdf':
        # Get the smiles of the tree. For some reason, the smiles in ground truth route is stored in teh metadata instead of smiles
        smiles = curr_reaction['metadata']['mapped_reaction_smiles']

        # Get reactants and products
        reactants, products = smiles.split('>>')

        reactants_fp = sum([ UniqueMolecule(None, reactant).fingerprint(2,64) for reactant in reactants.split('.') ])
        products_fp = sum([ UniqueMolecule(None, product).fingerprint(2,64) for product in products.split('.') ])

        return products_fp - reactants_fp
    
    elif mode == 'drfp':
        smiles = curr_reaction['metadata']['mapped_reaction_smiles']
        drfp = np.array(DrfpEncoder.encode(smiles)).astype(np.int64)
        return np.squeeze(drfp)
    
    elif mode == 'rxnfp':
        if(generator is None):
            print('The generator must be given. Default to tree concatenate instead')
            return get_embedding(curr_reaction)
        
        smiles = curr_reaction['metadata']['mapped_reaction_smiles']    
        return np.squeeze(np.array(generator.convert(smiles)))
    
    else:
        print('Unknown embedding method. Default to SDF instead')
        return get_embedding(curr_reaction)
    
    
def process_tree(tree, reaction_class, target, mode='sdf', generator=None):
    i = 0
    trees = [tree]
    f = []
    
    feasibility_input = []
    
    while len(trees) != 0:
        curr_reaction = trees.pop()
        if('reaction' in curr_reaction['type']):
            
            # Get reaction_class 
            r_class, feasibility = get_feasibility_values(curr_reaction, reaction_class, i)
            f.append(feasibility)
            
            # Get embedding
            embed = []
            if(mode != 'none'):
                embed = get_embedding(curr_reaction, mode, generator)

            for achild in curr_reaction['children']:
                trees.append(achild)
            i += 1
            
            # Assemble the feasibility input
            fi = np.concatenate((r_class, [feasibility], target, embed))
            feasibility_input.append(fi)
            
        else:
            if 'children' in curr_reaction:
                [ trees.append(child) for child in curr_reaction['children'] ]
    
    ws = sum([1.1 ** (n + 1) for n in range(i)])   # Normalizing term
    wf = sum([(1.1**(j+1) * w) / ws for w, j in zip(f, range(i))])
    
    
    return feasibility_input, wf
    

def data_process(distances_dict, mode='sdf'):   # Fix: remove mode='sdf'
    start = datetime.now()
    df = pd.DataFrame()

    # Loop through the dictionary and append each entry to the DataFrame
    for key, value_list in distances_dict.items():
        for entry in value_list:
            df = df.append({
                'key': key,
                'values': entry
            }, ignore_index=True)

    # Rename columns
    df.rename(columns={'key': 'SMILES', 'values': 'Data'}, inplace=True)
    
    # Convert the 'Data' column to separate columns
    df = pd.concat([df.drop(['Data'], axis=1), df['Data'].apply(pd.Series)], axis=1)
    # df = df.drop(df.columns[-1], axis=1)
    df.columns=(['SMILES', 'cost', 'price', 'stability', 'feasibility', 'reaction_list', 'distance', 'reactants'])
    
    reaction_class = pd.read_csv('reaction_class_summ_20.csv')
    reaction_feasibility=[]
    
    rxnfp_generator = None
    if(mode == 'rxnfp'):
        model, tokenizer = get_default_model_and_tokenizer()
        rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
            
            
    target_mols=[]
    for m in df['SMILES']:
        target_mols.append(UniqueMolecule(None,m).fingerprint(2,64))
    
    feasibility_input = []
    feasibility = []
    
    for i in range(len(df)):
        fi, wf = process_tree(df['reactants'][i], reaction_class, target_mols[i], mode, rxnfp_generator)
        feasibility_input.append(fi)
        feasibility.append(wf)
    
    df['fingerprint']=target_mols
    df['feasibility_input']=feasibility_input
    df['feasibility'] = feasibility

    inputs = [ [c]+[p]+[s]+[fe] for c,p,s,fe in zip(df['cost'] ,df['price'] ,df['stability'],df['feasibility']) ]
    df['inputs']=inputs

    rows_with_empty_items = df.apply(lambda row: any(isinstance(item, list) and len(item) == 0 for item in row), axis=1)

    # rows_with_empty_items
    # Remove rows with empty items
    df=df[~rows_with_empty_items]
    
    end = datetime.now()
    total_time = str(end - start).split('.')[0]
    print('The preprocess for the data end at ' + str(end).split('.')[0])
    print('The preprocess took ' + total_time)
    
    return df


def input_process(df_, device='cpu'):
    df = df_.copy()

    X = np.array(df[['inputs', 'feasibility_input']])
    y = df['distance'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_list_inputs=X_train[:,1]
    X_train=X_train[:,0]
    # X_train=np.delete(X_train, 0, axis=1)

    X_test_list_inputs=X_test[:,1]
    X_test=X_test[:,0]

    X_train_tensor = torch.stack([torch.tensor(lst) for lst in X_train]).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.stack([torch.tensor(lst) for lst in X_test]).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train_tensor.shape[1]

    # Convert inputs to tensors
    # X_train_tensor = torch.stack([torch.tensor(lst) for lst in X_train]).to(device)
    # X_test_tensor = torch.stack([torch.tensor(lst) for lst in X_test]).to(device)
    #
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    #
    # # Calculate lengths of sequences
    # X_train_lengths = torch.tensor([len(seq) for seq in X_train_tensor], dtype=torch.long).to(device)
    # X_test_lengths = torch.tensor([len(seq) for seq in X_test_tensor], dtype=torch.long).to(device)
    #
    # return X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train_lengths, X_test_lengths
