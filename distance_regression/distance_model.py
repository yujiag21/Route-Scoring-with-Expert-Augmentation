import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from sklearn.model_selection import train_test_split
import torch.optim as optim
from distance_regression.data_processing import data_process, get_embedding, input_process
import sys
import warnings
import os
import errno
from signal import signal, SIGPIPE, SIG_DFL
import seaborn as sns
from itertools import product
import joblib
import logging
from collections.abc import Iterable
logger = logging.getLogger(__name__)

# Define the DeepSet encoder
class DeepSetEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_encoder, max_encoder, dropout_rate=0.0):
        super(DeepSetEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, max_encoder)
        self.fc2 = nn.Linear(max_encoder, max_encoder)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fcs = []
        for i in range(n_encoder):
            self.fcs.append(nn.Linear(max_encoder, max_encoder))
            self.fcs.append(nn.ReLU())
        self.fcs = nn.Sequential(*self.fcs)
        self.fc3 = nn.Linear(max_encoder, max_encoder)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(max_encoder, encoding_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        
        x = self.fcs(x)
        x = torch.relu(self.fc3(x))

        x = self.dropout2(x)
        x = self.fc4(x)
        encoded = torch.sum(x, dim=0)  # Aggregating the set using sum
        return encoded

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_main, max_main, dropout_rate=0.0):
        super(NeuralNetwork, self).__init__()
        half_max = int(max_main / 2)
        self.fc1 = nn.Linear(input_size, max_main)
        self.fc2 = nn.Linear(max_main, max_main)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fcs = []
        for i in range(n_main):
            self.fcs.append(nn.Linear(max_main, max_main))
            self.fcs.append(nn.ReLU())
        self.fcs = nn.Sequential(*self.fcs)
        self.fc3 = nn.Linear(max_main, max_main)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(max_main, 1)


    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        
        x = self.fcs(x)
        x = torch.relu(self.fc3(x))
        
        x = self.dropout2(x)
        x = self.fc4(x)
        # x = self.fcs(x)
        return x

def mytest(encoder, main_network, df_test,device, criterion, output_dir, epoch, save=False, mode=None):    
    X_test_mol_list_tensor = np.array(df_test['feasibility_input'])
    y_test_mol_tensor = torch.tensor(df_test['distance'].values, dtype=torch.float32).to(device)
    X_test_mol_tensor = torch.stack([torch.tensor(lst) for lst in np.array(df_test['inputs'])]).to(device)
    
    # test
    encoder.eval()
    main_network.eval()
    test_loss = 0.0
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
                predictions_list.append(predictions.cpu())

                # Compute loss
                test_loss += criterion(torch.squeeze(predictions.sum(dim=-1)), label)

        average_test_loss = test_loss
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {average_test_loss:.4f}')
        print(f'Test Loss: {average_test_loss / len(y_test_mol_tensor):.4f}')
        logger.info(f'Test Loss: {average_test_loss / len(y_test_mol_tensor):.4f}')
#     wandb.log({"Test mol Loss": average_test_loss / len(y_test_mol_tensor)})
    print(criterion(y_test_mol_tensor.cpu(), torch.tensor(predictions_list).cpu()).item())

    plt.figure()
    s = sample(list(range(len(y_test_mol_tensor))), 500)
    df_pre_full = pd.DataFrame({'y_test': y_test_mol_tensor.cpu(), 'predict': torch.tensor(predictions_list).cpu()})
    df_pre = df_pre_full.iloc[s]
    df_pre = df_pre.sort_values(by=['y_test'])
    y_test = np.array(df_pre['y_test'])
    predict = np.array(df_pre['predict'])

    plt.plot(predict, label='y_prediction')
    plt.plot(y_test, label='y_test')

    plt.xlabel('sample')
    plt.ylabel('distance')
    plt.legend()
#     wandb.log({"test_results_plot_500": plt})
    if save:
        plt.savefig(os.path.join(output_dir, f'test_results_plot_500_{mode}_{epoch}.png'))
    if (epoch % 10 == 0):
        joblib.dump(predict, os.path.join(output_dir, f'predict_value_{mode}_{epoch}.joblib'))
        joblib.dump(y_test, os.path.join(output_dir, f'true_value_{mode}_{epoch}.joblib'))
        joblib.dump(np.array(df_pre_full['y_test']), os.path.join(output_dir, f'full_true_list_{mode}_{epoch}.joblib'))
        joblib.dump(np.array(df_pre_full['predict']), os.path.join(output_dir, f'full_pred_list_{mode}_{epoch}.joblib'))
    plt.show()
    plt.close()  # Close the figure to prevent memory leaks

    # ranking test
    encoder.eval()
    main_network.eval()
    test_loss = 0.0
    average_rank_loss_list = []
    top_k = []
    f = []
    top_1=0
    with torch.no_grad():
        test_smiles = df_test['SMILES'].sample(n=100, random_state=2001).unique()  # meta test
        test_df = df_test[df_test['SMILES'].isin(test_smiles)]
        for m, l in list(test_df.groupby(['SMILES'])):
            sorted_indices = np.argsort(l['distance'])
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(l['distance']))
            predictions_l = []
            #         for r in l:
            list_inputs_list = l['feasibility_input']
            #             inputs = r['inputs']
#             inputs_tensor = torch.stack([torch.tensor(lst) for lst in np.array(l['inputs'])]).to(device)
            inputs_tensor = torch.stack([torch.tensor(lst) for lst in np.array(l['inputs'])]).to(device)
            for i, (inputs, list_inputs) in enumerate(zip(inputs_tensor, list_inputs_list), 0):
                if len(inputs) > 0 and len(list_inputs) > 0:
                    # Forward pass
                    #             print(list_inputs_tensor.shape)

                    list_inputs_tensor = torch.tensor(list_inputs, dtype=torch.float32).to(device)
                    encoded = encoder(list_inputs_tensor)
                    x = torch.relu(encoder.fc1(list_inputs_tensor))
                    x = encoder.fc2(x)
                    f.append([[xxx.item() for xxx in xx] for xx in x])
                    predictions = main_network(encoded.float(), inputs.float())

                    predictions_l.append(predictions.cpu().item())

            sorted_indices = np.argsort(predictions_l)

            ranks_pre = np.empty_like(sorted_indices)
            ranks_pre[sorted_indices] = np.arange(len(predictions_l))
            top_k.append({r: rp for r, rp in zip(ranks, ranks_pre)}[0])
            rank_test_loss = criterion(torch.tensor(ranks, dtype=torch.float32),
                                  torch.tensor(ranks_pre, dtype=torch.float32)).item()
            average_rank_test_loss = rank_test_loss / len(ranks_pre)
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {average_test_loss:.4f}')
            # print(f'Test Loss: {average_rank_test_loss:.4f}')
            #         print(rankings,rankings_pre)
            average_rank_loss_list.append(average_rank_test_loss)
            if ranks_pre[0]==0 or ranks_pre[1]==0:
                top_1 = top_1+1

        logger.info(f'average_rank_test_loss:{np.mean(average_rank_loss_list)}')
        logger.info(f'average_top-1_acc:{top_1/100}')
#         wandb.log({"average_rank_loss": np.mean(average_rank_loss_list)})


    sns.histplot(top_k, bins=20, kde=False)
    plt.title("Obsersed route ranks in top-K recommendations")
    plt.xlabel("K")
    plt.ylabel("Frequency")
    # wandb.log({"top-k": plt})
    if save:
        plt.savefig(os.path.join(output_dir, f'top-k_{mode}_{epoch}.png'))
        joblib.dump(top_k, os.path.join(output_dir, f'ranking_{mode}_{epoch}.joblib'))
        joblib.dump(average_rank_loss_list, os.path.join(output_dir, f'rank_loss_{mode}_{epoch}.joblib'))
    plt.show()
    plt.close()  # Close the figure to prevent memory leaks
    return

def mytrain(df_train, df_test, mode, lr, num_epochs, batch_size, encoding_size, n_main, n_encoder, max_main, max_encoder, dropout_rate, device, output_dir, seed=0):
    assert(isinstance(mode, str), 'The mode must be a string')
    
    test_losses = []
    running_losses = []
    
    # Padding to use batch
    # Group by the length of the reactions and use that as the inputs
    # Can try write a dedicated Dataset class
    
    
    X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, other_feature = input_process(df_train, device)
        
    # Define loss function (you can choose an appropriate loss function based on your task)
    criterion = nn.MSELoss().to(device)

    input_size = len(X_train_list_inputs[0][0])
    other_feature_size = other_feature

    # Create instances of the encoder and main network
    encoder = DeepSetEncoder(input_size, encoding_size, n_encoder, max_encoder, dropout_rate).to(device).float()
    main_network = NeuralNetwork(encoding_size+other_feature_size, n_main, max_main, dropout_rate).to(device).float()

    print(encoder, main_network)
    logger.info(repr(encoder))
    logger.info(repr(main_network))
    # Define optimizer for both encoder and main network
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(main_network.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    start = 0 
    
    # Training loop
    test_loss_list=[]
    running_loss_list=[]


    for epoch in range(start, num_epochs):
        logger.info(f'Training epoch {epoch} starts')
        running_loss=0
        encoder.train()
        main_network.train()
        optimizer.zero_grad()
        for i, (inputs, list_inputs, label) in enumerate(zip(X_train_tensor,X_train_list_inputs,y_train_tensor), 0):
            if len(inputs)>0 and len(list_inputs)>0:
                # Forward pass
                list_inputs_tensor = torch.tensor(np.array(list_inputs), dtype=torch.float32).to(device)
                encoded = encoder(list_inputs_tensor)

                predictions = main_network(encoded.float(), inputs.float())

                # Compute loss
                loss = criterion(torch.squeeze(predictions.sum(dim=-1)), label)    # If there is problem, try changing this to dimension 0
                running_loss += loss.item()

                # Backpropagation and optimization
                loss.backward()
            if ((i+1) % batch_size)==0:
            # optimizer the net
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        running_loss_list.append(running_loss)
        encoder.eval()
        main_network.eval()
        test_loss = 0.0
        predictions_list=[]
        with torch.no_grad():
            for i, (inputs, list_inputs, label) in enumerate(zip(X_test_tensor,X_test_list_inputs,y_test_tensor), 0):
                if len(inputs)>0 and len(list_inputs)>0:
                    list_inputs_tensor = torch.tensor(np.array(list_inputs), dtype=torch.float32).to(device)
                    encoded = encoder(list_inputs_tensor)
                    predictions = main_network(encoded.float(), inputs.float())
                    predictions_list.append(predictions)
                    # Compute loss
                    test_loss += criterion(torch.squeeze(predictions), label)

            average_test_loss = test_loss.cpu()
            test_loss_list.append(average_test_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(X_train_tensor):.4f}, Test Loss: {average_test_loss/len(X_test_tensor):.4f}')
            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(X_train_tensor):.4f}, Test Loss: {average_test_loss / len(X_test_tensor):.4f}')

        if epoch % 10 == 0:
            checkpoint = f'{output_dir}/checkpoints'
            os.makedirs(checkpoint, exist_ok = True)
            save_path = f'{checkpoint}/checkpoints_{epoch}'
            os.makedirs(save_path, exist_ok = True)
            torch.save(encoder.state_dict(), f"{save_path}/encoder_{mode}.pt")
            torch.save(main_network.state_dict(), f"{save_path}/main_network_{mode}.pt")
            
            mytest(encoder=encoder, main_network=main_network, df_test=df_test.copy(), device=device, criterion=criterion, output_dir=save_path, epoch=epoch, save=True, mode=mode)


    test_losses.append(np.array(test_loss_list) / len(X_test_tensor))
    running_losses.append(np.array(running_loss_list) / len(X_train_tensor))
    print('Training and testing finished')

    # Save encoder and main network
    model_path = f'{output_dir}/final_models/'
    os.makedirs(model_path, exist_ok=True)
    torch.save(encoder.state_dict(), f"{model_path}/encoder_{mode}.pt")
    torch.save(main_network.state_dict(), f"{model_path}/main_network_{mode}.pt")
    print('Models saved')
        
    return test_losses, running_losses

