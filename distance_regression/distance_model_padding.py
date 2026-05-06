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
import torch.nn as nn
import torch
import os
import errno
from signal import signal, SIGPIPE, SIG_DFL
import seaborn as sns
from itertools import product
import joblib
import logging
import torch.nn.utils.rnn as rnn_utils
from collections.abc import Iterable
logger = logging.getLogger(__name__)

# Define the DeepSet encoder
# class DeepSetEncoder(nn.Module):
#     def __init__(self, input_size, encoding_size):
#         super(DeepSetEncoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, encoding_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         encoded = torch.sum(x, dim=0)  # Aggregating the set using sum
#         return encoded
#
# # Define the neural network model
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 1)
#
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=0)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class DeepSetEncoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(DeepSetEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, encoding_size)
    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x = packed_input.data
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        # encoded = torch.sum(x, dim=1)  # Aggregating the set using sum
        # Repackage the processed data
        packed_output = rnn_utils.PackedSequence(x, packed_input.batch_sizes, packed_input.sorted_indices,
                                                 packed_input.unsorted_indices)

        # Unpack the sequence
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # 使用掩码对有效的时间步进行聚合
        batch_size, max_seq_len, encoding_size = output.size()
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).to(output.device)  # (batch_size, max_seq_len, 1)
        output = output * mask  # 只保留有效部分，填充部分将被设为0

        # 对有效的时间步聚合
        encoded = torch.sum(output, dim=1)
        return encoded
# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def mytest(encoder, main_network, df_test,device, criterion, output_dir, epoch, save=False, mode=None):
    test_mol = np.array(df_test['SMILES'])
    X_test_mol_list_tensor = np.array(df_test['feasibility_input'])
    y_test_mol_tensor = torch.tensor(df_test['distance'].values, dtype=torch.float32).to(device)
    X_test_mol_tensor = torch.stack([torch.tensor(lst) for lst in np.array(df_test['inputs'])]).to(device)
    X_test_lengths = torch.tensor([len(seq) for seq in X_test_mol_list_tensor], dtype=torch.long).to(device)

    X_test_list_inputs_padded = [rnn_utils.pad_sequence([torch.tensor(item, dtype=torch.float32) for item in lst],
                                                        batch_first=True) for lst in X_test_mol_list_tensor]

    # test
    encoder.eval()
    main_network.eval()
    test_loss = 0.0
    predictions_list = []
    test_mol_list = []
    distance_list = []
    batch_size = 256

    with torch.no_grad():
        permutation = list(range(len(X_test_mol_tensor)))

        for i in range(0, len(X_test_mol_tensor), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = [X_test_mol_tensor[j] for j in indices]
            batch_lengths = X_test_lengths[indices]
            batch_list_inputs = [X_test_list_inputs_padded[j] for j in indices]
            batch_y = y_test_mol_tensor[indices]
            batch_mol = test_mol[indices]

            # Sort by lengths (required by pack_padded_sequence)
            batch_lengths, sort_idx = batch_lengths.sort(0, descending=True)
            batch_X = [batch_X[j] for j in sort_idx]
            batch_list_inputs = [batch_list_inputs[j] for j in sort_idx]
            batch_y = batch_y[sort_idx]
            batch_mol = [batch_mol[j] for j in sort_idx]

            batch_list_inputs_tensor = rnn_utils.pad_sequence(batch_list_inputs, batch_first=True).to(device)

            # Convert list inputs to tensors and pack them
            batch_lengths = batch_lengths.cpu().to(torch.int64)
            encoded = encoder(batch_list_inputs_tensor.data.float(), batch_lengths)
            # Forward pass with the original main network input
            batch_X_tensor = torch.stack(batch_X).to(device)
            predictions = main_network(encoded.float(), batch_X_tensor)

            predictions_list.extend(predictions.cpu())
            test_mol_list.extend(batch_mol)
            distance_list.extend(batch_y.cpu())

            # Compute loss
            loss = criterion(predictions.squeeze(dim=-1), batch_y)
            test_loss += loss.item()

        #     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {average_test_loss:.4f}')
    print(f'Test Loss: {test_loss / len(y_test_mol_tensor):.4f}')
    logger.info(f'Test Loss: {test_loss / len(y_test_mol_tensor):.4f}')
    # wandb.log({"Test mol Loss": average_test_loss / len(y_test_mol_tensor)})
#     print(criterion(distance_list.cpu(), torch.tensor(predictions_list).cpu()).item())

    plt.figure()
    s = sample(list(range(len(distance_list))), 500)
    df_pre_full = pd.DataFrame({'y_test': distance_list, 'predict': predictions_list})
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
    if (epoch % 60 == 0):
        joblib.dump(predict, os.path.join(output_dir, f'predict_value_{mode}_{epoch}.joblib'))
        joblib.dump(y_test, os.path.join(output_dir, f'true_value_{mode}_{epoch}.joblib'))
        joblib.dump(np.array(df_pre_full['y_test']), os.path.join(output_dir, f'full_true_list_{mode}_{epoch}.joblib'))
        joblib.dump(np.array(df_pre_full['predict']), os.path.join(output_dir, f'full_pred_list_{mode}_{epoch}.joblib'))
    plt.show()
    plt.close()  # Close the figure to prevent memory leaks

    # ranking acc
    top_1 = 0
    top_k = []
    average_rank_loss_list =[]
    df_prediction = pd.DataFrame({'SMILES': test_mol_list, 'prediction_distance': predictions_list,'distance':distance_list})
    groupby_list = list(df_prediction.groupby(['SMILES']))
    number_mol = len(groupby_list)
    for m, df in groupby_list:
        df['rank'] = df['distance'].rank(method='min').astype(int)
        df['rank_pre'] = df['prediction_distance'].rank(method='min').astype(int)
        # sorted_indices = np.argsort(l['distance'])
        # ranks = np.empty_like(sorted_indices)
        # ranks[sorted_indices] = np.arange(len(l['distance']))
        # sorted_indices = np.argsort(l['prediction_distance'])
        #
        # ranks_pre = np.empty_like(sorted_indices)
        # ranks_pre[sorted_indices] = np.arange(len(l['prediction_distance']))
        # top_k.append({r: rp for r, rp in zip(ranks, ranks_pre)}[0])
        top_k.append(df[df['rank']==1]['rank_pre'].iloc[0])
        rank_test_loss = criterion(torch.tensor(np.array(df['rank']), dtype=torch.float32),
                                   torch.tensor(np.array(df['rank_pre']), dtype=torch.float32)).item()
        average_rank_test_loss = rank_test_loss / len(df)
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {average_test_loss:.4f}')
        # print(f'Test Loss: {average_rank_test_loss:.4f}')
        #         print(rankings,rankings_pre)
        average_rank_loss_list.append(average_rank_test_loss)
        if 1 in list(df[df['rank']==1]['rank_pre']):
            top_1 = top_1 + 1

    logger.info(f'average_rank_test_loss:{np.mean(average_rank_loss_list)}')
    logger.info(f'average_top-1_acc of {number_mol} molecules:{top_1/number_mol}')

    sns.histplot(top_k, bins=20, kde=False)
    plt.title("Obsersed route ranks in top-K recommendations")
    plt.xlabel("K")
    plt.ylabel("Frequency")
    # wandb.log({"top-k": plt})
    if save:
        plt.savefig(os.path.join(output_dir, f'top-k_{mode}_{epoch}.png'))
    # if (epoch % 20 == 0) or (epoch % 70 == 0):
        joblib.dump(top_k, os.path.join(output_dir, f'ranking_{mode}_{epoch}.joblib'))
        joblib.dump(average_rank_loss_list, os.path.join(output_dir, f'rank_loss_{mode}_{epoch}.joblib'))
    plt.show()
    plt.close()

    return

def mytrain(df_train, df_test, mode, lr, num_epochs, batch_size, encoding_size, device, output_dir, seed=0):
    assert(isinstance(mode, str), 'The mode must be a string')
    
    test_losses = []
    running_losses = []
    
    # Padding to use batch
    # Group by the length of the reactions and use that as the inputs
    # Can try write a dedicated Dataset class

    # X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train_lengths, X_test_lengths = input_process(df_train, device)
    X_train_list_inputs, X_test_list_inputs, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, other_feature = input_process(df_train, device)
    # Calculate lengths of sequences
    X_train_lengths = torch.tensor([len(seq) for seq in X_train_list_inputs], dtype=torch.long).to(device)
    X_train_list_inputs_padded = [rnn_utils.pad_sequence([torch.tensor(item, dtype=torch.float32) for item in lst],
                                                         batch_first=True) for lst in X_train_list_inputs]

    # other_feature = X_train_tensor.shape[1]
    # Define loss function (you can choose an appropriate loss function based on your task)
    criterion = nn.MSELoss().to(device)
    input_size = len(X_train_list_inputs[0][0])
    other_feature_size = other_feature

    # Create instances of the encoder and main network
    encoder = DeepSetEncoder(input_size, encoding_size).to(device).float()
    main_network = NeuralNetwork(encoding_size+other_feature_size).to(device).float()

    print(encoder, main_network)
    # Define optimizer for both encoder and main network
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(main_network.parameters()),
        lr=lr
    )

    # Training loop
    # test_loss_list=[]
    running_loss_list=[]
    for epoch in range(num_epochs):
        running_loss = 0.0
        permutation = torch.randperm(len(X_train_tensor))

        for i in range(0, len(X_train_tensor), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = [X_train_tensor[j] for j in indices]
            batch_lengths = X_train_lengths[indices]
            batch_list_inputs = [X_train_list_inputs_padded[j] for j in indices]
            batch_y = y_train_tensor[indices]

            # Sort by lengths (required by pack_padded_sequence)
            batch_lengths, sort_idx = batch_lengths.sort(0, descending=True)
            batch_X = [batch_X[j] for j in sort_idx]
            batch_list_inputs = [batch_list_inputs[j] for j in sort_idx]
            batch_y = batch_y[sort_idx]

            batch_list_inputs_tensor = rnn_utils.pad_sequence(batch_list_inputs, batch_first=True).to(device)

            # Convert list inputs to tensors and pack them
            batch_lengths = batch_lengths.cpu().to(torch.int64)

            encoded = encoder(batch_list_inputs_tensor.data.float(),batch_lengths)

            # Forward pass with the original main network input
            batch_X_tensor = torch.stack(batch_X).to(device)
            predictions = main_network(encoded.float(), batch_X_tensor)

            # Compute loss
            loss = criterion(predictions.squeeze(dim=-1), batch_y)
            running_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(X_train_tensor):.4f}")

        running_loss_list.append(running_loss)

        if epoch % 10 == 0:
            checkpoint = f'{output_dir}/checkpoints'
            os.makedirs(checkpoint, exist_ok = True)
            save_path = f'{checkpoint}/checkpoints_{epoch}'
            os.makedirs(save_path, exist_ok = True)
            torch.save(encoder.state_dict(), f"{save_path}/encoder_{mode}.pt")
            torch.save(main_network.state_dict(), f"{save_path}/main_network_{mode}.pt")
            
            mytest(encoder=encoder, main_network=main_network, df_test=df_test.copy(), device=device, criterion=criterion, output_dir=save_path, epoch=epoch, save=True, mode=mode)


    # test_losses.append(np.array(test_loss_list) / len(X_test_tensor))
    # running_losses.append(np.array(running_loss_list) / len(X_train_tensor))
    print('Training and testing finished')

    # Save encoder and main network
    model_path = f'{output_dir}/final_models/'
    os.makedirs(model_path, exist_ok=True)
    torch.save(encoder.state_dict(), f"{model_path}/encoder_{mode}.pt")
    torch.save(main_network.state_dict(), f"{model_path}/main_network_{mode}.pt")
    print('Models saved')
        
    return test_losses, running_losses

