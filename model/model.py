import torch
import torch.nn as nn

class DeepSetEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_encoder=2, max_encoder=1024, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, encoding_size)

    def forward(self, x):  # x: [num_items, input_size] (可变长度集合)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        encoded = torch.sum(x, dim=0)  # DeepSets 的 sum 聚合
        return encoded


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_main=2, max_main=1024, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 1)  # 回归头（基座）

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x  # 回归值

