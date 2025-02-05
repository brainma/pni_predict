import torch
from torch import Tensor
import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, num_classes = 2):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout2(x)
        x = self.softmax(self.fc4(x))
        return x

