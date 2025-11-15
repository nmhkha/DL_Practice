import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms

class SimpleCNN(nn.Module):
    def __init__(self, output_size=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.permute(0, 2, 1) # --> (batch, 1, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1) # --> (batch, 64)
        return self.fc(x)