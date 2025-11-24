import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 5
HIDDEN = 32
OUTPUT_DIM = 2

class Net(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=HIDDEN, out_dim=OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # outputs in [-1,1]
        return x
