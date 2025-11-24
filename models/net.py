import torch
import torch.nn as nn
import torch.nn.functional as F

# Observation vector length
INPUT_DIM = 7  # [d1,d2,d3,speed,heading,dist_to_finish,progress]
HIDDEN = 64
OUTPUT_DIM = 2  # steering, throttle

class ActorCritic(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=HIDDEN, out_dim=OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # Policy head
        self.mu = nn.Linear(hidden, out_dim)

        # Value head
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs):
        # Ensure obs is 2D: (batch_size x input_dim)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        action = torch.tanh(self.mu(x))  # steering + throttle in [-1,1]
        value = self.value(x)            # scalar value per car

        return action, value  # DO NOT squeeze batch dimension
