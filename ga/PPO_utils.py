import torch
import torch.nn as nn
import torch.optim as optim

# --- Hyperparameters ---
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
UPDATE_EPOCHS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- PPO Buffer ---
class PPOBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def get_tensors(self):
        obs_tensor = torch.stack(self.obs).to(device)
        actions_tensor = torch.stack(self.actions).to(device)
        log_probs_tensor = torch.stack(self.log_probs).to(device)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values_tensor = torch.stack(self.values).to(device)
        return obs_tensor, actions_tensor, log_probs_tensor, rewards_tensor, dones_tensor, values_tensor


# --- PPO Update ---
def ppo_update(agent, buffer: PPOBuffer, optimizer):
    obs, actions, old_log_probs, rewards, dones, values = buffer.get_tensors()
    
    # compute advantages and returns
    returns = []
    advantages = []
    gae = 0
    last_val = 0
    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        delta = rewards[i] + GAMMA * last_val * mask - values[i].squeeze()
        gae = delta + GAMMA * LAMBDA * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i].squeeze())
        last_val = values[i].squeeze()

    returns = torch.stack(returns).detach()
    advantages = torch.stack(advantages).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(UPDATE_EPOCHS):
        action_preds, value_preds = agent(obs)
        # ensure action_preds matches actions shape
        dist = torch.distributions.Normal(action_preds, 0.1)
        log_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(log_probs - old_log_probs.squeeze())

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = ((returns - value_preds.squeeze()) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
