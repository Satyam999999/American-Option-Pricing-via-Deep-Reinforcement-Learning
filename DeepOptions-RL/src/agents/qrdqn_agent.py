import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QRDQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_quantiles, hidden_dim=128):
        super(QRDQN, self).__init__()
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * num_quantiles) 
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x)
        return x.view(batch_size, self.output_dim, self.num_quantiles)

class QRDQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.action_dim = action_dim
        self.num_quantiles = config['agent']['num_quantiles']
        self.gamma = config['agent']['gamma']
        self.lr = config['agent']['learning_rate']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QRDQN(state_dim, action_dim, self.num_quantiles).to(self.device)
        self.target_net = QRDQN(state_dim, action_dim, self.num_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        tau = torch.arange(0, self.num_quantiles + 1, device=self.device, dtype=torch.float32) / self.num_quantiles
        self.tau_hat = ((tau[:-1] + tau[1:]) / 2.0).view(1, self.num_quantiles)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                dist = self.policy_net(state_t)
                mean_q = dist.mean(dim=2)
                return mean_q.argmax().item()

    def optimize_model(self, memory):
        if len(memory) < self.config['agent']['batch_size']: return None
        
        states, actions, rewards, next_states, dones = memory.sample()
        
        current_dist = self.policy_net(states)
        batch_idx = torch.arange(len(states), device=self.device)
        current_quantiles = current_dist[batch_idx, actions.squeeze(), :] 

        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_actions = next_dist.mean(dim=2).argmax(dim=1)
            next_quantiles = next_dist[batch_idx, next_actions, :]
            
            rewards = rewards.expand_as(next_quantiles)
            dones = dones.expand_as(next_quantiles)
            target_quantiles = rewards + (self.gamma * next_quantiles * (1 - dones))

        # Quantile Huber Loss
        u = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        k = 1.0
        absolute_u = u.abs()
        huber_loss = torch.where(absolute_u <= k, 0.5 * u.pow(2), k * (absolute_u - 0.5 * k))
        
        delta = (u < 0).float().detach()
        element_wise_loss = torch.abs(self.tau_hat.unsqueeze(1) - delta) * huber_loss
        loss = element_wise_loss.sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self, tau=0.005):
        """Soft Update"""
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)