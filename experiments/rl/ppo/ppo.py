"""PPO Agent with MLP Actor/Critic and HTJU-family optimizers.

Reference hyperparameters from HTJU paper (Appendix D, Table D.2–D.3).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from htju import HTJU


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        mu = self.mu_head(out)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std

    def distribution(self, mu, std):
        return torch.distributions.Normal(mu, std)

    def choose_action(self, state):
        mu, std = self.forward(state)
        dist = self.distribution(mu, std)
        action = dist.sample()
        return action.detach().numpy()


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        return self.fc3(out)


class PPO:
    def __init__(self, state_dim, action_dim, opt_name='htju',
                 batch_size=256, epsilon=0.2, gamma=0.99, lambd=0.95,
                 total_epochs=500):
        self.actor_net = Actor(state_dim, action_dim)
        self.critic_net = Critic(state_dim)

        # Optimizer selection with paper hyperparameters
        if opt_name == 'htju':
            # HTJU paper Table D.2: actor + critic share same HTJU config
            self.actor_optim = HTJU(
                self.actor_net.parameters(),
                t_lr=3e-4, h_lr=3e-4,
                t_weight_decay=1e-3, h_weight_decay=1e-3,
                t_betas=(0.9, 0.999), h_betas=(0.9, 0.999),
                H_optim='Adam', weight_decay_type='AdamW',
                total_epoch=total_epochs, epoch_now=0,
                hessian_scale=0.05,
            )
            self.critic_optim = HTJU(
                self.critic_net.parameters(),
                t_lr=3e-4, h_lr=3e-4,
                t_weight_decay=1e-3, h_weight_decay=1e-3,
                t_betas=(0.9, 0.999), h_betas=(0.9, 0.999),
                H_optim='Adam', weight_decay_type='AdamW',
                total_epoch=total_epochs, epoch_now=0,
                hessian_scale=0.05,
            )
        elif opt_name == 'adam':
            self.actor_optim = optim.Adam(
                self.actor_net.parameters(), lr=3e-4, betas=(0.9, 0.999))
            self.critic_optim = optim.Adam(
                self.critic_net.parameters(), lr=3e-4, betas=(0.9, 0.999))
        elif opt_name == 'sgd':
            self.actor_optim = optim.SGD(
                self.actor_net.parameters(), lr=3e-4, momentum=0.9)
            self.critic_optim = optim.SGD(
                self.critic_net.parameters(), lr=3e-4, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        self.opt_name = opt_name
        self.critic_loss_func = nn.MSELoss()
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambd = lambd
        self.total_epochs = total_epochs

    def train(self, memory, epoch_idx):
        states, actions, rewards, masks = zip(*memory)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            values = self.critic_net(states).squeeze(1)
            returns, advants = self.get_gae(rewards, masks, values)
            mu, std = self.actor_net(states)
            pi_old = self.actor_net.distribution(mu, std)
            old_log_probs = pi_old.log_prob(actions).sum(1, keepdim=True)

        # Linear entropy decay over training
        curr_ent = max(0.01, 0.1 * (1 - epoch_idx / self.total_epochs))

        n = len(states)
        arr = np.arange(n)
        for _ in range(10):
            np.random.shuffle(arr)
            for i in range(n // self.batch_size):
                b_index = arr[self.batch_size * i: self.batch_size * (i + 1)]
                b_advants = advants[b_index].unsqueeze(1)
                b_advants = (b_advants - b_advants.mean()) / (b_advants.std() + 1e-8)

                mu_n, std_n = self.actor_net(states[b_index])
                pi = self.actor_net.distribution(mu_n, std_n)
                new_log_probs = pi.log_prob(actions[b_index]).sum(1, keepdim=True)

                ratio = torch.exp(new_log_probs - old_log_probs[b_index])
                surr1 = ratio * b_advants
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * b_advants

                actor_loss = -torch.min(surr1, surr2).mean() - curr_ent * pi.entropy().mean()
                critic_loss = self.critic_loss_func(
                    self.critic_net(states[b_index]), returns[b_index].unsqueeze(1))

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

    def get_gae(self, rewards, masks, values):
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns, running_advants, next_value = 0, 0, 0
        for t in reversed(range(len(rewards))):
            running_returns = rewards[t] + self.gamma * running_returns * masks[t]
            td_error = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            running_advants = td_error + self.gamma * self.lambd * running_advants * masks[t]
            returns[t] = running_returns
            advants[t] = running_advants
            next_value = values[t]
        return returns, advants
