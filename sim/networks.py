"""
networks.py
-----------
Actor and critic MLPs used by the safe-RL algorithm.

Actor: Gaussian policy over per-cell resource fraction phi in [0, 1].
       The raw output is passed through a sigmoid; log_std is a learnable
       per-dimension parameter (state-independent for variance stability).
       Log-prob is computed under the pre-sigmoid Gaussian with a
       tanh-correction (standard SAC/PPO trick).

Critic: scalar V(s) for the augmented cost c_lambda = energy + lambda * g_tau.
        Two critics (online + target) keep training stable; target is
        soft-updated.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy over pre-sigmoid actions. The action emitted to the
    env is sigmoid(raw_action) elementwise, in [0, 1]."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64,
                 n_layers: int = 2):
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden, n_layers)
        # Per-dim log-std as a learnable parameter, init small.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: torch.Tensor):
        """Returns (mean_pre, log_std)."""
        mean_pre = self.net(state)
        return mean_pre, self.log_std.expand_as(mean_pre)

    def sample(self, state: torch.Tensor):
        """Stochastic action sample with log-prob under the policy."""
        mean_pre, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean_pre, std)
        raw = normal.rsample()
        # Tanh-Sigmoid mapping. We use sigmoid for [0, 1] support.
        action = torch.sigmoid(raw)
        # Change-of-variable: log p(action) = log p(raw) - sum log d sigmoid / d raw
        # d sigmoid(x) / dx = sigmoid(x) * (1 - sigmoid(x))
        log_prob = normal.log_prob(raw).sum(-1)
        log_prob = log_prob - (action * (1 - action) + 1e-8).log().sum(-1)
        return action, log_prob, raw

    def log_prob(self, state: torch.Tensor, action: torch.Tensor,
                 raw: torch.Tensor) -> torch.Tensor:
        """Re-evaluate log-prob of a previously-sampled action+raw under the
        current policy. Used for PPO ratio computation."""
        mean_pre, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean_pre, std)
        lp = normal.log_prob(raw).sum(-1)
        lp = lp - (action * (1 - action) + 1e-8).log().sum(-1)
        return lp


class Critic(nn.Module):
    """V(state) for the augmented (energy + lambda * risk) cost."""

    def __init__(self, state_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden, n_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
