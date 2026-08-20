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
                 n_layers: int = 2, log_std_init: float = -0.5):
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden, n_layers)
        # Per-dim log-std as a learnable parameter. The initial scale governs
        # how much pre-sigmoid exploration noise the actor injects; at -0.5
        # (std ~0.61) the noise is large enough to wash out state-dependence
        # early in training, so it is exposed for tuning.
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

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


class FactoredActor(nn.Module):
    """Per-cell factored policy: a discrete sleep/active decision times a
    continuous resource share conditional on being active.

    The purely continuous Actor above emits phi = sigmoid(raw) with
    raw ~ N(~0, 0.61), so reaching the sleep region phi < EPS_SLEEP = 0.05
    requires raw < -2.94, about 4.85 sigma out. That is ~6e-7 per cell-slot,
    so across a 70k-draw episode the actor essentially never explores sleep
    and is confined to the throttling sub-problem, which is exactly where a
    per-cell myopic rule is already near-optimal. Factoring the decision
    gives sleep an O(1) exploration probability and matches how the O-RAN
    energy-saving literature parameterizes cell on/off (Bordin et al., CCNC
    2025; Kairos, INFOCOM 2025).

    Emitted action is 0 for a sleeping cell and clip(phi, eps, 1) otherwise,
    which is what the environment's sleep threshold expects.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64,
                 n_layers: int = 2, log_std_init: float = -0.5,
                 eps_sleep: float = 0.05, active_bias: float = 1.0):
        super().__init__()
        self.B = action_dim
        self.eps = eps_sleep
        # 2B outputs: active-logits, then pre-sigmoid means for phi.
        self.net = _mlp(state_dim, 2 * action_dim, hidden, n_layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        # Start biased toward "active" so early training resembles the
        # always-on fallback rather than a blackout.
        self.active_bias = active_bias

    def _heads(self, state: torch.Tensor):
        out = self.net(state)
        logits = out[..., :self.B] + self.active_bias
        mean_pre = out[..., self.B:]
        return logits, mean_pre, self.log_std.expand_as(mean_pre)

    def forward(self, state: torch.Tensor):
        logits, mean_pre, log_std = self._heads(state)
        return mean_pre, log_std

    def _compose(self, z: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        phi = torch.sigmoid(raw).clamp(self.eps, 1.0)
        return z * phi

    def sample(self, state: torch.Tensor):
        logits, mean_pre, log_std = self._heads(state)
        std = log_std.exp()
        bern = torch.distributions.Bernoulli(logits=logits)
        z = bern.sample()
        normal = torch.distributions.Normal(mean_pre, std)
        raw = normal.rsample()
        action = self._compose(z, raw)
        lp = self._logp(logits, mean_pre, std, z, raw)
        # Pack (z, raw) so the PPO ratio can be re-evaluated exactly.
        return action, lp, torch.cat([z, raw], dim=-1)

    @staticmethod
    def _logp(logits, mean_pre, std, z, raw):
        bern = torch.distributions.Bernoulli(logits=logits)
        lp = bern.log_prob(z).sum(-1)
        normal = torch.distributions.Normal(mean_pre, std)
        s = torch.sigmoid(raw)
        cont = normal.log_prob(raw) - (s * (1 - s) + 1e-8).log()
        # Score the continuous head only where the cell is actually active:
        # phi has no effect on the environment for a sleeping cell.
        return lp + (z * cont).sum(-1)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor,
                 packed: torch.Tensor) -> torch.Tensor:
        logits, mean_pre, log_std = self._heads(state)
        z, raw = packed[..., :self.B], packed[..., self.B:]
        return self._logp(logits, mean_pre, log_std.exp(), z, raw)

    @torch.no_grad()
    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        logits, mean_pre, _ = self._heads(state)
        z = (torch.sigmoid(logits) > 0.5).float()
        return self._compose(z, mean_pre)


class Critic(nn.Module):
    """V(state) for the augmented (energy + lambda * risk) cost."""

    def __init__(self, state_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden, n_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
