"""
algorithm.py
------------
Risk-Limited Safe Primal-Dual Actor-Critic for Energy-Aware O-RAN Scheduling.

Implements Algorithm 1 of the paper. Two-timescale primal-dual structure:

  Fast:  actor (policy gradient via PPO clip), critic (TD on c_lambda)
  Slow:  dual lambda (subgradient ascent on g_tau - Gamma)
         CVaR threshold tau (subgradient descent on Rockafellar-Uryasev)

A Lyapunov-guided safety filter projects exploratory actions onto a
backlog-aware safe set so that physical-queue stability is preserved even
during training.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F

from .config import SimCfg
from .networks import Actor, Critic
from .safety_filter import safe_project
from .env import CellularEnv


# ---------------------------------------------------------------------------
class SafeRLController:
    """Primal-dual safe actor-critic. Self-contained training loop.

    Attributes maintained across training (so we can checkpoint):
        actor, critic, critic_target
        lam (dual variable, >= 0)
        tau (CVaR threshold)
        z   (risk virtual queue, used for diagnostics)
        t_slot (global slot counter for Robbins-Monro schedules)
    """

    def __init__(self, cfg: SimCfg, seed: int, device: str = "cpu",
                 use_safety_filter: bool = True,
                 enforce_risk: bool = True):
        self.cfg = cfg
        self.device = device
        self.use_safety_filter = use_safety_filter
        self.enforce_risk = enforce_risk

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = Actor(cfg.state_dim, cfg.action_dim,
                           hidden=cfg.algo.hidden,
                           n_layers=cfg.algo.n_layers).to(device)
        self.critic = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                             n_layers=cfg.algo.n_layers).to(device)
        self.critic_target = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                                    n_layers=cfg.algo.n_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(),
                                          lr=cfg.algo.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                            lr=cfg.algo.lr_critic)

        self.lam = 0.0 if enforce_risk else 0.0
        self.tau = cfg.algo.tau_init
        self.z = 0.0
        self.t_slot = 0
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray, q_phys: np.ndarray,
            stochastic: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, float]:
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        if stochastic:
            a, lp, raw = self.actor.sample(s)
            a_np = a.cpu().numpy().squeeze(0).astype(np.float32)
            raw_np = raw.cpu().numpy().squeeze(0).astype(np.float32)
            lp_v = float(lp.cpu().numpy().item())
        else:
            mean_pre, _ = self.actor(s)
            raw_np = mean_pre.cpu().numpy().squeeze(0).astype(np.float32)
            a_np = (1.0 / (1.0 + np.exp(-raw_np))).astype(np.float32)
            lp_v = 0.0
        if self.use_safety_filter:
            a_np = safe_project(a_np, q_phys, self.cfg)
        return a_np, raw_np, state, lp_v

    # -----------------------------------------------------------------
    def update_tau_z_lambda(self, loss: float):
        """Slow-timescale updates: tau (CVaR threshold), risk virtual queue z,
        and dual lambda. Called once per step from the rollout loop."""
        cfg = self.cfg.algo
        t = max(self.t_slot, 1)
        # Robbins-Monro decay
        alpha_t = cfg.lr_tau / (1.0 + t * 1e-4)
        beta_t = cfg.lr_dual / (1.0 + t * 1e-4)

        # tau subgradient: 1 - (1-beta)^{-1} * 1{loss > tau}
        ind = 1.0 if loss > self.tau else 0.0
        g_grad = 1.0 - ind / (1.0 - cfg.beta)
        self.tau = float(np.clip(self.tau - alpha_t * g_grad,
                                  0.0, cfg.ell_max))

        # g_tau = tau + (1-beta)^{-1} max(loss - tau, 0)
        g_tau = self.tau + max(loss - self.tau, 0.0) / (1.0 - cfg.beta)

        # Risk virtual queue (diagnostic)
        self.z = max(self.z + g_tau - cfg.Gamma, 0.0)

        # Dual ascent on lambda (with cap and warmup).
        if self.enforce_risk and self.t_slot >= cfg.risk_warmup_slots:
            new_lam = max(self.lam + beta_t * (g_tau - cfg.Gamma), 0.0)
            self.lam = float(min(new_lam, cfg.lam_max))

        self.t_slot += 1
        return g_tau

    # -----------------------------------------------------------------
    def _augmented_cost(self, energy_W: float, g_tau: float) -> float:
        # Normalize energy to comparable scale to risk term.
        # energy ~ 200-1400 W; g_tau ~ 0-20 (after CVaR / ell_max normalization).
        # Scale energy by 1/1000 so combined cost is in similar units.
        return energy_W * 1e-3 + self.lam * g_tau

    # -----------------------------------------------------------------
    def collect_rollout(self, env: CellularEnv,
                        n_slots: int) -> Dict[str, np.ndarray]:
        """One rollout of length n_slots starting from env's current state."""
        states = np.zeros((n_slots, self.cfg.state_dim), dtype=np.float32)
        actions = np.zeros((n_slots, self.cfg.action_dim), dtype=np.float32)
        raws = np.zeros((n_slots, self.cfg.action_dim), dtype=np.float32)
        log_probs = np.zeros(n_slots, dtype=np.float32)
        costs = np.zeros(n_slots, dtype=np.float32)
        next_states = np.zeros((n_slots, self.cfg.state_dim), dtype=np.float32)
        dones = np.zeros(n_slots, dtype=np.float32)
        info = {"energy": [], "loss": [], "g_tau": [], "viol": []}

        s = env.state()
        for k in range(n_slots):
            a, raw, s_in, lp = self.act(s, env.q, stochastic=True)
            s_next, cost, done = env.step(a)
            g_tau = self.update_tau_z_lambda(cost["loss"])
            augmented = self._augmented_cost(cost["energy_W"], g_tau)

            states[k] = s
            actions[k] = a
            raws[k] = raw
            log_probs[k] = lp
            costs[k] = augmented
            next_states[k] = s_next
            dones[k] = float(done)
            info["energy"].append(cost["energy_W"])
            info["loss"].append(cost["loss"])
            info["g_tau"].append(g_tau)
            info["viol"].append(1.0 if cost["loss"] > self.cfg.algo.Gamma
                                else 0.0)

            if done:
                s = env.reset()
            else:
                s = s_next

        return {
            "states": states, "actions": actions, "raws": raws,
            "log_probs": log_probs, "costs": costs,
            "next_states": next_states, "dones": dones,
            "energy": np.array(info["energy"], dtype=np.float32),
            "loss": np.array(info["loss"], dtype=np.float32),
            "g_tau": np.array(info["g_tau"], dtype=np.float32),
            "viol": np.array(info["viol"], dtype=np.float32),
        }

    # -----------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg.algo
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        raws = torch.from_numpy(batch["raws"]).to(self.device)
        log_probs_old = torch.from_numpy(batch["log_probs"]).to(self.device)
        costs = torch.from_numpy(batch["costs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        # GAE advantages (cost-style: lower V means better state).
        with torch.no_grad():
            V_now = self.critic(states)
            V_next = self.critic_target(next_states)
            # For a COST MDP (we are minimizing), advantage:
            #   A = c + gamma V(next) - V(now)
            # Lower A means action led to lower-than-expected cost (better).
            deltas = costs + cfg.gamma_disc * (1 - dones) * V_next - V_now
            advs = torch.zeros_like(deltas)
            last = 0.0
            for k in reversed(range(len(deltas))):
                last = deltas[k] + cfg.gamma_disc * cfg.gae_lambda * \
                    (1 - dones[k]) * last
                advs[k] = last
            returns = advs + V_now
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)

        # PPO update over a few epochs of mini-batches.
        n = len(states)
        bs = max(1, n // cfg.n_mini)
        actor_losses, critic_losses = [], []
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                s_b = states[idx]
                a_b = actions[idx]
                r_b = raws[idx]
                lp_old_b = log_probs_old[idx]
                adv_b = advs[idx]
                ret_b = returns[idx]

                lp_new = self.actor.log_prob(s_b, a_b, r_b)
                ratio = (lp_new - lp_old_b).exp()
                # PPO clipped objective: minimize ratio * adv with clipping.
                # NOTE: cost-style means we MINIMIZE cost, so PPO-cost is
                # symmetric to PPO-reward with adv sign flipped.
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip,
                                    1 + cfg.ppo_clip) * adv_b
                actor_loss = torch.max(surr1, surr2).mean()

                V_pred = self.critic(s_b)
                critic_loss = F.mse_loss(V_pred, ret_b)

                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_critic.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))

        # Soft-update target critic.
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                               self.critic_target.parameters()):
                p_t.data.mul_(1 - cfg.target_tau)
                p_t.data.add_(p.data * cfg.target_tau)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "lambda": self.lam,
            "tau": self.tau,
            "z": self.z,
        }
