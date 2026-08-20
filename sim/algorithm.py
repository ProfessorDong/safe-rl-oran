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
                           n_layers=cfg.algo.n_layers,
                           log_std_init=cfg.algo.log_std_init).to(device)
        self.critic = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                             n_layers=cfg.algo.n_layers).to(device)
        self.critic_target = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                                    n_layers=cfg.algo.n_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(),
                                          lr=cfg.algo.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                            lr=cfg.algo.lr_critic)

        # Risk critic. `self.critic` regresses the ENERGY cost only when
        # two_critics is on, and the augmented cost c_lambda otherwise (the
        # submitted behaviour). Splitting them keeps each regression target
        # stationary in lambda.
        self.two_critics = bool(cfg.algo.two_critics)
        if self.two_critics:
            self.critic_r = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                                   n_layers=cfg.algo.n_layers).to(device)
            self.critic_r_target = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                                          n_layers=cfg.algo.n_layers).to(device)
            self.critic_r_target.load_state_dict(self.critic_r.state_dict())
            self.opt_critic_r = torch.optim.Adam(self.critic_r.parameters(),
                                                 lr=cfg.algo.lr_critic)
        else:
            self.critic_r = None

        self.lam = (cfg.algo.fixed_lambda if cfg.algo.fixed_lambda >= 0
                    else 0.0)
        # PID dual-controller state (Stooke et al., ICML 2020, Alg. 2).
        self.use_pid = bool(cfg.algo.use_pid_dual) and cfg.algo.fixed_lambda < 0
        self._pid_I = 0.0
        self._pid_prev_delta = 0.0
        self.tau = (cfg.algo.fixed_tau if cfg.algo.fixed_tau >= 0
                    else cfg.algo.tau_init)
        self.z = 0.0
        self.t_slot = 0
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    def augment(self, env_state: np.ndarray) -> np.ndarray:
        """Append the controller's own (lambda, tau) to the environment
        observation.

        The augmented cost c_lambda = V*P + lambda*g_tau depends on lambda and
        tau, so a policy/value function conditioned only on the environment
        state is solving a non-stationary problem: the same observation maps
        to different costs at different points in training. Feeding the dual
        state to the networks restores stationarity of the joint recursion
        and is what makes the primal-dual iteration well posed.
        """
        if not self.cfg.algo.dual_in_state:
            return env_state.astype(np.float32)
        extra = np.array([
            self.lam / max(self.cfg.algo.lam_max, 1e-6),
            self.tau / max(self.cfg.algo.ell_max, 1e-6),
        ], dtype=np.float32)
        return np.concatenate([env_state, extra]).astype(np.float32)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray, q_phys: np.ndarray,
            stochastic: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, float]:
        state = self.augment(state)
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
        if cfg.diminishing_schedule:
            # Theory-aligned Robbins-Monro: alpha_t = c / (1+t)^p with
            # beta exponent (slow timescale) > alpha exponent (faster).
            # Verifies Theorem 2 hypotheses under uncapped dual.
            alpha_t = cfg.lr_tau  / (1.0 + t) ** cfg.diminishing_beta_exp
            beta_t  = cfg.lr_dual / (1.0 + t) ** cfg.diminishing_beta_exp
        else:
            # Practical (slow) Robbins-Monro used in the headline experiments.
            alpha_t = cfg.lr_tau / (1.0 + t * 1e-4)
            beta_t = cfg.lr_dual / (1.0 + t * 1e-4)

        # tau subgradient: 1 - (1-beta)^{-1} * 1{loss > tau}
        # Skip the update if tau is frozen for an ablation.
        if cfg.fixed_tau < 0:
            ind = 1.0 if loss > self.tau else 0.0
            g_grad = 1.0 - ind / (1.0 - cfg.beta)
            self.tau = float(np.clip(self.tau - alpha_t * g_grad,
                                      0.0, cfg.ell_max))

        # g_tau = tau + (1-beta)^{-1} max(loss - tau, 0)
        g_tau = self.tau + max(loss - self.tau, 0.0) / (1.0 - cfg.beta)

        # Risk virtual queue (diagnostic)
        self.z = max(self.z + g_tau - cfg.Gamma, 0.0)

        # Dual ascent on lambda (with cap and warmup).
        # Skip the ascent if lambda is frozen for an ablation, and skip the
        # per-slot path entirely when the PID controller runs per rollout.
        pid_deferred = self.use_pid and cfg.pid_per_rollout
        if (cfg.fixed_lambda < 0 and self.enforce_risk and not pid_deferred
                and self.t_slot >= cfg.risk_warmup_slots):
            if self.use_pid:
                self.lam = self._pid_update(g_tau)
            else:
                new_lam = max(self.lam + beta_t * (g_tau - cfg.Gamma), 0.0)
                self.lam = float(min(new_lam, cfg.lam_max))

        self.t_slot += 1
        return g_tau

    # -----------------------------------------------------------------
    def _pid_update(self, jc: float) -> float:
        """PID-controlled Lagrange multiplier (Stooke et al., ICML 2020).

        `jc` is the measured constraint quantity (here the mean risk cost
        g_tau over the control interval) and the setpoint is Gamma. Unlike
        the integral-only rule, lambda is recomputed from the current error
        rather than accumulated, so a persistent violation no longer drives
        it monotonically into the cap.
        """
        cfg = self.cfg.algo
        delta = jc - cfg.Gamma
        # Derivative term is rectified so it resists increases in the
        # constraint but does not fight decreases.
        d_term = max(delta - self._pid_prev_delta, 0.0)
        self._pid_I = max(self._pid_I + delta, 0.0)
        lam = (cfg.pid_kp * delta + cfg.pid_ki * self._pid_I
               + cfg.pid_kd * d_term)
        self._pid_prev_delta = delta
        return float(min(max(lam, 0.0), cfg.lam_max))

    # -----------------------------------------------------------------
    def _augmented_cost(self, energy_W: float, g_tau: float) -> float:
        # c_lambda = V * P_RAN + lambda * g_tau. V_energy_weight defaults to 1e-3
        # to normalize power (~1000 W) to commensurable scale with g_tau (~3-10).
        return energy_W * self.cfg.algo.V_energy_weight + self.lam * g_tau

    # -----------------------------------------------------------------
    def collect_rollout(self, env: CellularEnv,
                        n_slots: int) -> Dict[str, np.ndarray]:
        """One rollout of length n_slots starting from env's current state."""
        states = np.zeros((n_slots, self.cfg.state_dim), dtype=np.float32)
        actions = np.zeros((n_slots, self.cfg.action_dim), dtype=np.float32)
        raws = np.zeros((n_slots, self.cfg.action_dim), dtype=np.float32)
        log_probs = np.zeros(n_slots, dtype=np.float32)
        costs = np.zeros(n_slots, dtype=np.float32)
        costs_e = np.zeros(n_slots, dtype=np.float32)
        costs_r = np.zeros(n_slots, dtype=np.float32)
        lams = np.zeros(n_slots, dtype=np.float32)
        next_states = np.zeros((n_slots, self.cfg.state_dim), dtype=np.float32)
        dones = np.zeros(n_slots, dtype=np.float32)
        info = {"energy": [], "loss": [], "g_tau": [], "viol": []}

        s = env.state()
        for k in range(n_slots):
            a, raw, s_in, lp = self.act(s, env.q, stochastic=True)
            lam_k = self.lam            # dual in force when the action was taken
            s_next, cost, done = env.step(a)
            g_tau = self.update_tau_z_lambda(cost["loss"])
            augmented = self._augmented_cost(cost["energy_W"], g_tau)

            states[k] = s_in            # augmented observation actually seen
            actions[k] = a
            raws[k] = raw
            log_probs[k] = lp
            costs[k] = augmented
            costs_e[k] = cost["energy_W"] * self.cfg.algo.V_energy_weight
            costs_r[k] = g_tau
            lams[k] = lam_k
            next_states[k] = self.augment(s_next)
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
            "costs_e": costs_e, "costs_r": costs_r, "lams": lams,
            "next_states": next_states, "dones": dones,
            "energy": np.array(info["energy"], dtype=np.float32),
            "loss": np.array(info["loss"], dtype=np.float32),
            "g_tau": np.array(info["g_tau"], dtype=np.float32),
            "viol": np.array(info["viol"], dtype=np.float32),
        }

    # -----------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg.algo
        # Per-rollout PID dual control. Stooke et al. apply feedback control
        # at the RL-iteration rate using a batch estimate of the constraint,
        # which is far less noisy than a single slot's realization.
        if (self.use_pid and cfg.pid_per_rollout and self.enforce_risk
                and cfg.fixed_lambda < 0
                and self.t_slot >= cfg.risk_warmup_slots):
            self.lam = self._pid_update(float(np.mean(batch["g_tau"])))
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        raws = torch.from_numpy(batch["raws"]).to(self.device)
        log_probs_old = torch.from_numpy(batch["log_probs"]).to(self.device)
        costs = torch.from_numpy(batch["costs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        def _gae(cost_vec, critic, critic_target):
            """GAE for a cost MDP: A = c + gamma V(next) - V(now), lower is
            better. Returns (advantage, bootstrapped return)."""
            V_now = critic(states)
            V_next = critic_target(next_states)
            deltas = cost_vec + cfg.gamma_disc * (1 - dones) * V_next - V_now
            adv = torch.zeros_like(deltas)
            last = 0.0
            for k in reversed(range(len(deltas))):
                last = deltas[k] + cfg.gamma_disc * cfg.gae_lambda * \
                    (1 - dones[k]) * last
                adv[k] = last
            return adv, adv + V_now

        def _std(x):
            return (x - x.mean()) / (x.std() + 1e-6)

        with torch.no_grad():
            if self.two_critics:
                costs_e = torch.from_numpy(batch["costs_e"]).to(self.device)
                costs_r = torch.from_numpy(batch["costs_r"]).to(self.device)
                lams = torch.from_numpy(batch["lams"]).to(self.device)
                adv_e, returns_e = _gae(costs_e, self.critic,
                                        self.critic_target)
                adv_r, returns_r = _gae(costs_r, self.critic_r,
                                        self.critic_r_target)
                # Standardize each channel FIRST, then mix with the dual.
                # Standardizing the mixture instead (the submitted code path)
                # divides out lambda, so the constraint loses all influence on
                # the actor no matter how large the dual grows. The 1/(1+lam)
                # normalization keeps the mixture bounded while sweeping the
                # objective from pure energy (lam=0) to pure risk (lam -> inf).
                w = lams / (1.0 + lams)
                advs = (1.0 - w) * _std(adv_e) + w * _std(adv_r)
                advs = _std(advs)
                returns = returns_e
            else:
                adv, returns = _gae(costs, self.critic, self.critic_target)
                advs = _std(adv)
                returns_r = None

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

                if self.two_critics:
                    V_r_pred = self.critic_r(s_b)
                    critic_r_loss = F.mse_loss(V_r_pred, returns_r[idx])
                    self.opt_critic_r.zero_grad()
                    critic_r_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_r.parameters(), 1.0)
                    self.opt_critic_r.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))

        # Soft-update target critics.
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                               self.critic_target.parameters()):
                p_t.data.mul_(1 - cfg.target_tau)
                p_t.data.add_(p.data * cfg.target_tau)
            if self.two_critics:
                for p, p_t in zip(self.critic_r.parameters(),
                                   self.critic_r_target.parameters()):
                    p_t.data.mul_(1 - cfg.target_tau)
                    p_t.data.add_(p.data * cfg.target_tau)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "lambda": self.lam,
            "tau": self.tau,
            "z": self.z,
        }
