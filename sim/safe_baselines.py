"""
safe_baselines.py
-----------------
Constrained and risk-sensitive RL baselines, on the PPO backbone, network sizes,
rollout length and data of the proposed controller, so that a comparison
isolates the constraint mechanism.

  LagrangianPPO  Expected-cost PPO-Lagrangian (Tessler et al., ICLR 2019; Ray
                 et al., 2019). Risk critic, actor advantage and multiplier all
                 use the raw per-slot loss; the constraint is E[l] <= d_mean.

  CRPO           Constraint-Rectified Policy Optimization (Xu, Liang and Lan,
                 ICML 2021). No multiplier: each update descends the energy
                 cost when the rollout estimate of the constraint is within
                 tolerance and the constraint cost otherwise. The constraint is
                 the same per-slot CVaR surrogate g_tau as the proposed method,
                 so the comparison isolates the constraint-handling rule.

  WCSAC_GS       PPO adaptation of WCSAC with a Gaussian safety critic (Yang,
                 Simao, Tindemans and Spaan, Machine Learning 2023). The safety
                 critic estimates the mean J(s) and variance M(s) of the
                 normalized discounted loss-return G = (1-gamma) sum gamma^k l,
                 and the constraint is on their Eq. (13) statistic
                     Gamma_pi(s) = J(s) + k_alpha sqrt(M(s)) <= d_ret.
                 SAC differentiates Gamma_pi through a Q-network; PPO has no
                 action-value critic, so the actor instead uses the likelihood-
                 ratio gradient of J + k_alpha sqrt(M), estimated from GAE on
                 the loss (for J) and on the local variance u = delta_J^2 with
                 discount gamma^2 (for M), weighted by k_alpha / (2 sqrt(M)).

Budget translation. The per-slot CVaR budget Gamma has no numerical equivalent
for a mean or for a return statistic. Each baseline's budget is therefore set
to be equally tight relative to the always-on reference, measured on
validation traces: d = (Gamma / Gamma_AO) * (the baseline's constrained
statistic evaluated on the always-on loss process). See r2.calibrate().

Up to R2 these classes had implementation defects that invalidated the
comparison: PPO-Lagrangian's multiplier used the mean loss while its critic and
actor used g_tau; CRPO routed its update through a multiplier of 1e6 that was
also fed to the networks as an input of 2e4, saturating every first-layer
unit; WCSAC fitted one critic first to normalized Monte Carlo returns and then
to unnormalized GAE returns, ignored episode boundaries, used g_tau as its cost
and had no risk term in its actor gradient.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

from .config import SimCfg
from .algorithm import SafeRLController
from .networks import Critic


def gaussian_cvar_coeff(alpha: float) -> float:
    """alpha^-1 * phi(Phi^-1(alpha)); 2.063 at alpha = 0.05."""
    return float(norm.pdf(norm.ppf(alpha)) / alpha)


def _clone_with(cfg: SimCfg, **algo_overrides) -> SimCfg:
    import copy
    c = copy.deepcopy(cfg)
    for k, v in algo_overrides.items():
        setattr(c.algo, k, v)
    return c


def _std(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# ---------------------------------------------------------------------------
class LagrangianPPO(SafeRLController):
    """Expected-cost Lagrangian PPO: constraint E[l] <= d_mean."""

    def __init__(self, cfg: SimCfg, seed: int, d_mean: float = None, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False)
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        self.d = float(d_mean if d_mean is not None else cfg.algo.Gamma)

    def risk_cost(self, loss, g_tau):
        return loss

    def update_actor_critic(self, batch):
        cfg = self.cfg.algo
        if self.t_slot >= cfg.risk_warmup_slots:
            viol = float(np.mean(batch["loss"])) - self.d
            step = cfg.lr_dual * cfg.rollout_slots
            self.lam = float(np.clip(self.lam + step * viol, 0.0, cfg.lam_max))
        return super().update_actor_critic(batch)


# ---------------------------------------------------------------------------
class CRPO(SafeRLController):
    """Constraint-Rectified Policy Optimization (Xu et al., ICML 2021)."""

    def __init__(self, cfg: SimCfg, seed: int, tol: float = 0.0, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False, fixed_lambda=-1.0)
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        self.tol = tol
        self._on_constraint = False

    def augment(self, env_state):
        # CRPO has no multiplier. The networks see the current update mode
        # (0 = objective, 1 = constraint) in the multiplier slot, a bounded
        # input, together with tau.
        if not self.cfg.algo.dual_in_state:
            return env_state.astype(np.float32)
        extra = np.array([1.0 if self._on_constraint else 0.0,
                          self.tau / max(self.cfg.algo.ell_max, 1e-6)],
                         dtype=np.float32)
        return np.concatenate([env_state, extra]).astype(np.float32)

    def update_actor_critic(self, batch):
        # Constraint estimate: rollout mean of the per-slot CVaR surrogate.
        jc = float(np.mean(batch["g_tau"]))
        self._on_constraint = jc > self.cfg.algo.Gamma + self.tol
        # Route the actor fully onto one channel of the mixed advantage
        # (w = lam/(1+lam) -> 1 or 0) through the batch only; nothing large
        # ever reaches the networks' inputs.
        batch["lams"] = np.full_like(batch["lams"],
                                     1e9 if self._on_constraint else 0.0)
        self.lam = 0.0
        info = super().update_actor_critic(batch)
        info["on_constraint"] = float(self._on_constraint)
        return info


# ---------------------------------------------------------------------------
class WCSAC_GS(SafeRLController):
    """PPO adaptation of WCSAC with a Gaussian safety critic."""

    def __init__(self, cfg: SimCfg, seed: int, d_ret: float = None, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False, two_critics=True)
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        a = cfg.algo
        # critic_r (inherited) estimates J(s): the value of the per-slot cost
        # (1-gamma) * l with discount gamma. critic_m estimates M(s), the
        # variance of G, through the variance Bellman equation
        #   M(s) = E[delta_J^2 + gamma^2 M(s')],
        # delta_J = (1-gamma) l + gamma J(s') - J(s).
        self.critic_m = Critic(cfg.state_dim, hidden=a.hidden,
                               n_layers=a.n_layers).to(self.device)
        self.critic_m_target = Critic(cfg.state_dim, hidden=a.hidden,
                                      n_layers=a.n_layers).to(self.device)
        self.critic_m_target.load_state_dict(self.critic_m.state_dict())
        self.opt_critic_m = torch.optim.Adam(self.critic_m.parameters(),
                                             lr=a.lr_critic)
        self.k_alpha = gaussian_cvar_coeff(1.0 - a.beta)
        self.d = float(d_ret if d_ret is not None else a.Gamma)
        self.last_gamma_pi = 0.0

    def risk_cost(self, loss, g_tau):
        return (1.0 - self.cfg.algo.gamma_disc) * loss

    def update_actor_critic(self, batch, critics_only: bool = False):
        """PPO step with the Gaussian safety critic. With critics_only the
        actor and multiplier are left unchanged, which is how the budget is
        calibrated: the safety critic is fitted to the always-on policy."""
        cfg = self.cfg.algo
        dev = self.device
        g = cfg.gamma_disc
        lam_gae = cfg.gae_lambda
        states = torch.from_numpy(batch["states"]).to(dev)
        next_states = torch.from_numpy(batch["next_states"]).to(dev)
        actions = torch.from_numpy(batch["actions"]).to(dev)
        raws = torch.from_numpy(batch["raws"]).to(dev)
        lp_old = torch.from_numpy(batch["log_probs"]).to(dev)
        dones = torch.from_numpy(batch["dones"]).to(dev)
        c_e = torch.from_numpy(batch["costs_e"]).to(dev)
        c_j = torch.from_numpy(batch["costs_r"]).to(dev)   # (1-gamma) * loss

        def gae(cost, V_now, V_next, disc):
            d = cost + disc * (1 - dones) * V_next - V_now
            adv = torch.zeros_like(d)
            last = 0.0
            for k in reversed(range(len(d))):
                last = d[k] + disc * lam_gae * (1 - dones[k]) * last
                adv[k] = last
            return adv, adv + V_now, d

        with torch.no_grad():
            adv_e, ret_e, _ = gae(c_e, self.critic(states),
                                  self.critic_target(next_states), g)
            J_now = self.critic_r(states)
            adv_j, ret_j, dJ = gae(c_j, J_now,
                                   self.critic_r_target(next_states), g)
            u = dJ.pow(2)                                  # local variance
            M_now = self.critic_m(states).clamp(min=0.0)
            adv_m, ret_m, _ = gae(u, M_now,
                                  self.critic_m_target(next_states).clamp(min=0.0),
                                  g * g)
            sd = M_now.clamp(min=1e-8).sqrt()
            gamma_pi_s = J_now + self.k_alpha * sd
            self.last_gamma_pi = float(gamma_pi_s.mean())
            adv_risk = adv_j + self.k_alpha * adv_m / (2.0 * sd.clamp(min=1e-3))

        if self.t_slot >= cfg.risk_warmup_slots and not critics_only:
            step = cfg.lr_dual * cfg.rollout_slots
            self.lam = float(np.clip(
                self.lam + step * (self.last_gamma_pi - self.d),
                0.0, cfg.lam_max))
        lam = torch.full_like(adv_e, self.lam)
        w = lam / (1.0 + lam)
        with torch.no_grad():
            advs = _std((1.0 - w) * _std(adv_e) + w * _std(adv_risk))

        n = len(states)
        bs = max(1, n // cfg.n_mini)
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                lp_new = self.actor.log_prob(states[idx], actions[idx], raws[idx])
                ratio = (lp_new - lp_old[idx]).exp()
                a_b = advs[idx]
                actor_loss = torch.max(
                    ratio * a_b,
                    torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a_b
                ).mean()
                if not critics_only:
                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.opt_actor.step()
                for net, opt, tgt in ((self.critic, self.opt_critic, ret_e),
                                      (self.critic_r, self.opt_critic_r, ret_j),
                                      (self.critic_m, self.opt_critic_m, ret_m)):
                    opt.zero_grad()
                    F.mse_loss(net(states[idx]), tgt[idx]).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

        with torch.no_grad():
            for net, tgt in ((self.critic, self.critic_target),
                             (self.critic_r, self.critic_r_target),
                             (self.critic_m, self.critic_m_target)):
                for p, p_t in zip(net.parameters(), tgt.parameters()):
                    p_t.data.mul_(1 - cfg.target_tau)
                    p_t.data.add_(p.data * cfg.target_tau)
        return {"lambda": self.lam, "tau": self.tau, "z": self.z,
                "wcsac_gamma_pi": self.last_gamma_pi, "wcsac_budget": self.d}


# ---------------------------------------------------------------------------
def make_safe_baseline(name: str, cfg: SimCfg, seed: int, budgets: dict = None):
    budgets = budgets or {}
    if name == "LagrangianPPO":
        return LagrangianPPO(cfg, seed=seed, d_mean=budgets.get("d_mean"))
    if name == "CRPO":
        return CRPO(cfg, seed=seed)
    if name in ("WCSAC", "WCSAC_GS"):
        return WCSAC_GS(cfg, seed=seed, d_ret=budgets.get("d_ret"))
    raise ValueError(f"unknown safe baseline: {name}")
