"""
safe_baselines.py
-----------------
Modern constrained / risk-sensitive RL baselines requested by the reviewers.

All three share the PPO backbone, network sizes, rollout length and data of
the proposed controller, so a comparison isolates the *constraint mechanism*
rather than incidental implementation differences:

  LagrangianPPO  Expected-cost constraint via a plain integral dual, i.e. the
                 RCPO/PPO-Lagrangian baseline (Tessler et al., ICLR 2019;
                 Ray et al., 2019). Shows what is lost by constraining the
                 mean of the per-slot loss instead of its tail.

  CRPO           Constraint-Rectified Policy Optimization (Xu, Liang & Lan,
                 ICML 2021). No dual variable at all: each update follows the
                 objective gradient while feasible and switches to the
                 constraint gradient once the violation exceeds a tolerance.
                 Included because it carries convergence guarantees, making
                 it the fairest theoretical comparator.

  WCSAC_GS       Worst-Case Soft Actor-Critic, Gaussian-approximation variant
                 (Yang, Simao, Tindemans & Spaan, Machine Learning 2023).
                 A distributional safety critic estimates the mean and
                 variance of the discounted cost-return and constrains
                     Gamma_pi = Q_c + alpha^-1 phi(Phi^-1(alpha)) sqrt(V_c),
                 their Eq. (13). Note this is CVaR of the RETURN, whereas the
                 proposed method constrains CVaR of the PER-SLOT loss; the
                 constraint is applied in the equivalent normalized form
                 (1 - gamma) * Q_c <= Gamma, which keeps the safety critic on
                 the same numerical scale as the other critics.
                 That difference is itself a contribution axis and is
                 reported rather than hidden.
"""
from __future__ import annotations
from typing import Dict
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


class LagrangianPPO(SafeRLController):
    """Expected-cost Lagrangian PPO (no CVaR, no safety filter)."""

    def __init__(self, cfg: SimCfg, seed: int, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False)
        # enforce_risk=False disables SafeRLController's own per-slot dual
        # ascent. Without this the parent's multiplier update runs inside
        # collect_rollout and overwrites the one this baseline computes,
        # so the baseline would not be running its own algorithm.
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)

    def update_actor_critic(self, batch):
        # Dual on the MEAN per-slot loss, not on g_tau. The step is scaled by
        # the rollout length so that this per-update ascent has the same
        # effective rate as a per-slot ascent at lr_dual, which is what the
        # proposed controller's dual sees. Without the scaling the baseline
        # would be handicapped by a factor of rollout_slots.
        cfg = self.cfg.algo
        if self.t_slot >= cfg.risk_warmup_slots:
            viol = float(np.mean(batch["loss"])) - cfg.Gamma
            step = cfg.lr_dual * cfg.rollout_slots
            self.lam = float(np.clip(self.lam + step * viol,
                                     0.0, cfg.lam_max))
        return super().update_actor_critic(batch)


class CRPO(SafeRLController):
    """Constraint-Rectified Policy Optimization (Xu et al., ICML 2021)."""

    def __init__(self, cfg: SimCfg, seed: int, tol: float = 0.0, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False, fixed_lambda=-1.0)
        # See LagrangianPPO: the parent's per-slot dual must be off so that
        # CRPO's own objective selection is what drives the actor.
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        self.tol = tol
        self._on_constraint = False

    def update_actor_critic(self, batch):
        # CRPO has no dual: pick which objective this update descends.
        jc = float(np.mean(batch["g_tau"]))
        self._on_constraint = jc > self.cfg.algo.Gamma + self.tol
        # lam = 1 routes the mixed advantage fully onto the risk channel,
        # lam = 0 fully onto the energy channel (w = lam/(1+lam) with the
        # mixing weight forced to the corresponding extreme).
        self.lam = 1e6 if self._on_constraint else 0.0
        # The parent mixes advantages with the per-slot lambdas recorded during
        # the rollout. CRPO decides per update, so stamp the decision onto the
        # batch; otherwise the choice never reaches the actor.
        batch["lams"] = np.full_like(batch["lams"], self.lam)
        info = super().update_actor_critic(batch)
        info["on_constraint"] = float(self._on_constraint)
        return info


class WCSAC_GS(SafeRLController):
    """WCSAC with a Gaussian safety critic, on the PPO backbone."""

    def __init__(self, cfg: SimCfg, seed: int, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False, two_critics=True)
        # See LagrangianPPO: the parent's per-slot dual on the per-slot
        # surrogate must be off, since WCSAC constrains the cost-RETURN.
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        # Second-moment head for the cost-return, giving the variance
        # V_c = E[C^2] - Q_c^2 used by their Eq. (13).
        self.critic_r2 = Critic(cfg.state_dim, hidden=cfg.algo.hidden,
                                n_layers=cfg.algo.n_layers).to(self.device)
        self.opt_critic_r2 = torch.optim.Adam(self.critic_r2.parameters(),
                                              lr=cfg.algo.lr_critic)
        self.k_alpha = gaussian_cvar_coeff(1.0 - cfg.algo.beta)
        # The constraint is on the discounted cost-RETURN, whose scale is
        # 1/(1-gamma) ~ 100x the per-slot cost. Regressing a target of order
        # 1000 from a zero initialization is far slower than the actor's drift
        # to the energy minimum, so the dual never engages and the baseline
        # fails for a reason that belongs to the port rather than to WCSAC.
        # Working in the equivalent normalized form (1-gamma) * Q_c <= Gamma
        # puts the safety critic on the same scale as every other critic here.
        self.d_return = cfg.algo.Gamma

    def update_actor_critic(self, batch):
        cfg = self.cfg.algo
        states = torch.from_numpy(batch["states"]).to(self.device)
        costs_r = torch.from_numpy(batch["costs_r"]).to(self.device)

        # Monte-Carlo discounted cost-return targets for the two moments.
        with torch.no_grad():
            g = torch.zeros_like(costs_r)
            run = 0.0
            for k in reversed(range(len(costs_r))):
                run = costs_r[k] + cfg.gamma_disc * run
                g[k] = run
            g = g * (1.0 - cfg.gamma_disc)   # normalize to per-slot scale
            q_pred = self.critic_r(states)
            m2_pred = self.critic_r2(states)
            var = torch.clamp(m2_pred - q_pred.pow(2), min=1e-6)
            gamma_pi = float((q_pred + self.k_alpha * var.sqrt()).mean())

        # Fit both moments.
        for _ in range(cfg.n_epochs):
            self.opt_critic_r.zero_grad()
            F.mse_loss(self.critic_r(states), g).backward()
            self.opt_critic_r.step()
            self.opt_critic_r2.zero_grad()
            F.mse_loss(self.critic_r2(states), g.pow(2)).backward()
            self.opt_critic_r2.step()

        if self.t_slot >= cfg.risk_warmup_slots:
            # Per-update step scaled by rollout length, as in LagrangianPPO.
            # No further rescaling is needed because gamma_pi and d_return are
            # both already on the per-slot scale.
            step = cfg.lr_dual * cfg.rollout_slots
            self.lam = float(np.clip(
                self.lam + step * (gamma_pi - self.d_return),
                0.0, cfg.lam_max))

        info = super().update_actor_critic(batch)
        info["wcsac_gamma_pi"] = gamma_pi
        info["wcsac_budget"] = self.d_return
        return info


def _clone_with(cfg: SimCfg, **algo_overrides) -> SimCfg:
    """Shallow copy of cfg with AlgoCfg fields overridden."""
    import copy
    c = copy.deepcopy(cfg)
    for k, v in algo_overrides.items():
        setattr(c.algo, k, v)
    return c


def make_safe_baseline(name: str, cfg: SimCfg, seed: int):
    if name == "LagrangianPPO":
        return LagrangianPPO(cfg, seed=seed)
    if name == "CRPO":
        return CRPO(cfg, seed=seed)
    if name in ("WCSAC", "WCSAC_GS"):
        return WCSAC_GS(cfg, seed=seed)
    raise ValueError(f"unknown safe baseline: {name}")
