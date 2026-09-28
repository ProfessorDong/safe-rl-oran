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

  WCSAC_GS       WCSAC with a Gaussian safety critic (Yang, Simao, Tindemans
                 and Spaan, Machine Learning 2023) on the PPO backbone:
                 state-action mean and variance critics of the normalized
                 discounted loss-return with the source's Bellman targets
                 (fitted by squared error), safety measure
                     Gamma(s, a) = Q_c(s, a) + k_alpha sqrt(V_c(s, a)),
                 constraint E[Gamma] <= d_ret, and the source's actor step
                 (minimize E_{a~pi} Gamma(s, a) with critics fixed) taken with
                 the likelihood-ratio estimator. See the class docstring.

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
class SACritic(torch.nn.Module):
    """State-action critic for the WCSAC Gaussian safety critic. With
    positive=True the output is softplus'd (a variance)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64,
                 n_layers: int = 2, positive: bool = False):
        super().__init__()
        from .networks import _mlp
        self.net = _mlp(state_dim + action_dim, 1, hidden, n_layers)
        self.positive = positive

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
        return F.softplus(out) if self.positive else out


def wcsac_targets(c, q_sa_tgt, q_next, v_next, not_done, gamma):
    """Bellman targets of WCSAC (Yang et al. 2023, Eqs. (14)-(15)).

    c         per-step cost, shape (n,)
    q_sa_tgt  target-network estimate of Q_c(s, a), shape (n,)
    q_next    Q_c(s', a'_k) for K actions a'_k ~ pi(s'), shape (n, K)
    v_next    V_c(s', a'_k), shape (n, K)
    Returns (Q target, V target). The expectation over a' ~ pi is the sample
    mean over the K draws; s' is the sampled next state."""
    eq = q_next.mean(-1)
    eq2 = q_next.pow(2).mean(-1)
    ev = v_next.mean(-1)
    q_t = c + gamma * not_done * eq
    v_t = (c.pow(2) - q_sa_tgt.pow(2)
           + not_done * (2 * gamma * c * eq + gamma ** 2 * ev
                         + gamma ** 2 * eq2))
    # Not clamped: a single sampled target can be negative, and clamping it
    # would bias the regression (its conditional mean is the variance).
    return q_t, v_t


def wcsac_lambda_targets(c, q_sa, q_next, v_next, q_next_taken, not_done,
                         gamma, lam):
    """Lambda-return targets for the WCSAC Gaussian safety critic.

    Mean: G_t = c_t + gamma nd_t [(1-lam) E Q'_t + lam G_{t+1}], Eq. (14) of
    the source in lambda-return form.

    Variance, in the direct form of the source's Eq. (15): with exact critics
    Eq. (15) equals
        V_c(s,a) = E[delta^2] + gamma^2 E[Var(G' | s')],
        delta = c + gamma E_{a'} Q_c(s', a') - Q_c(s, a),
        Var(G' | s') = E_{a'} V_c(s', a') + Var_{a'} Q_c(s', a'),
    which is the same identity with exact critics. It avoids explicitly
    subtracting squared means (c^2 + 2 gamma c EQ' + gamma^2 E[V' + Q'^2] - Q^2),
    but its accuracy still depends on the mean critic: a state-dependent
    error in Q' enters delta^2 through a cross term that is first order in
    general. Its lambda-return is
        H_t = delta_t^2 + gamma^2 nd_t [(1-lam)(E V'_t + Var Q'_t)
              + lam (H_{t+1} + (Q_c(s_{t+1}, a_{t+1}) - E Q'_t)^2)],
    where the last square is an on-policy sample of Var_{a'} Q'; both squares
    are corrected for the sampling variance of the K-sample mean E Q'.
    The segment's last step bootstraps without the lam part.
    q_sa: target-network Q_c(s_t, a_t); q_next, v_next: shape (n, K) at K
    sampled next actions; q_next_taken: target Q_c at (s_{t+1}, a_{t+1}) for
    the next row's action (ignored at the segment's last step)."""
    K = q_next.shape[-1]
    eq = q_next.mean(-1)
    ev = v_next.mean(-1)
    varq = q_next.var(-1, unbiased=True) if K > 1 else torch.zeros_like(eq)
    delta = c + gamma * not_done * eq - q_sa
    n = len(c)
    G = torch.zeros_like(c)
    H = torch.zeros_like(c)
    nxt = None
    for t in reversed(range(n)):
        nd = not_done[t]
        boot = ev[t] + varq[t]
        # E Q' is a K-sample mean, so delta^2 and (Q_taken - EQ')^2 each carry
        # an extra Var_{a'}Q'/K in expectation; subtract its unbiased estimate.
        corr = varq[t] / K
        if nxt is None:
            g = c[t] + gamma * nd * eq[t]
            h = delta[t] ** 2 - gamma ** 2 * nd * corr + gamma ** 2 * nd * boot
        else:
            g = c[t] + gamma * nd * ((1 - lam) * eq[t] + lam * nxt[0])
            h = delta[t] ** 2 - gamma ** 2 * nd * corr + gamma ** 2 * nd * (
                (1 - lam) * boot
                + lam * (nxt[1] + (q_next_taken[t] - eq[t]) ** 2 - corr))
        G[t], H[t] = g, h
        # An episode end at step t is handled by nd_t = 0 above.
        nxt = (g, h)
    return G, H


class WCSAC_GS(SafeRLController):
    """WCSAC with a Gaussian safety critic (Yang et al., Mach. Learn. 2023),
    on the PPO backbone of the other methods.

    Safety critic: state-action networks Q_c(s, a) and V_c(s, a) for the mean
    and variance of the normalized discounted loss-return, trained with the
    Bellman targets of Eqs. (14)-(15) of the source, the variance in its
    equivalent direct form, as lambda-returns (wcsac_lambda_targets, lambda =
    GAE lambda, as for the other critics), fitted by squared error; the target
    networks take a Polyak step after every gradient step, as in SAC
    (checked in sim/check_wcsac.py). Safety measure Gamma(s, a) = Q_c + k_alpha sqrt(V_c), its
    Eq. (13).

    Actor: WCSAC's actor step minimizes E_{a ~ pi}[kappa Gamma(s, a)] at the
    visited states with the critics held fixed (a policy-improvement step
    against the safety critic, not the gradient of a return statistic). SAC
    takes it by reparameterization; here it is taken with the likelihood-
    ratio estimator, advantage Gamma(s, a) - mean_k Gamma(s, a_k), a_k ~ pi,
    inside the PPO objective. The multiplier ascends E[Gamma(s, a)] - d over
    the rollout, as in the source's dual step. The critic input is the
    proposed action, so with the safety filter the filter is part of the
    environment, as for the proposed method.

    The R2 draft instead used state critics and a local-variance advantage
    that it described as the likelihood-ratio gradient of J + k sqrt(M); that
    was not a gradient of that statistic (it can have the wrong sign) and is
    replaced by this class.
    """

    K_NEXT = 4
    K_BASE = 8

    def __init__(self, cfg: SimCfg, seed: int, d_ret: float = None, **kw):
        cfg = _clone_with(cfg, use_pid_dual=False, two_critics=True)
        super().__init__(cfg, seed=seed, use_safety_filter=False,
                         enforce_risk=False, **kw)
        a = cfg.algo
        sd, ad = cfg.state_dim, cfg.action_dim
        mk = lambda pos: SACritic(sd, ad, a.hidden, a.n_layers, pos).to(self.device)
        self.q_c, self.q_c_t = mk(False), mk(False)
        self.v_c, self.v_c_t = mk(True), mk(True)
        self.q_c_t.load_state_dict(self.q_c.state_dict())
        self.v_c_t.load_state_dict(self.v_c.state_dict())
        self.opt_q_c = torch.optim.Adam(self.q_c.parameters(), lr=a.lr_critic)
        self.opt_v_c = torch.optim.Adam(self.v_c.parameters(), lr=a.lr_critic)
        self.k_alpha = gaussian_cvar_coeff(1.0 - a.beta)
        self.d = float(d_ret if d_ret is not None else a.Gamma)
        self.last_gamma_pi = 0.0

    def risk_cost(self, loss, g_tau):
        return (1.0 - self.cfg.algo.gamma_disc) * loss

    def _proposed(self, raws):
        B = self.cfg.action_dim
        return self.actor._compose(raws[..., :B], raws[..., B:])

    fixed_action = None     # set to evaluate a fixed policy (calibration)

    def _sample_actions(self, states, K):
        with torch.no_grad():
            s_rep = states.unsqueeze(1).expand(-1, K, -1).reshape(-1, states.shape[-1])
            if self.fixed_action is not None:
                a = self.fixed_action.to(s_rep).expand(s_rep.shape[0], -1)
            else:
                a, _, _ = self.actor.sample(s_rep)
        return s_rep, a

    def gamma_sa(self, s, a, target=False):
        q = (self.q_c_t if target else self.q_c)(s, a)
        v = (self.v_c_t if target else self.v_c)(s, a)
        return q + self.k_alpha * v.clamp(min=1e-12).sqrt()

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
        a_prop = (torch.from_numpy(batch["a_prop"]).to(dev)
                  if "a_prop" in batch else self._proposed(raws))
        n = len(states)
        nd = 1.0 - dones

        with torch.no_grad():
            V_now = self.critic(states)
            d = c_e + g * nd * self.critic_target(next_states) - V_now
            adv_e = torch.zeros_like(d)
            last = 0.0
            for k in reversed(range(n)):
                last = d[k] + g * lam_gae * nd[k] * last
                adv_e[k] = last
            ret_e = adv_e + V_now
            # Safety-critic targets, Eqs. (14)-(15), with target networks.
            sn, an = self._sample_actions(next_states, self.K_NEXT)
            q_next = self.q_c_t(sn, an).view(n, self.K_NEXT)
            v_next = self.v_c_t(sn, an).view(n, self.K_NEXT)
            q_sa = self.q_c_t(states, a_prop)
            # Q_c at the next row's (state, proposed action); the last row's
            # value is unused (bootstrap only).
            q_taken = torch.cat([q_sa[1:], q_sa[-1:]])
            q_t, v_t = wcsac_lambda_targets(c_j, q_sa, q_next, v_next,
                                            q_taken, nd, g, lam_gae)
            # Safety measure at the taken action and its per-state baseline.
            gam = self.gamma_sa(states, a_prop)
            sb, ab = self._sample_actions(states, self.K_BASE)
            base = self.gamma_sa(sb, ab).view(n, self.K_BASE).mean(-1)
            adv_risk = gam - base
            self.last_gamma_pi = float(gam.mean())

        if self.t_slot >= cfg.risk_warmup_slots and not critics_only:
            step = cfg.lr_dual * cfg.rollout_slots
            self.lam = float(np.clip(
                self.lam + step * (self.last_gamma_pi - self.d),
                0.0, cfg.lam_max))
        lam = torch.full_like(adv_e, self.lam)
        w = lam / (1.0 + lam)
        with torch.no_grad():
            advs = _std((1.0 - w) * _std(adv_e) + w * _std(adv_risk))

        bs = max(1, n // cfg.n_mini)
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                if not critics_only:
                    lp_new = self.actor.log_prob(states[idx], actions[idx],
                                                 raws[idx])
                    ratio = (lp_new - lp_old[idx]).exp()
                    a_b = advs[idx]
                    actor_loss = torch.max(
                        ratio * a_b,
                        torch.clamp(ratio, 1 - cfg.ppo_clip,
                                    1 + cfg.ppo_clip) * a_b).mean()
                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.opt_actor.step()
                self.opt_critic.zero_grad()
                F.mse_loss(self.critic(states[idx]), ret_e[idx]).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_critic.step()
                self.opt_q_c.zero_grad()
                F.mse_loss(self.q_c(states[idx], a_prop[idx]), q_t[idx]).backward()
                torch.nn.utils.clip_grad_norm_(self.q_c.parameters(), 1.0)
                self.opt_q_c.step()
                self.opt_v_c.zero_grad()
                F.mse_loss(self.v_c(states[idx], a_prop[idx]),
                           v_t[idx]).backward()
                torch.nn.utils.clip_grad_norm_(self.v_c.parameters(), 1.0)
                self.opt_v_c.step()
                # Safety-critic targets follow SAC/WCSAC: a Polyak step after
                # every gradient step, not once per rollout.
                with torch.no_grad():
                    for net, tgt in ((self.q_c, self.q_c_t), (self.v_c, self.v_c_t)):
                        for p_, p_t in zip(net.parameters(), tgt.parameters()):
                            p_t.data.mul_(1 - cfg.target_tau)
                            p_t.data.add_(p_.data * cfg.target_tau)

        with torch.no_grad():
            for net, tgt in ((self.critic, self.critic_target),):
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
