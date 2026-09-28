"""
baselines.py
------------
Three baselines for comparison against the proposed Safe-RL controller:

  1. UnconstrainedPPO  -- same actor-critic, but no safety filter and no
                          dual / risk virtual queue. Pure energy-minimizing
                          PPO. Expected to violate the CVaR constraint
                          during training and at convergence.
  2. LyapunovOnly      -- model-based Lyapunov drift-plus-penalty controller
                          using the conservative service-rate estimate.
                          Closed-form per-slot decision, no learning.
                          Equivalent to "no learning" lower bound.
  3. ThresholdHeuristic-- simple threshold rule: phi=1 if q > q_hi, phi=0
                          if q < q_lo (with hysteresis). Industry-style
                          baseline.
"""
from __future__ import annotations
import itertools
import numpy as np
from .config import SimCfg
from .env import CellularEnv
from .algorithm import SafeRLController


class LyapunovOnly:
    """Per-slot model-based drift-plus-penalty controller (no learning).

    Decompose per-cell:
        argmin_{phi in [0,1]}  V * P_b(phi) - q_b * mu_b(phi)
        where P_b ~ p_on + p_dyn*phi and mu_b ~ mu_max * phi.
    Setting derivative to zero gives the threshold rule:
        serve at full (phi=1) if q_b * mu_max > V * p_dyn,
        sleep (phi=0)         if q_b * mu_max < V * p_dyn.
    """
    name = "LyapunovOnly"

    def __init__(self, cfg: SimCfg, V: float = 0.001, **kwargs):
        self.cfg = cfg
        self.V = V

    def reset(self, seed: int = 0):
        pass

    def act(self, state: np.ndarray, q_phys: np.ndarray,
            stochastic: bool = True):
        cfg = self.cfg
        mu_max_per_slot = cfg.chan.mu_max_mbps * cfg.dt_s
        threshold_Mb = self.V * cfg.energy.p_dyn_W / max(mu_max_per_slot, 1e-3)
        # Binary phi: 1 if backlog above threshold, else 0
        phi = (q_phys > threshold_Mb).astype(np.float32)
        return phi, np.zeros_like(phi), state, 0.0

    def update_tau_z_lambda(self, loss: float):
        return 0.0

    def update_actor_critic(self, batch):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "lambda": 0.0,
                "tau": 0.0, "z": 0.0}

    def collect_rollout(self, env: CellularEnv, n_slots: int):
        return _model_based_rollout(self, env, n_slots)


def _model_based_rollout(ctl, env, n_slots: int):
    """Generic rollout helper for non-learning controllers (LyapunovOnly,
    Threshold). The controller's act() is called every slot; no parameters
    are updated."""
    cfg = ctl.cfg
    # Non-learning controllers consume the raw environment observation.
    states = np.zeros((n_slots, cfg.env_state_dim), dtype=np.float32)
    actions = np.zeros((n_slots, cfg.action_dim), dtype=np.float32)
    energy = np.zeros(n_slots, dtype=np.float32)
    loss = np.zeros(n_slots, dtype=np.float32)
    viol = np.zeros(n_slots, dtype=np.float32)
    backlog = np.zeros(n_slots, dtype=np.float32)
    arrivals = np.zeros(n_slots, dtype=np.float32)
    toggles = 0
    s = env.state()
    for k in range(n_slots):
        a, _, _, _ = ctl.act(s, env.q)
        s_next, cost, done = env.step(a)
        states[k] = s
        actions[k] = a
        energy[k] = cost["energy_W"]
        loss[k] = cost["loss"]
        viol[k] = 1.0 if cost["loss"] > cfg.algo.Gamma else 0.0
        backlog[k] = cost["q_total_Mb"]
        arrivals[k] = cost["arrived_Mb"]
        toggles += cost["n_toggles"]
        s = env.reset() if done else s_next
    return {"states": states, "actions": actions, "energy": energy,
            "loss": loss, "viol": viol, "backlog": backlog,
            "arrivals": arrivals, "toggles": toggles}


class ThresholdHeuristic:
    """Hysteretic threshold: wake all cells at full phi if any q > q_hi,
    sleep when q < q_lo. Buffer thresholds tuned to be vendor-style."""
    name = "Threshold"

    def __init__(self, cfg: SimCfg, q_lo_Mb: float = 1.0, q_hi_Mb: float = 5.0,
                 **kwargs):
        self.cfg = cfg
        self.q_lo = q_lo_Mb
        self.q_hi = q_hi_Mb
        self._mode = np.ones(cfg.topo.B, dtype=np.float32)  # 1=awake

    def reset(self, seed: int = 0):
        self._mode = np.ones(self.cfg.topo.B, dtype=np.float32)

    def act(self, state: np.ndarray, q_phys: np.ndarray,
            stochastic: bool = True):
        # Hysteresis: if any cell crosses q_hi, all wake; if all below q_lo, all sleep.
        for b in range(self.cfg.topo.B):
            if q_phys[b] > self.q_hi:
                self._mode[b] = 1.0
            elif q_phys[b] < self.q_lo:
                self._mode[b] = 0.0
        return self._mode.copy(), np.zeros_like(self._mode), state, 0.0

    def update_tau_z_lambda(self, loss: float):
        return 0.0

    def update_actor_critic(self, batch):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "lambda": 0.0,
                "tau": 0.0, "z": 0.0}

    def collect_rollout(self, env: CellularEnv, n_slots: int):
        return _model_based_rollout(self, env, n_slots)


def run_episode(ctl, env: CellularEnv, cfg: SimCfg, seed: int,
                stochastic: bool = False, keep_series: bool = False) -> dict:
    """Evaluate any controller for one full episode.

    Shared by every experiment so that learned and non-learning controllers
    are measured through identical code. Handles the optional `bind`/`observe`
    hooks that the LCB safety filter needs for its causal estimates. With
    keep_series the per-slot loss, power and backlog are returned too, so any
    summary based on those three series can be recomputed later without
    rerunning (delay-proxy, toggle, share and filter statistics need a
    replay).
    """
    from .metrics import (summarize_episode, empirical_cvar,
                          empirical_cvar_inclusive)
    from .safety_filter import new_filter_stats
    s = env.reset(seed=seed)
    if hasattr(ctl, "bind"):
        ctl.bind(env)
    if hasattr(ctl, "reset"):
        ctl.reset(seed)
    if hasattr(ctl, "filter_stats"):
        ctl.filter_stats = new_filter_stats()
    P, L, Q, A, AW, SH = [], [], [], [], [], []
    tog = 0
    off = strand = offered = 0.0
    done = False
    while not done:
        a, _, _, _ = ctl.act(s, env.q, stochastic=stochastic)
        s, c, done = env.step(a)
        if hasattr(ctl, "observe"):
            ctl.observe(c)
        P.append(c["energy_W"]); L.append(c["loss"])
        Q.append(c["q_total_Mb"]); A.append(c["arrived_Mb"])
        AW.append(c["n_awake"]); SH.append(c["share_exec"])
        tog += c["n_toggles"]
        off += c.get("offloaded_Mb", 0.0); strand += c.get("stranded_Mb", 0.0)
        offered += c.get("offered_Mb", 0.0)
    L = np.array(L)
    m = summarize_episode(np.array(P), L, np.array(Q), np.array(A),
                          toggles=tog, beta=cfg.algo.beta,
                          Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s)
    m["cvar_beta_inclusive_old"] = empirical_cvar_inclusive(L, cfg.algo.beta)
    m["cvar_95"] = empirical_cvar(L, 0.95)
    m["mean_loss"] = float(L.mean())
    m["awake_mean"] = float(np.mean(AW))
    m["share_exec_mean"] = float(np.mean(SH))
    # Percentages of the nominal offered load (not of the inflated effective
    # workload, which was the denominator up to R2).
    m["offload_pct"] = 100.0 * off / max(offered, 1e-9)
    m["stranded_pct"] = 100.0 * strand / max(offered, 1e-9)
    fs = getattr(ctl, "filter_stats", None)
    if fs:
        m["filter"] = dict(fs)
    if keep_series:
        m["_series"] = dict(loss=L.astype(np.float32),
                            power=np.array(P, dtype=np.float32),
                            backlog=np.array(Q, dtype=np.float32))
    return m


class AlwaysOn:
    """All cells awake at full resource share.

    The maximum-service, maximum-energy reference. It is also the CVaR-optimal
    policy in this model, so it fixes the right-hand end of the feasible range
    for the risk budget Gamma: no policy can achieve a smaller CVaR, hence a
    budget below its CVaR is infeasible by construction.
    """
    name = "AlwaysOn"

    def __init__(self, cfg: SimCfg, **kwargs):
        self.cfg = cfg

    def reset(self, seed: int = 0):
        pass

    def act(self, state, q_phys, stochastic: bool = True):
        a = np.ones(self.cfg.topo.B, dtype=np.float32)
        return a, np.zeros_like(a), state, 0.0

    def update_tau_z_lambda(self, loss: float):
        return 0.0

    def update_actor_critic(self, batch):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "lambda": 0.0,
                "tau": 0.0, "z": 0.0}

    def collect_rollout(self, env: CellularEnv, n_slots: int):
        return _model_based_rollout(self, env, n_slots)


class DriftPlusPenalty:
    """Backlog-proportional share rule (no learning), rho = `slack`.

    Per slot each cell requests the share proportional to its own backlog,
    phi_b = clip(rho * q_b / mu_cap, phi_min, 1), and never sleeps. This is a
    heuristic, NOT a drift-plus-penalty minimizer: with affine power and
    service the per-slot DPP objective is affine in phi and is minimized at an
    endpoint (see TextbookDPP). Up to R2 it was mislabeled as the interior
    drift-plus-penalty solution. The class name is kept for compatibility.
    """
    name = "DriftPlusPenalty"

    def __init__(self, cfg: SimCfg, slack: float = 1.0, phi_min: float = 0.05,
                 **kwargs):
        self.cfg = cfg
        self.slack = slack
        self.phi_min = phi_min

    def reset(self, seed: int = 0):
        pass

    def act(self, state, q_phys, stochastic: bool = True):
        mu_cap = self.cfg.chan.mu_max_mbps * self.cfg.dt_s
        a = np.clip(self.slack * q_phys / max(mu_cap, 1e-9),
                    self.phi_min, 1.0).astype(np.float32)
        return a, np.zeros_like(a), state, 0.0

    def update_tau_z_lambda(self, loss: float):
        return 0.0

    def update_actor_critic(self, batch):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "lambda": 0.0,
                "tau": 0.0, "z": 0.0}

    def collect_rollout(self, env: CellularEnv, n_slots: int):
        return _model_based_rollout(self, env, n_slots)


class SleepAwareDrift(DriftPlusPenalty):
    """Drift-plus-penalty with a per-cell sleep threshold.

    Extends DriftPlusPenalty by putting a cell to sleep when its backlog falls
    below `q_sleep`, letting neighbouring cells absorb the load through the
    coverage overlap. This is the strongest non-learning controller in the
    comparison and the one a learned policy has to beat.
    """
    name = "SleepAwareDrift"

    def __init__(self, cfg: SimCfg, q_sleep_Mb: float = 0.05, **kwargs):
        super().__init__(cfg, **kwargs)
        self.q_sleep = q_sleep_Mb

    def act(self, state, q_phys, stochastic: bool = True):
        a, r, s, lp = super().act(state, q_phys, stochastic)
        a = np.where(q_phys < self.q_sleep, 0.0, a).astype(np.float32)
        return a, r, s, lp


class TextbookDPP:
    """Per-slot drift-plus-penalty minimizer over the joint action (no learning).

    Each slot it minimizes the drift-plus-penalty expression
        V * P(a) + sum_b Q_b * (E[A_b | a] - E[mu_b | a])
    over every sleep pattern the hysteresis admits in that slot (cells still
    inside their minimum dwell keep their state) and, for each awake cell, the
    share phi in {phi_lo, 1}, where phi_lo = 0 for a cell locked awake (its
    sleep request would be refused and the zero share executed) and phi_min
    otherwise. The expected effective arrival E[A_b | a] passes
    a causal EWMA of each cell's nominal load (weight 0.01, as in the safety
    filter) through the offloading map of the environment for the pattern, so
    offloading onto awake neighbors is anticipated. Power includes the
    switching penalty for every toggle and wake-up service is halved, as in the
    environment; E[mu] uses the mean channel multiplier. Given a pattern the
    expression is affine in each awake cell's share, so the share optimum is an
    endpoint and the joint minimum is exact over the admissible set.

    Up to the R2 draft this class decided each cell separately and dropped the
    action dependence of A_b; that per-cell rule did not minimize the
    expression in the offloading model and is replaced by this one.
    """
    name = "TextbookDPP"
    EWMA = 0.01

    def __init__(self, cfg: SimCfg, V: float = 1e-3, **kwargs):
        self.cfg = cfg
        self.V = V
        self._m = 1.0
        self._env = None
        self._a_hat = None
        self._t_seen = -1
        K = cfg.topo.B
        self._pats = np.array(list(itertools.product([0, 1], repeat=K)),
                              dtype=np.int32)

    def bind(self, env):
        self._m = float(np.mean(env.channel_pool))
        self._env = env
        K = self.cfg.topo.B
        # A_eff = M_s @ a for each pattern s (the offloading map is linear).
        eye = np.eye(K)
        self._M = np.stack([np.stack([env.redistribute(eye[j], s)[0]
                                      for j in range(K)], axis=1)
                            for s in self._pats]).astype(np.float64)
        self._a_hat = np.full(K, float(self.cfg.arr.base_rate_Mb_per_slot))
        self._t_seen = -1

    def reset(self, seed: int = 0):
        pass

    def _update_estimate(self):
        env = self._env
        t = int(env.t)
        if t > 0 and t - 1 != self._t_seen:
            self._a_hat += self.EWMA * (env.arrivals[:, t - 1] - self._a_hat)
            self._t_seen = t - 1

    def act(self, state, q_phys, stochastic: bool = True):
        c, e = self.cfg, self.cfg.energy
        env = self._env
        self._update_estimate()
        mu_cap = c.chan.mu_max_mbps * c.dt_s
        mu_floor = c.chan.mu_min_mbps * c.dt_s
        q = np.asarray(q_phys, dtype=np.float64)
        s_prev = env.s_prev.astype(np.int32)
        min_d = np.where(s_prev == 1, e.min_on_slots, e.min_off_slots)
        locked = env.dwell < min_d
        pats = self._pats
        ok = np.all(~locked[None, :] | (pats == s_prev[None, :]), axis=1)
        pats, M = pats[ok], self._M[ok]
        A = M @ self._a_hat                                # (P, K)
        wake = (pats == 1) & (s_prev[None, :] == 0)
        wf = np.where(wake, getattr(e, "wake_service_frac", 1.0), 1.0)
        # Best share per awake cell: endpoint of an affine function. The
        # lower endpoint is 0 for a cell locked awake by its minimum on-dwell
        # (a sleep request is refused and the zero share is executed), and
        # EPS_SLEEP_DPP otherwise (a smaller share would request sleep).
        lo = np.where((s_prev == 1) & locked, 0.0, EPS_SLEEP_DPP)
        gain = q[None, :] * wf * (mu_cap - mu_floor) * self._m   # per unit phi
        phi = np.where(gain > self.V * e.p_dyn_W, 1.0, lo[None, :])
        mu = wf * (mu_floor + (mu_cap - mu_floor) * phi) * self._m
        P = np.where(pats == 1, e.p_on_W + e.p_dyn_W * phi, e.p_slp_W).sum(1)
        P = P + e.p_sw_W * (pats != s_prev[None, :]).sum(1)
        obj = (self.V * P + (q[None, :] * A).sum(1)
               - (q[None, :] * mu * (pats == 1)).sum(1))
        k = int(np.argmin(obj))
        a = np.where(pats[k] == 1, phi[k], 0.0).astype(np.float32)
        return a, np.zeros_like(a), state, 0.0

    def update_tau_z_lambda(self, loss: float):
        return 0.0

    def update_actor_critic(self, batch):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "lambda": 0.0,
                "tau": 0.0, "z": 0.0}

    def collect_rollout(self, env: CellularEnv, n_slots: int):
        return _model_based_rollout(self, env, n_slots)


EPS_SLEEP_DPP = 0.05


def make_baseline(name: str, cfg: SimCfg, seed: int) -> object:
    """Factory."""
    if name == "SafeRL":
        return SafeRLController(cfg, seed=seed,
                                 use_safety_filter=True,
                                 enforce_risk=True)
    if name == "UnconstrainedPPO":
        return SafeRLController(cfg, seed=seed,
                                 use_safety_filter=False,
                                 enforce_risk=False)
    if name == "LyapunovOnly":
        return LyapunovOnly(cfg)
    if name == "Threshold":
        return ThresholdHeuristic(cfg)
    if name == "AlwaysOn":
        return AlwaysOn(cfg)
    if name.startswith("DriftPlusPenalty"):
        # "DriftPlusPenalty" or "DriftPlusPenalty:<slack>"
        slack = float(name.split(":")[1]) if ":" in name else 1.0
        return DriftPlusPenalty(cfg, slack=slack)
    if name.startswith("TextbookDPP"):
        V = float(name.split(":")[1]) if ":" in name else 1e-3
        return TextbookDPP(cfg, V=V)
    if name.startswith("SleepAwareDrift"):
        # "SleepAwareDrift" or "SleepAwareDrift:<q_sleep_Mb>"
        qs = float(name.split(":")[1]) if ":" in name else 0.05
        return SleepAwareDrift(cfg, q_sleep_Mb=qs)
    raise ValueError(f"unknown baseline: {name}")
