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
    """Continuous drift-plus-penalty controller (no learning).

    Per slot each cell requests exactly the share needed to drain its own
    backlog, phi_b = clip(slack * q_b / mu_cap, phi_min, 1). This is the
    interior solution of the per-cell drift-plus-penalty problem, as opposed
    to the binary bang-bang rule used as the Lyapunov baseline in the
    submitted version, which chattered at >5000 toggles/min because it could
    only choose between full service and sleep. `slack` trades tail risk
    against power and traces the controller's Pareto frontier.
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
    if name.startswith("SleepAwareDrift"):
        # "SleepAwareDrift" or "SleepAwareDrift:<q_sleep_Mb>"
        qs = float(name.split(":")[1]) if ":" in name else 0.05
        return SleepAwareDrift(cfg, q_sleep_Mb=qs)
    raise ValueError(f"unknown baseline: {name}")
