"""
env.py
------
Gym-style 7-cell O-RAN environment for safe-RL training and evaluation.

State (dim = 2B + 2):
    [q_b/Q_scale for b in 1..B]      # normalized backlog per cell
    [mu_b/mu_max for b in 1..B]      # last-slot service rate per cell
    [cos(2*pi*t/T_day), sin(...)]    # time-of-day cyclic feature

Action (dim = B): per-cell resource share phi_b in [0, 1] (continuous).
Cells with phi_b < eps_sleep are taken to be in deep sleep this slot.

Per-slot dynamics:
    1. arrivals[b] arrive (Mbit) -- supplied externally by arrivals module.
    2. predicted service mu_hat[b] = phi_b * mu_max * channel_realization[b];
       LCB version subtracts kappa * sigma.
    3. served = min(q_b, mu[b]); q_{b}(t+1) = q_b - mu[b] + a_b.
    4. energy P_b based on sleep/awake + phi_b.
    5. loss l(t) = sum_b ([q_b - mu_b]_+ + a_b) / a_bar_b / B  (normalized)
       capped at ell_max.
    6. return cost dict {energy, loss}; reward = -energy (the dual + risk
       virtual queue handle the constraint cost separately in algorithm.py).

Independence checks vs Paper 3:
    - No ISAC posterior anywhere in the state.
    - No Wasserstein ambiguity radius input.
    - Action is continuous resource fraction (not discrete codebook).
    - Loss normalization is per-cell mean-arrival, normalized by B for
      [0, ell_max] range; Paper 3 used summed-backlog over mean-arrival.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from .config import SimCfg
from .channel_lumos5g import load_channel_multipliers


Q_SCALE_Mb = 50.0  # normalization for q (Mbit)
EPS_SLEEP = 0.05   # phi < this -> deep sleep


class CellularEnv:
    """Minimal gym-style env. Single-process, NumPy-only inside step()."""

    def __init__(self, cfg: SimCfg, arrivals: np.ndarray, seed: int = 0):
        """
        arrivals: shape (B, T) -- pre-generated per-cell per-slot arrivals
                  (Mbit). The env consumes one slot per step().
        """
        self.cfg = cfg
        self.B = cfg.topo.B
        self.T = arrivals.shape[1]
        assert arrivals.shape[0] == self.B
        self.arrivals = arrivals
        self.rng = np.random.default_rng(seed)
        # Channel-multiplier pool from Lumos5G (or synthetic fallback)
        self.channel_pool, self.channel_source = load_channel_multipliers(cfg)
        self._reset_state()

    def _reset_state(self):
        self.t = 0
        self.q = np.zeros(self.B, dtype=np.float32)        # backlog (Mbit)
        self.s_prev = np.ones(self.B, dtype=np.int32)      # 1 = awake
        self.mu_last = np.zeros(self.B, dtype=np.float32)  # last service rate
        self.a_bar = self.arrivals.mean(axis=1) + 1e-6
        self.dwell = np.full(self.B, self.cfg.energy.min_on_slots,
                             dtype=np.int32)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self.state()

    def state(self) -> np.ndarray:
        slots_per_day = max(1, int(round(86400.0 / self.cfg.dt_s)))
        phi_t = 2 * np.pi * (self.t % slots_per_day) / slots_per_day
        feat = np.concatenate([
            self.q / Q_SCALE_Mb,
            self.mu_last / max(self.cfg.chan.mu_max_mbps * self.cfg.dt_s, 1e-6),
            np.array([np.cos(phi_t), np.sin(phi_t)], dtype=np.float32),
        ])
        return feat.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict, bool]:
        """One slot. Returns (next_state, cost_dict, done)."""
        phi = np.clip(action, 0.0, 1.0).astype(np.float32)
        # sleep state via threshold + min-dwell hysteresis
        want_sleep = phi < EPS_SLEEP
        s_t = self.s_prev.copy()
        for b in range(self.B):
            min_dwell = (self.cfg.energy.min_on_slots if s_t[b] == 1
                         else self.cfg.energy.min_off_slots)
            if self.dwell[b] < min_dwell:
                self.dwell[b] += 1
                continue
            new_s = 0 if want_sleep[b] else 1
            if new_s != s_t[b]:
                s_t[b] = new_s
                self.dwell[b] = 0
            else:
                self.dwell[b] += 1

        # Channel realization: draw from Lumos5G-derived multiplier pool
        # (or synthetic log-normal fallback if dataset absent).
        mu_jitter = self.channel_pool[
            self.rng.integers(0, len(self.channel_pool), size=self.B)
        ].astype(np.float32)
        mu_cap = self.cfg.chan.mu_max_mbps * self.cfg.dt_s  # Mbit/slot at full phi
        mu_floor = self.cfg.chan.mu_min_mbps * self.cfg.dt_s
        mu_b = s_t * (mu_floor + (mu_cap - mu_floor) * phi) * mu_jitter

        # Queue update
        a_t = self.arrivals[:, self.t]
        served = np.minimum(self.q, mu_b)
        q_next = np.maximum(self.q - mu_b, 0.0) + a_t

        # Energy
        e = self.cfg.energy
        P = np.where(
            s_t == 0,
            e.p_slp_W,
            e.p_on_W + e.p_dyn_W * phi,
        ).astype(np.float32)
        # Switching penalty
        toggled = (s_t != self.s_prev).astype(np.float32)
        P_total = float(P.sum() + e.p_sw_W * toggled.sum())

        # Per-slot QoS loss (normalized, capped). Paper-4 specific form:
        #   l(t) = (1/B) sum_b ([q_b - mu_b]_+ + a_b) / a_bar_b
        # which is bounded in [0, ell_max].
        per_cell = (np.maximum(self.q - mu_b, 0.0) + a_t) / self.a_bar
        loss = float(min(per_cell.mean(), self.cfg.algo.ell_max))

        # Bookkeeping
        self.q = q_next
        self.s_prev = s_t
        self.mu_last = mu_b
        self.t += 1
        done = self.t >= self.T

        cost = {
            "energy_W": P_total,
            "loss": loss,
            "served_Mb": float(served.sum()),
            "arrived_Mb": float(a_t.sum()),
            "q_total_Mb": float(self.q.sum()),
            "n_awake": int(s_t.sum()),
            "n_toggles": int(toggled.sum()),
        }
        return self.state(), cost, done
