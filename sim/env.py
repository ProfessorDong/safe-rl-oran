"""
env.py
------
Gym-style 7-cell O-RAN environment for safe-RL training and evaluation.

State (dim = 4B + 2 with sleep_state_in_obs, else 2B + 2):
    [q_b/Q_scale for b in 1..B]      # normalized backlog per cell
    [mu_b/mu_max for b in 1..B]      # last-slot service rate per cell
    [s_prev_b for b in 1..B]         # sleep indicator (1 = awake)
    [dwell_b/max_dwell for b in 1..B]# min-dwell counter (hysteresis phase)
    [cos(2*pi*t/T_day), sin(...)]    # time-of-day cyclic feature

The controller appends its own (lambda, tau) on top of this (see
algorithm.py), giving the network input of cfg.state_dim.

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


def build_coverage(B: int) -> np.ndarray:
    """Row-stochastic neighbour-coverage matrix for a hexagonal cluster.

    Cell 0 is the centre site and cells 1..B-1 form the surrounding ring, so
    the centre neighbours every ring cell and each ring cell neighbours the
    centre plus its two ring-adjacent siblings. Row b gives the fraction of
    cell b's offloadable traffic that each neighbour would pick up if b went
    to sleep; rows sum to one (or to zero for an isolated cell).
    """
    cov = np.zeros((B, B), dtype=np.float64)
    if B == 1:
        return cov
    ring = list(range(1, B))
    n_ring = len(ring)
    for b in range(B):
        if b == 0:
            nbrs = ring
        else:
            i = ring.index(b)
            nbrs = [0, ring[(i - 1) % n_ring], ring[(i + 1) % n_ring]]
            nbrs = list(dict.fromkeys(n for n in nbrs if n != b))
        if nbrs:
            cov[b, nbrs] = 1.0 / len(nbrs)
    return cov


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
        self.cov = build_coverage(self.B)
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
        # Normalize the per-cell backlog by the ONE-SLOT full-service capacity,
        # not by the cluster-wide safety threshold Q_SCALE_Mb (= 50 Mb). A
        # per-cell backlog is ~1 Mb, so dividing by 50 delivered the single
        # most decision-relevant feature to the network as ~0.02, numerically
        # indistinguishable from zero, and the actor could not condition on
        # backlog at all. q_b / mu_cap is the natural scale: it is exactly the
        # number of full-service slots needed to drain the cell.
        mu_cap = max(self.cfg.chan.mu_max_mbps * self.cfg.dt_s, 1e-6)
        parts = [
            np.clip(self.q / mu_cap, 0.0, 20.0),
            self.mu_last / mu_cap,
        ]
        if self.cfg.algo.sleep_state_in_obs:
            # The sleep indicator and the min-dwell counter gate which
            # actions the hysteresis will actually honour this slot, so they
            # belong to the Markov state; the submitted version hid them.
            max_dwell = max(self.cfg.energy.min_on_slots,
                            self.cfg.energy.min_off_slots)
            parts += [
                self.s_prev.astype(np.float32),
                np.minimum(self.dwell, max_dwell).astype(np.float32) / max_dwell,
            ]
        parts.append(np.array([np.cos(phi_t), np.sin(phi_t)], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _redistribute(self, a_raw: np.ndarray, s_t: np.ndarray):
        """Route a sleeping cell's offered traffic onto awake neighbours.

        Returns (effective per-cell arrivals, offloaded Mb, stranded Mb). A
        fraction `overlap_fraction` of a sleeping cell's load lies in the
        coverage overlap and is shared among its awake neighbours in the
        proportions of `self.cov`, inflated by 1/`offload_efficiency` because
        a UE served by a farther cell consumes more resources. The remainder,
        plus everything belonging to a sleeping cell with no awake neighbour,
        stays queued where it arrived.
        """
        topo = self.cfg.topo
        if not getattr(topo, "enable_offload", False):
            return a_raw.astype(np.float32), 0.0, 0.0

        awake = (s_t == 1).astype(np.float64)
        asleep = 1.0 - awake
        a = a_raw.astype(np.float64)

        # Awake cells keep their own traffic.
        a_eff = a * awake

        # Share of each sleeping cell's offloadable load reaching each awake
        # neighbour, renormalized over the neighbours that are actually awake.
        share = self.cov * awake[None, :]                 # (B, B)
        reach = share.sum(axis=1)                          # (B,)
        covered = asleep * (reach > 0)
        frac = float(topo.overlap_fraction)
        eta = max(float(topo.offload_efficiency), 1e-6)

        moved = a * covered * frac                         # Mb leaving sleepers
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(reach[:, None] > 0, share / reach[:, None], 0.0)
        a_eff += (moved[:, None] * weights).sum(axis=0) / eta

        # Not offloadable: outside the overlap, or no awake neighbour at all.
        stranded = a * asleep - moved
        a_eff += stranded
        return a_eff.astype(np.float32), float(moved.sum()), float(stranded.sum())

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
        e_wake_frac = getattr(self.cfg.energy, "wake_service_frac", 1.0)
        mu_b = s_t * (mu_floor + (mu_cap - mu_floor) * phi) * mu_jitter
        # A cell that wakes this slot spends the ASM activation slope unable
        # to carry data (Salem et al., VTC-Fall 2017), so it delivers only a
        # fraction of a slot's service. Without this the model let a waking
        # cell serve at full rate, making sleep-state chattering almost free.
        woke = ((s_t == 1) & (self.s_prev == 0))
        mu_b = mu_b * np.where(woke, e_wake_frac, 1.0)

        # Queue update. Traffic offered to a sleeping cell is picked up by its
        # awake neighbours (O-RAN Carrier and Cell Switch Off/On), at reduced
        # spectral efficiency; whatever cannot be covered is stranded and
        # stays queued at its own cell until it wakes.
        a_raw = self.arrivals[:, self.t]
        a_t, offloaded_Mb, stranded_Mb = self._redistribute(a_raw, s_t)
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
            "offloaded_Mb": offloaded_Mb,
            "stranded_Mb": stranded_Mb,
        }
        return self.state(), cost, done
