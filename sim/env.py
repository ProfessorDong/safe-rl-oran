"""
env.py
------
Gym-style K-cell O-RAN environment for safe-RL training and evaluation.

State (dim = 4K + 2 with sleep_state_in_obs, else 2K + 2):
    [q_b/mu_cap for b in 1..K]        # backlog in full-service slots
    [mu_b/mu_cap for b in 1..K]       # last-slot service
    [s_prev_b for b in 1..K]          # sleep indicator (1 = awake)
    [dwell_b/max_dwell for b in 1..K] # min-dwell counter (hysteresis phase)
    [cos(2*pi*t/T), sin(2*pi*t/T)]    # phase of the (compressed) daily cycle

Each episode compresses one 24-hour traffic profile onto its T slots, so the
phase feature is t/T. (Up to R2 the feature used the physical clock t*dt/86400,
which advanced by only 20 s or 100 s per episode and mapped the same observed
phase to traffic hours five times apart in training and evaluation.)

Action (dim = K): per-cell resource share phi_b in [0, 1]. A cell with
phi_b < EPS_SLEEP requests deep sleep; the request is honored subject to a
minimum-dwell hysteresis, except that a safety wake issued by the safety filter
(`force_awake`) overrides the minimum off-dwell.

Per-slot dynamics:
    1. sleep pattern s(t) from the request, hysteresis and safety wakes;
    2. effective arrivals A_b(t) after offloading from sleeping cells;
    3. service mu_b = s_b (mu_floor + (mu_cap - mu_floor) phi_b) J_b(t), halved
       in a slot in which the cell wakes;
    4. Q_b(t+1) = [Q_b(t) - mu_b]_+ + A_b(t);
    5. power P = sum_b P_b(s_b, phi_b) + p_sw * #toggles  (watts);
    6. loss l(t) = min{ell_max, (1/K) sum_b Q_b(t+1) / abar_b}.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from scipy.special import ndtr, ndtri
from .config import SimCfg
from .channel_lumos5g import load_channel_multipliers


Q_SCALE_Mb = 50.0  # normalization for q (Mbit), aggregate-filter threshold scale
EPS_SLEEP = 0.05   # phi < this -> deep-sleep request

# Axial directions of a hexagonal grid.
_HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def _hex_cells(K: int):
    """Axial coordinates of a centered hexagonal cluster of K cells, or None
    if K is not a centered hexagonal number (1, 7, 19, 37, 61, ...)."""
    cells = [(0, 0)]
    r = 0
    while len(cells) < K:
        r += 1
        q, s = _HEX_DIRS[4][0] * r, _HEX_DIRS[4][1] * r
        for d in range(6):
            for _ in range(r):
                cells.append((q, s))
                q, s = q + _HEX_DIRS[d][0], s + _HEX_DIRS[d][1]
    return cells if len(cells) == K else None


def build_coverage(B: int) -> np.ndarray:
    """Row-stochastic neighbor-coverage matrix.

    For a centered hexagonal number of cells the cluster is a true hexagonal
    grid (cell 0 at the center, rings around it) and each cell's neighbors are
    its geometric hexagonal neighbors inside the cluster: interior cells have
    six, boundary cells fewer. For K = 7 this is the center plus a six-cell
    ring, with each ring cell adjacent to the center and to two ring cells.
    Up to R2 every K used a wheel (center adjacent to all K-1 others), which for
    K = 61 gave the center 60 neighbors. Any other K falls back to the wheel.
    Row b gives the fraction of cell b's offloadable traffic each neighbor picks
    up if b sleeps; rows sum to one (zero for an isolated cell).
    """
    cov = np.zeros((B, B), dtype=np.float64)
    if B == 1:
        return cov
    cells = _hex_cells(B)
    if cells is not None:
        index = {c: i for i, c in enumerate(cells)}
        for b, (q, s) in enumerate(cells):
            nbrs = [index[(q + dq, s + ds)] for dq, ds in _HEX_DIRS
                    if (q + dq, s + ds) in index]
            if nbrs:
                cov[b, nbrs] = 1.0 / len(nbrs)
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

    def __init__(self, cfg: SimCfg, arrivals: np.ndarray, seed: int = 0,
                 arrival_regen: Optional[Callable[[int], np.ndarray]] = None):
        """
        arrivals: (K, T) per-cell per-slot nominal offered load (Mbit).
        arrival_regen: optional k -> (K, T) generator. When given, every reset
            after the first draws a fresh arrival realization for episode k, so
            a training run does not replay one noise realization hundreds of
            times.
        """
        self.cfg = cfg
        self.B = cfg.topo.B
        self.T = arrivals.shape[1]
        assert arrivals.shape[0] == self.B
        self.arrivals = arrivals
        self._regen = arrival_regen
        self._episode = 0
        self.rng = np.random.default_rng(seed)
        self.channel_pool, self.channel_source = load_channel_multipliers(cfg)
        self._pool_sorted = np.sort(np.asarray(self.channel_pool,
                                               dtype=np.float32))
        self.cov = build_coverage(self.B)
        self._pending_force = None
        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.t = 0
        self.q = np.zeros(self.B, dtype=np.float32)        # backlog (Mbit)
        self.s_prev = np.ones(self.B, dtype=np.int32)      # 1 = awake
        self.mu_last = np.zeros(self.B, dtype=np.float32)  # last service
        # Loss normalizer: per-cell mean nominal load of this episode. This is
        # an offline metric normalization (it uses the whole episode) and is
        # not available to, nor used by, any controller.
        self.a_bar = self.arrivals.mean(axis=1) + 1e-6
        # Dwell counter = slots already spent in the current sleep state,
        # including the current one; a change is allowed once it reaches the
        # minimum dwell. Start "unlocked".
        self.dwell = np.full(self.B, max(self.cfg.energy.min_on_slots,
                                         self.cfg.energy.min_off_slots),
                             dtype=np.int32)
        # Correlated-channel state. The AR(1) latent starts in stationarity;
        # it is drawn only when the correlated channel is on, so the i.i.d.
        # channel consumes exactly the random stream it did before R2.
        ch = self.cfg.chan
        if (getattr(ch, "ar1_rho", 0.0) or getattr(ch, "spatial_rho", 0.0)
                or getattr(ch, "load_coupling", 0.0)):
            self._ar = self.rng.normal(0.0, 1.0, size=self.B)
        else:
            self._ar = np.zeros(self.B)
        # Standard-normal score of each slot's load within its own cell's
        # episode (rank transform), used by the load-coupled channel so that
        # the channel latent stays exactly N(0, 1).
        ranks = np.argsort(np.argsort(self.arrivals, axis=1), axis=1)
        self._load_z = ndtri((ranks + 0.5) / self.T)
        self._pending_force = None

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._regen is not None:
            self._episode += 1
            self.arrivals = self._regen(self._episode)
            self.T = self.arrivals.shape[1]
        self._reset_state()
        return self.state()

    def state(self) -> np.ndarray:
        phi_t = 2.0 * np.pi * (self.t % self.T) / self.T
        mu_cap = max(self.cfg.chan.mu_max_mbps * self.cfg.dt_s, 1e-6)
        parts = [
            np.clip(self.q / mu_cap, 0.0, 20.0),
            self.mu_last / mu_cap,
        ]
        if self.cfg.algo.sleep_state_in_obs:
            max_dwell = max(self.cfg.energy.min_on_slots,
                            self.cfg.energy.min_off_slots)
            parts += [
                self.s_prev.astype(np.float32),
                np.minimum(self.dwell, max_dwell).astype(np.float32) / max_dwell,
            ]
        parts.append(np.array([np.cos(phi_t), np.sin(phi_t)], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    def predict_sleep(self, action: np.ndarray,
                      force_awake: Optional[np.ndarray] = None,
                      commit: bool = False) -> np.ndarray:
        """Sleep pattern the hysteresis will execute for `action`.

        A cell requests sleep when phi < EPS_SLEEP. A request that would change
        the state is honored only once the cell has spent the minimum dwell in
        its current state. A safety wake (`force_awake`) wakes a sleeping cell
        immediately and keeps an awake cell awake, overriding the minimum
        off-dwell: the dwell is an anti-chatter policy, not a physical limit,
        and the physical activation cost is charged separately (half a slot of
        service and the switching penalty). With commit=False nothing is
        mutated, so the safety filter can call this to predict the pattern.
        """
        want_sleep = np.asarray(action) < EPS_SLEEP
        force = (np.zeros(self.B, dtype=bool) if force_awake is None
                 else np.asarray(force_awake, dtype=bool))
        s_t = self.s_prev.copy()
        dwell = self.dwell.copy()
        e = self.cfg.energy
        for b in range(self.B):
            if force[b]:
                if s_t[b] == 0:
                    s_t[b] = 1
                    dwell[b] = 1
                else:
                    dwell[b] += 1
                continue
            min_dwell = e.min_on_slots if s_t[b] == 1 else e.min_off_slots
            new_s = 0 if want_sleep[b] else 1
            if new_s != s_t[b] and dwell[b] >= min_dwell:
                s_t[b] = new_s
                dwell[b] = 1
            else:
                dwell[b] += 1
        if commit:
            self.dwell = dwell
        return s_t

    # ------------------------------------------------------------------
    def _draw_channel(self) -> np.ndarray:
        """Per-cell service-rate multiplier J_b(t).

        With all dependence knobs at zero this samples the Lumos5G pool
        independently per cell and slot. Otherwise a Gaussian copula imposes
        dependence while preserving the pool as the marginal: the latent
            z_b(t) = sqrt(1 - c^2) * x_b(t) - c * zeta_b(t)
        combines an AR(1) factor x (persistence rho, a fraction s of its
        innovation shared across the cluster, stationary N(0, 1)) with the
        standard-normal rank score zeta of the cell's own load, which is
        N(0, 1) over the episode and independent of x. Hence z is exactly
        N(0, 1) and Phi(z) is uniform, so reading off the empirical quantile
        leaves the Lumos5G marginal intact. (Up to R2 the load term entered
        unstandardized, which shifted the marginal: the pool mean moved from
        1.043 to 1.079.)
        """
        ch = self.cfg.chan
        rho = float(getattr(ch, "ar1_rho", 0.0))
        srho = float(getattr(ch, "spatial_rho", 0.0))
        c = float(getattr(ch, "load_coupling", 0.0))
        if rho == 0.0 and srho == 0.0 and c == 0.0:
            return self.channel_pool[
                self.rng.integers(0, len(self.channel_pool), size=self.B)
            ].astype(np.float32)
        common = self.rng.normal(0.0, 1.0)
        idio = self.rng.normal(0.0, 1.0, size=self.B)
        innov = np.sqrt(srho) * common + np.sqrt(max(1.0 - srho, 0.0)) * idio
        self._ar = rho * self._ar + np.sqrt(max(1.0 - rho * rho, 0.0)) * innov
        z = (np.sqrt(max(1.0 - c * c, 0.0)) * self._ar
             - c * self._load_z[:, self.t % self.T])
        u = ndtr(z)
        n = len(self._pool_sorted)
        idx = np.clip((u * n).astype(int), 0, n - 1)
        return self._pool_sorted[idx].astype(np.float32)

    # ------------------------------------------------------------------
    def redistribute(self, a_raw: np.ndarray, s_t: np.ndarray):
        """Effective per-cell arrivals under sleep pattern s_t.

        A fraction `overlap_fraction` of a sleeping cell's load lies in the
        coverage overlap and is shared among its awake neighbors in the
        proportions of `self.cov`, inflated by 1/`offload_efficiency`. The
        rest, and all load of a sleeping cell with no awake neighbor, stays
        queued where it arrived. Pure function of its inputs, so the safety
        filter can evaluate it for a predicted pattern.

        Returns (effective arrivals, offloaded Mb, stranded Mb).
        """
        topo = self.cfg.topo
        if not getattr(topo, "enable_offload", False):
            return np.asarray(a_raw, dtype=np.float32), 0.0, 0.0
        awake = (np.asarray(s_t) == 1).astype(np.float64)
        asleep = 1.0 - awake
        a = np.asarray(a_raw, dtype=np.float64)
        a_eff = a * awake
        share = self.cov * awake[None, :]
        reach = share.sum(axis=1)
        covered = asleep * (reach > 0)
        frac = float(topo.overlap_fraction)
        eta = max(float(topo.offload_efficiency), 1e-6)
        moved = a * covered * frac
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(reach[:, None] > 0,
                               share / np.where(reach[:, None] > 0,
                                                reach[:, None], 1.0), 0.0)
        a_eff = a_eff + (moved[:, None] * weights).sum(axis=0) / eta
        stranded = a * asleep - moved
        a_eff = a_eff + stranded
        return a_eff.astype(np.float32), float(moved.sum()), float(stranded.sum())

    # Backward-compatible alias.
    _redistribute = redistribute

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray,
             force_awake: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, Dict, bool]:
        """One slot. Returns (next_state, cost_dict, done).

        `force_awake` (or a mask left in `_pending_force` by the safety filter
        of the controller bound to this environment) marks safety wakes.
        """
        if force_awake is None:
            force_awake = self._pending_force
        self._pending_force = None
        phi = np.clip(action, 0.0, 1.0).astype(np.float32)
        s_t = self.predict_sleep(phi, force_awake=force_awake, commit=True)

        mu_jitter = self._draw_channel()
        mu_cap = self.cfg.chan.mu_max_mbps * self.cfg.dt_s
        mu_floor = self.cfg.chan.mu_min_mbps * self.cfg.dt_s
        e_wake_frac = getattr(self.cfg.energy, "wake_service_frac", 1.0)
        mu_b = s_t * (mu_floor + (mu_cap - mu_floor) * phi) * mu_jitter
        woke = (s_t == 1) & (self.s_prev == 0)
        mu_b = mu_b * np.where(woke, e_wake_frac, 1.0)

        a_raw = self.arrivals[:, self.t]
        a_t, offloaded_Mb, stranded_Mb = self.redistribute(a_raw, s_t)
        served = np.minimum(self.q, mu_b)
        q_next = np.maximum(self.q - mu_b, 0.0) + a_t

        e = self.cfg.energy
        P = np.where(s_t == 0, e.p_slp_W, e.p_on_W + e.p_dyn_W * phi
                     ).astype(np.float32)
        toggled = (s_t != self.s_prev).astype(np.float32)
        P_total = float(P.sum() + e.p_sw_W * toggled.sum())

        per_cell = q_next / self.a_bar
        loss = float(min(per_cell.mean(), self.cfg.algo.ell_max))

        self.q = q_next.astype(np.float32)
        self.s_prev = s_t
        self.mu_last = mu_b.astype(np.float32)
        self.t += 1
        done = self.t >= self.T

        cost = {
            "energy_W": P_total,
            "loss": loss,
            "served_Mb": float(served.sum()),
            "arrived_Mb": float(a_t.sum()),       # effective (after offload)
            "offered_Mb": float(a_raw.sum()),     # nominal offered load
            "q_total_Mb": float(self.q.sum()),
            "n_awake": int(s_t.sum()),
            "n_toggles": int(toggled.sum()),
            "share_exec": float(np.where(s_t == 1, phi, 0.0).mean()),
            "a_vec": a_t.astype(np.float32),
            "a_raw": np.asarray(a_raw, dtype=np.float32),
            "offloaded_Mb": offloaded_Mb,
            "stranded_Mb": stranded_Mb,
        }
        return self.state(), cost, done
