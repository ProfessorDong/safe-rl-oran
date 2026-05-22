"""
config.py
---------
Single source of truth for all simulator parameters. Modeled after Paper 3's
config.py but deliberately structured differently to keep Paper 4 independent.

Paper 4 simulates a 7-cell O-RAN cluster with primal-dual safe-RL control.
No ISAC, no Wasserstein ambiguity, no Milan/Geolife. Real datasets are
Shanghai Telecom (arrivals), Lumos5G (channel realism), and optionally
NetMob 2023 (cross-region validation).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# ========== topology ==========
@dataclass
class TopologyCfg:
    B: int = 7                   # number of cells in cluster
    isd_m: float = 250.0         # inter-site distance, m (urban dense, slightly larger than Paper 3)


# ========== time ==========
@dataclass
class TimeCfg:
    dt_ms: float = 10.0          # slot duration, ms
    T_slots_train: int = 2000    # 20 s per training episode (short, many seeds)
    T_slots_eval: int = 10000    # 100 s per eval episode (longer for tail stats)


# ========== channel / service rate ==========
@dataclass
class ChannelCfg:
    # Reference service rate per cell at full resource share (Mbps).
    # Calibrated to match Lumos5G median 5G throughput.
    mu_max_mbps: float = 80.0
    mu_min_mbps: float = 5.0     # minimum when cell is awake but throttled
    sinr_var: float = 0.20       # log-normal multiplicative noise on service rate
    # When Lumos5G data is present, mu_max is modulated per slot by the
    # empirical SINR trace; otherwise we use the synthetic noise above.


# ========== energy (EARTH-style) ==========
@dataclass
class EnergyCfg:
    p_on_W: float = 130.0        # active baseline (RF + BB)
    p_slp_W: float = 8.0         # deep sleep
    p_dyn_W: float = 100.0       # dynamic component, scales with resource share
    p_sw_W: float = 5.0          # switching penalty per toggle
    min_on_slots: int = 5
    min_off_slots: int = 5


# ========== arrivals ==========
@dataclass
class ArrivalsCfg:
    base_rate_Mb_per_slot: float = 0.3   # mean arrival per cell per slot (Mbits)
    # Shanghai Telecom-driven arrival mode (if data present):
    #   shanghai_csv: relative path under sim/data/
    shanghai_csv: str = "shanghai_telecom_sessions.csv"
    netmob_csv: str = "netmob2023_orange_france.csv"  # cross-region (optional)
    # Synthetic fallback (Poisson + Pareto-bursty):
    burst_prob: float = 0.04
    pareto_shape: float = 2.5
    burst_scale: float = 1.5
    # Diurnal modulation (sinusoid over the day):
    diurnal_amp: float = 0.40            # 1 +/- amp swing over 24 h
    busy_hour_offset_frac: float = 0.83   # peak at 20:00 (=0.83 of day)


# ========== RL / algorithm ==========
@dataclass
class AlgoCfg:
    # Constraint targets.
    beta: float = 0.95           # CVaR confidence
    Gamma: float = 3.0           # CVaR budget on per-slot loss (dimensionless)
    # Loss is normalized to [0, ell_max].
    ell_max: float = 10.0
    lam_max: float = 50.0        # cap on the dual variable for stability
    risk_warmup_slots: int = 1000  # delay activating risk constraint until t > this

    # Actor / critic.
    hidden: int = 64             # MLP hidden width (small; the state is low-dim)
    n_layers: int = 2            # MLP depth
    gamma_disc: float = 0.99     # discount factor for the cost MDP
    gae_lambda: float = 0.95     # GAE-lambda for advantage estimation
    ppo_clip: float = 0.2        # PPO clipping
    n_epochs: int = 4            # PPO update epochs per rollout
    n_mini: int = 4              # PPO mini-batches per epoch

    # Step sizes (two-timescale).
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_dual: float = 3e-3        # slow timescale for dual ascent (gentle)
    lr_tau: float = 1e-2         # CVaR threshold update
    tau_init: float = 0.5

    # Safety filter (Lyapunov-guided).
    q_safety_threshold_Mb: float = 50.0   # if total backlog > this, force max service
    lcb_kappa: float = 1.0       # conservative LCB factor: mu_LCB = mu_hat - kappa * sigma_hat

    # Ablation knobs (negative = "active learning", non-negative = "frozen at this value").
    fixed_lambda: float = -1.0   # if >= 0, freeze dual lambda at this value (no dual ascent)
    fixed_tau: float = -1.0      # if >= 0, freeze CVaR threshold tau at this value (no tau-update)

    # Cor. 2 / Prop. 1 verification knob.
    # V is the energy weight in the actor cost  c_lambda = V * P_RAN + lam * g_tau.
    # Default 1e-3 normalizes power (~1000W) to be commensurable with g_tau (~3-10).
    # Sweeping V tests the [O(1/V), O(1)] asymptotic tradeoff of Corollary 2.
    V_energy_weight: float = 1.0e-3

    # Theorem 2 / Theorem 3 verification: diminishing Robbins-Monro stepsize schedule.
    # When True, alpha_t/beta_t/gamma_t decay as t^{-exponent}; uncaps the dual too.
    diminishing_schedule: bool = False
    diminishing_alpha_exp: float = 0.55    # actor exponent (informational; actor LR kept constant in PPO)
    diminishing_beta_exp:  float = 0.55    # dual+tau exponent; p in (0.5, 1] satisfies Robbins-Monro

    # Rollout sizing.
    rollout_slots: int = 256     # actor/critic update every this many env steps
    target_tau: float = 0.005    # soft-target update for critic baseline


# ========== top-level bundle ==========
@dataclass
class SimCfg:
    topo: TopologyCfg = field(default_factory=TopologyCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    chan: ChannelCfg = field(default_factory=ChannelCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    arr: ArrivalsCfg = field(default_factory=ArrivalsCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)

    data_dir: str = "sim/data"
    results_dir: str = "sim/results"

    @property
    def dt_s(self) -> float:
        return self.time.dt_ms * 1e-3

    @property
    def state_dim(self) -> int:
        # per-cell queue (B) + recent service rate (B) + time-of-day cos/sin (2)
        return 2 * self.topo.B + 2

    @property
    def action_dim(self) -> int:
        # per-cell resource share in [0, 1]
        return self.topo.B


def default_cfg(**overrides) -> SimCfg:
    cfg = SimCfg()
    for k, v in overrides.items():
        if "." in k:
            head, tail = k.split(".", 1)
            setattr(getattr(cfg, head), tail, v)
        else:
            setattr(cfg, k, v)
    return cfg


# Canonical seed sequence for Paper 4 (different from Paper 3's 20260517).
CANONICAL_SEED = 20260601


def canonical_seeds(n: int = 10):
    """Reproducible per-seed integer list used across all experiments."""
    rng = np.random.SeedSequence(CANONICAL_SEED)
    return [int(s) for s in rng.generate_state(n)]
