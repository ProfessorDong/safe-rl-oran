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

    # --- inter-cell coverage overlap (R1 revision) ------------------------
    # The submitted model had B independent per-cell queues with no coupling,
    # so a sleeping cell simply starved its own users, cell sleep was never
    # worthwhile, and the optimal policy factorized per cell. That removes
    # the very mechanism the O-RAN "Carrier and Cell Switch Off/On" use case
    # relies on: when a cell sleeps, neighbouring cells with overlapping
    # coverage absorb its traffic.
    enable_offload: bool = True
    # Offloaded traffic is carried at lower spectral efficiency because the
    # UE is served by a farther cell. A UE at its own cell edge that is
    # picked up by a neighbour sits at roughly the ISD rather than the cell
    # radius; under a 3GPP UMa exponent this costs a factor of order 0.5-0.7
    # in achievable rate, so offloaded bits consume 1/eta times the resources.
    offload_efficiency: float = 0.6
    # Fraction of a cell's traffic that lies in the overlap region and is
    # therefore reachable by a neighbour at all. Traffic outside the overlap
    # is stranded when the cell sleeps and stays in its own queue.
    overlap_fraction: float = 0.8


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
    min_on_slots: int = 5
    min_off_slots: int = 5

    # --- sleep-mode transition cost (R1 revision) -------------------------
    # Calibrated to the advanced-sleep-mode (ASM) measurements of Salem et
    # al. (VTC-Fall 2017, Table II) rather than the placeholder 5 W used in
    # the submitted version. Their 3-sector 2x2 MIMO site draws 750 W at full
    # load, 328 W idle and 28.5 W in SM3; per sector that is ~250 W / ~109 W
    # / ~9.5 W, which brackets (p_on + p_dyn) / p_on / p_slp here, so the
    # steady-state levels were already sector-calibrated.
    #
    # The transition was not. At a 10 ms slot the relevant depth is SM3
    # (10 ms transition time), split by Salem et al. into a deactivation
    # slope and an activation slope of half the transition each. During
    # deactivation the radio still draws active power; during activation it
    # draws sleep power but cannot carry data. Charging the excess
    # deactivation energy (p_on - p_slp) * 5 ms = 0.61 J to the sleep-entry
    # toggle and amortizing over the on->off->on pair gives
    #   p_sw ~ 0.61 J / 2 / 10 ms ~ 30 W per toggle,
    # a 6x increase over the placeholder. `wake_service_frac` charges the
    # other half of the cost: a cell that wakes this slot spends the 5 ms
    # activation slope unable to serve, so it delivers half a slot of data.
    p_sw_W: float = 30.0
    wake_service_frac: float = 0.5   # served fraction of a slot on wake-up
    asm_transition_ms: float = 10.0  # SM3 transition time (informational)


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
    log_std_init: float = -0.5   # initial actor exploration scale (pre-sigmoid)
    # Factored per-cell action: discrete sleep/active x continuous share.
    # With the purely continuous sigmoid actor the sleep region phi < 0.05
    # sits ~4.85 sigma out (~6e-7 per cell-slot), so the learner never
    # explored cell sleep at all and was confined to throttling.
    factored_action: bool = True
    active_bias: float = 1.0     # initial logit bias toward "active"
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
    # `lcb` is the filter the theory specifies (Eq. 9-10): per-cell projection
    # onto {a : mu_LCB(x,a) >= a_hat + delta} whenever q_b >= q0_cell.
    # `aggregate` is the coarse top-half-backlog rule of the submitted
    # version, retained for the comparison Reviewer 3 asked for.
    safety_filter_kind: str = "lcb"       # "lcb" | "aggregate" | "none"
    q_safety_threshold_Mb: float = 50.0   # aggregate rule: cluster trigger
    q0_cell_Mb: float = 1.0               # lcb rule: per-cell activation q0
    delta_margin_Mb: float = 0.02         # lcb rule: drift margin delta
    lcb_kappa: float = 1.0       # conservative LCB factor: mu_LCB = mu_hat - kappa * sigma_hat
    # Service-predictor error, swept for the robustness study (R1.3, R2.2).
    pred_bias: float = 0.0       # >0 = optimistic predictor (overestimates service)
    pred_scale: float = 1.0      # scaling on the predicted uncertainty

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

    # --- R1 revision: primal-dual conditioning fixes -----------------------
    # Separate energy and risk critics. A single critic on c_lambda has to
    # represent a value function whose SCALE moves with lambda, which makes
    # the regression target non-stationary and destroys the advantage
    # estimates once the dual grows. Two critics on the (fixed) energy and
    # risk costs are each stationary; the dual then only re-weights their
    # advantages. Section IV-B of the submitted paper already flagged this
    # variant; the revision adopts it as the default.
    two_critics: bool = True
    # Dual-conditioned policy: append (lambda, tau) to the actor/critic
    # observation. Without this the cost is non-stationary from the network's
    # point of view (Reviewer 3, point 5).
    dual_in_state: bool = True
    # Expose the per-cell sleep state and min-dwell counter, which are part
    # of the true Markov state (they gate the admissible action through the
    # hysteresis) but were hidden from the actor in the submitted version.
    sleep_state_in_obs: bool = True

    # PID dual control (Stooke, Achiam & Abbeel, ICML 2020, Alg. 2).
    # The submitted update lam <- [lam + beta*(g_tau - Gamma)]_+ is pure
    # INTEGRAL control, which is exactly why lam wound up to lam_max and
    # stayed pinned there: an integral controller facing a persistent
    # positive error accumulates without bound. PID recomputes the multiplier
    # from the current error each iteration,
    #     lam = [K_P * delta + K_I * I + K_D * (delta - delta_prev)_+]_+ ,
    # so the proportional term responds immediately, the derivative term acts
    # before an overshoot, and the integral term still removes steady-state
    # error. Setting K_P = K_D = 0 recovers the submitted behaviour.
    use_pid_dual: bool = True
    pid_kp: float = 1.0
    pid_ki: float = 3e-3         # matches the submitted lr_dual
    pid_kd: float = 2.0
    pid_per_rollout: bool = True  # control at update rate, not per slot


# ========== top-level bundle ==========
@dataclass
class SimCfg:
    topo: TopologyCfg = field(default_factory=TopologyCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    chan: ChannelCfg = field(default_factory=ChannelCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    arr: ArrivalsCfg = field(default_factory=ArrivalsCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)

    # When set, the loaders raise instead of silently substituting a
    # synthetic trace. Every experiment reported in the paper runs with this
    # on, so a missing dataset fails loudly rather than producing plausible
    # numbers from the wrong source.
    require_real_data: bool = True

    data_dir: str = "sim/data"
    results_dir: str = "sim/results"

    @property
    def dt_s(self) -> float:
        return self.time.dt_ms * 1e-3

    @property
    def env_state_dim(self) -> int:
        """Observation emitted by the environment.

        Base: per-cell queue (B) + recent service rate (B) + time-of-day (2).
        With `sleep_state_in_obs`, adds per-cell sleep indicator (B) and
        normalized min-dwell counter (B), which complete the Markov state.
        """
        d = 2 * self.topo.B + 2
        if self.algo.sleep_state_in_obs:
            d += 2 * self.topo.B
        return d

    @property
    def state_dim(self) -> int:
        """Full network input: environment observation plus, when
        `dual_in_state` is set, the controller's own (lambda, tau)."""
        d = self.env_state_dim
        if self.algo.dual_in_state:
            d += 2
        return d

    @property
    def raw_dim(self) -> int:
        """Width of the stored policy-sample record used for the PPO ratio:
        (z, raw) for the factored actor, raw alone otherwise."""
        return 2 * self.topo.B if self.algo.factored_action else self.topo.B

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
