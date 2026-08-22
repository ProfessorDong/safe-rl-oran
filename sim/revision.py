"""
revision.py
-----------
Experiment suite for the TCCN major revision (TCCN-TP-26-0764).

Run everything:
    python3 -m sim.revision all --seeds 10

Individual experiments:
    python3 -m sim.revision feasibility    # R1: CVaR feasibility test
    python3 -m sim.revision headline       # R2: full controller comparison
    python3 -m sim.revision filter         # R3: safety-filter study
    python3 -m sim.revision sensitivity    # R4: beta / lam_max / q0 / kappa
    python3 -m sim.revision scalability    # R5: cluster size and wall-clock
    python3 -m sim.revision stress         # R6: Theorem 1 stress test

Results are written as JSON under sim/results/rev_*.json.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import multiprocessing as mp
from typing import Dict, List

import numpy as np

from .config import SimCfg, default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import make_baseline, run_episode
from .safe_baselines import make_safe_baseline
from .metrics import empirical_cvar, bootstrap_ci

RESULTS = "sim/results"
# Single training budget shared by every experiment that trains, recorded in
# each result file's provenance block.
_TRAIN_UPDATES = [None]
# Headline risk budget. Chosen from the feasibility test: the CVaR-optimal
# AlwaysOn policy attains ~3.0, so any budget at or below that is infeasible
# by construction; 3.5 leaves a genuine Slater margin.
GAMMA_HEADLINE = 3.5

METRIC_KEYS = ("avg_power_W", "cvar_beta", "viol_rate", "p99_delay_ms",
               "toggles_per_min", "awake_mean", "stranded_pct")

NON_LEARNING = [
    ("AlwaysOn", "Always-on"),
    ("Threshold", "Threshold heuristic"),
    ("LyapunovOnly", "Lyapunov drift (bang-bang)"),
    ("DriftPlusPenalty", "Drift-plus-penalty ($\\rho{=}1$)"),
    ("DriftPlusPenalty:2.0", "Drift-plus-penalty ($\\rho{=}2$)"),
    ("SleepAwareDrift", "Sleep-aware drift ($q_s{=}0.05$)"),
    ("SleepAwareDrift:0.2", "Sleep-aware drift ($q_s{=}0.2$)"),
]
LEARNING = [
    ("LagrangianPPO", "PPO-Lagrangian (expected cost)"),
    ("CRPO", "CRPO"),
    ("WCSAC", "WCSAC-GS"),
    # Each learned baseline is also run WITH the LCB safety filter. Without
    # these rows the comparison against the proposed controller confounds the
    # constraint mechanism with the filter, which Section VI-D shows is the
    # single largest lever; with them, the two effects separate.
    ("LagrangianPPO+filter", "PPO-Lagrangian + LCB filter"),
    ("CRPO+filter", "CRPO + LCB filter"),
    ("WCSAC+filter", "WCSAC-GS + LCB filter"),
    ("SafeRL", "Proposed (Safe-RL + LCB filter)"),
]


def _provenance() -> dict:
    """Record which real datasets actually fed the run.

    The submitted pipeline could silently substitute a fallback trace, so
    every result file now carries the resolved source and a fingerprint of
    the arrival profile.
    """
    cfg = default_cfg()
    arr, src = generate_arrivals(cfg, T=256, seed=0)
    env = CellularEnv(cfg, arr, seed=0)
    return dict(arrival_source=src, channel_source=env.channel_source,
                require_real_data=bool(cfg.require_real_data),
                seed_sequence=20260601,
                train_updates=_TRAIN_UPDATES[0])


def _save(obj, name):
    if isinstance(obj, dict):
        obj = dict(obj)
        obj["_provenance"] = _provenance()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  -> {path}")


def _cfg(Gamma=GAMMA_HEADLINE, **over) -> SimCfg:
    c = default_cfg()
    c.algo.Gamma = Gamma
    for k, v in over.items():
        head, tail = (k.split(".", 1) + [None])[:2] if "." in k else (None, None)
        if head:
            setattr(getattr(c, head), tail, v)
        else:
            setattr(c.algo, k, v)
    return c


def _agg(rows: List[dict]) -> dict:
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = bootstrap_ci(vals)
    out["_n"] = len(rows)
    return out


# ===================================================================== R1
def _feas_one(args):
    seed, Gamma = args
    cfg = _cfg(Gamma)
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    # Policy-independent floor: the loss contains the arrival term, which no
    # scheduler can influence, so CVaR of that term alone lower-bounds any
    # achievable CVaR.
    abar = arr.mean(axis=1) + 1e-6
    floor = empirical_cvar((arr / abar[:, None]).mean(axis=0), cfg.algo.beta)
    env = CellularEnv(cfg, arr, seed=seed + 9000)
    m = run_episode(make_baseline("AlwaysOn", cfg, seed), env, cfg, seed + 9000)
    return dict(seed=seed, floor=float(floor), always_on=m["cvar_beta"],
                always_on_power=m["avg_power_W"])


def exp_feasibility(seeds, pool):
    print("[R1] CVaR feasibility test")
    rows = pool.map(_feas_one, [(s, GAMMA_HEADLINE) for s in seeds])
    floor = np.array([r["floor"] for r in rows])
    ao = np.array([r["always_on"] for r in rows])
    out = dict(
        floor=bootstrap_ci(floor.tolist()),
        always_on_cvar=bootstrap_ci(ao.tolist()),
        always_on_power=bootstrap_ci([r["always_on_power"] for r in rows]),
        gamma_min_any_policy=float(floor.mean()),
        gamma_min_implementable=float(ao.mean()),
        gamma_submitted=3.0,
        submitted_was_feasible=bool(3.0 >= ao.mean()),
        gamma_headline=GAMMA_HEADLINE,
        per_seed=rows,
    )
    print(f"  policy-independent floor      : {floor.mean():.3f}")
    print(f"  best implementable (always-on): {ao.mean():.3f}")
    print(f"  submitted Gamma = 3.0 feasible: {out['submitted_was_feasible']}")
    _save(out, "rev_feasibility.json")
    return out


# ===================================================================== R2
def _headline_one(args):
    name, seed, Gamma, n_updates = args
    cfg = _cfg(Gamma)
    if name in dict(NON_LEARNING):
        arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
        env = CellularEnv(cfg, arr, seed=seed + 9000)
        ctl = make_baseline(name, cfg, seed=seed)
    else:
        arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
        env_t = CellularEnv(cfg, arr, seed=seed)
        base, with_filter = (name[:-7], True) if name.endswith("+filter") \
            else (name, False)
        ctl = (make_baseline("SafeRL", cfg, seed=seed) if base == "SafeRL"
               else make_safe_baseline(base, cfg, seed=seed))
        if with_filter:
            ctl.use_safety_filter = True
        for _ in range(n_updates):
            b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
            ctl.update_actor_critic(b)
        arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
        env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m["controller"] = name
    m["seed"] = seed
    return m


def exp_headline(seeds, pool, n_updates=800):
    print(f"[R2] Headline comparison (Gamma={GAMMA_HEADLINE}, "
          f"{n_updates} updates)")
    names = [n for n, _ in NON_LEARNING] + [n for n, _ in LEARNING]
    jobs = [(n, s, GAMMA_HEADLINE, n_updates) for n in names for s in seeds]
    rows = pool.map(_headline_one, jobs)
    out = {}
    label = dict(NON_LEARNING + LEARNING)
    print(f"  {'controller':>34} {'P(W)':>8} {'CVaR':>7} {'viol':>7} "
          f"{'p99(ms)':>8} {'tog/min':>8}")
    for n in names:
        rs = [r for r in rows if r["controller"] == n]
        a = _agg(rs)
        a["label"] = label[n]
        out[n] = a
        print(f"  {label[n]:>34} {a['avg_power_W']['mean']:8.1f} "
              f"{a['cvar_beta']['mean']:7.3f} "
              f"{a['viol_rate']['mean']*100:6.1f}% "
              f"{a['p99_delay_ms']['mean']:8.1f} "
              f"{a['toggles_per_min']['mean']:8.1f}")
    out["_gamma"] = GAMMA_HEADLINE
    out["_pareto"] = _pareto([(n, out[n]["avg_power_W"]["mean"],
                               out[n]["cvar_beta"]["mean"]) for n in names])
    _save(out, "rev_headline.json")
    return out


def _pareto(points):
    """Non-dominated set for (power, CVaR), both minimized."""
    keep = []
    for n, p, c in points:
        if not any((p2 <= p and c2 <= c and (p2 < p or c2 < c))
                   for n2, p2, c2 in points if n2 != n):
            keep.append(n)
    return keep


# ===================================================================== R3
def _filter_one(args):
    kind, bias, seed, n_updates = args
    cfg = _cfg(safety_filter_kind=kind, pred_bias=bias)
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env_t = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    ctl.use_safety_filter = (kind != "none")
    for _ in range(n_updates):
        b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m.update(kind=kind, bias=bias, seed=seed)
    return m


def exp_filter(seeds, pool, n_updates=400):
    print("[R3] Safety-filter study (kind x service-predictor error)")
    combos = ([("none", 0.0), ("aggregate", 0.0)]
              + [("lcb", b) for b in (0.0, 0.25, 0.5, 1.0, -0.25)])
    jobs = [(k, b, s, n_updates) for k, b in combos for s in seeds]
    rows = pool.map(_filter_one, jobs)
    out = {}
    print(f"  {'filter':>12} {'pred bias':>10} {'P(W)':>8} {'CVaR':>7} "
          f"{'p99(ms)':>8}")
    for k, b in combos:
        rs = [r for r in rows if r["kind"] == k and r["bias"] == b]
        a = _agg(rs)
        out[f"{k}_bias{b}"] = a
        print(f"  {k:>12} {b:>10.2f} {a['avg_power_W']['mean']:8.1f} "
              f"{a['cvar_beta']['mean']:7.3f} {a['p99_delay_ms']['mean']:8.1f}")
    _save(out, "rev_filter.json")
    return out


# ===================================================================== R4
def _sens_one(args):
    knob, val, seed, n_updates = args
    over = {knob: val}
    cfg = _cfg(**over)
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env_t = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    for _ in range(n_updates):
        b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m.update(knob=knob, val=val, seed=seed)
    return m


def exp_sensitivity(seeds, pool, n_updates=400):
    print("[R4] Sensitivity to beta, lam_max, q0_cell, kappa, delta")
    grid = ([("beta", v) for v in (0.90, 0.95, 0.99)]
            + [("lam_max", v) for v in (10.0, 50.0, 200.0)]
            + [("q0_cell_Mb", v) for v in (0.25, 1.0, 4.0)]
            + [("lcb_kappa", v) for v in (0.0, 0.5, 1.0, 2.0)]
            + [("delta_margin_Mb", v) for v in (0.0, 0.02, 0.1)])
    jobs = [(k, v, s, n_updates) for k, v in grid for s in seeds]
    rows = pool.map(_sens_one, jobs)
    out = {}
    print(f"  {'knob':>18} {'value':>8} {'P(W)':>8} {'CVaR':>7} {'viol':>7}")
    for k, v in grid:
        rs = [r for r in rows if r["knob"] == k and r["val"] == v]
        a = _agg(rs)
        out[f"{k}={v}"] = a
        print(f"  {k:>18} {v:>8} {a['avg_power_W']['mean']:8.1f} "
              f"{a['cvar_beta']['mean']:7.3f} {a['viol_rate']['mean']*100:6.1f}%")
    _save(out, "rev_sensitivity.json")
    return out


# ===================================================================== R5
def _scale_one(args):
    K, seed, n_updates = args
    cfg = _cfg()
    cfg.topo.B = K
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env_t = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    t0 = time.time()
    for _ in range(n_updates):
        b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
    train_s = time.time() - t0
    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    t0 = time.time()
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    infer_us = (time.time() - t0) / cfg.time.T_slots_eval * 1e6
    m.update(K=K, seed=seed, train_s=train_s, infer_us_per_slot=infer_us,
             power_per_cell=m["avg_power_W"] / K)
    return m


def exp_scalability(seeds, pool, n_updates=200):
    print("[R5] Scalability in cluster size (with per-slot inference time)")
    Ks = [7, 19, 37, 61]
    jobs = [(K, s, n_updates) for K in Ks for s in seeds]
    rows = pool.map(_scale_one, jobs)
    out = {}
    print(f"  {'K':>4} {'P/cell(W)':>10} {'CVaR':>7} {'infer(us/slot)':>15} "
          f"{'train(s)':>9}")
    for K in Ks:
        rs = [r for r in rows if r["K"] == K]
        a = _agg(rs)
        a["power_per_cell"] = bootstrap_ci([r["power_per_cell"] for r in rs])
        a["infer_us_per_slot"] = bootstrap_ci([r["infer_us_per_slot"] for r in rs])
        a["train_s"] = bootstrap_ci([r["train_s"] for r in rs])
        out[f"K={K}"] = a
        print(f"  {K:>4} {a['power_per_cell']['mean']:10.1f} "
              f"{a['cvar_beta']['mean']:7.3f} "
              f"{a['infer_us_per_slot']['mean']:15.1f} "
              f"{a['train_s']['mean']:9.1f}")
    _save(out, "rev_scalability.json")
    return out


# ===================================================================== R6
def _stress_one(args):
    rate, use_filter, seed = args
    cfg = _cfg(Gamma=100.0, fixed_lambda=0.0)   # risk pressure removed
    cfg.arr.base_rate_Mb_per_slot = rate
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr, seed=seed + 9000)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    ctl.use_safety_filter = use_filter
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m.update(rate=rate, use_filter=use_filter, seed=seed)
    return m


def exp_stress(seeds, pool):
    print("[R6] Theorem 1 stress test (filter on/off, risk pressure removed)")
    rates = [0.30, 0.50, 0.70]
    jobs = [(r, f, s) for r in rates for f in (True, False) for s in seeds]
    rows = pool.map(_stress_one, jobs)
    out = {}
    print(f"  {'rate':>6} {'filter':>7} {'Qbar(Mb)':>10} {'p99(ms)':>9}")
    for r in rates:
        for f in (True, False):
            rs = [x for x in rows if x["rate"] == r and x["use_filter"] == f]
            qb = bootstrap_ci([x["avg_backlog_Mb"] for x in rs])
            p99 = bootstrap_ci([x["p99_delay_ms"] for x in rs])
            out[f"rate{r}_filter{f}"] = dict(backlog=qb, p99=p99)
            print(f"  {r:>6} {str(f):>7} {qb['mean']:10.1f} {p99['mean']:9.1f}")
    _save(out, "rev_stress.json")
    return out


# ===================================================================== R7
def _vsweep_one(args):
    V, seed, n_updates = args
    cfg = _cfg(V_energy_weight=V)
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env_t = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    for _ in range(n_updates):
        b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m.update(V=V, seed=seed)
    return m


def exp_vsweep(seeds, pool, n_updates=400):
    """Corollary 2: the safety filter should bound time-average backlog
    uniformly in the Lyapunov weight V, which is stricter than the canonical
    Neely O(V) growth."""
    print("[R7] V-sweep (Corollary 2)")
    Vs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    jobs = [(V, s, n_updates) for V in Vs for s in seeds]
    rows = pool.map(_vsweep_one, jobs)
    out = {}
    print(f"  {'V':>8} {'Qbar(Mb)':>10} {'P(W)':>8} {'CVaR':>7}")
    for V in Vs:
        rs = [r for r in rows if r["V"] == V]
        qb = bootstrap_ci([r["avg_backlog_Mb"] for r in rs])
        pw = bootstrap_ci([r["avg_power_W"] for r in rs])
        cv = bootstrap_ci([r["cvar_beta"] for r in rs])
        out[f"V={V}"] = dict(backlog=qb, power=pw, cvar=cv)
        print(f"  {V:8.0e} {qb['mean']:10.2f} {pw['mean']:8.1f} {cv['mean']:7.3f}")
    qs = [out[f"V={V}"]["backlog"]["mean"] for V in Vs]
    out["_ratio_max_over_min"] = max(qs) / max(min(qs), 1e-9)
    out["_neely_OV_would_predict"] = qs[0] * (Vs[-1] / Vs[0])
    print(f"  backlog spread over 4 decades of V: {max(qs)/max(min(qs),1e-9):.1f}x "
          f"(Neely O(V) would predict {qs[0]*(Vs[-1]/Vs[0]):.0f} Mb at V=1)")
    _save(out, "rev_vsweep.json")
    return out


# ===================================================================== R8
def _corr_one(args):
    name, corr, seed, n_updates = args
    cfg = _cfg()
    if corr:
        cfg.chan.ar1_rho = 0.9
        cfg.chan.spatial_rho = 0.4
        cfg.chan.load_coupling = 0.6
    if name == "SafeRL":
        arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
        env_t = CellularEnv(cfg, arr, seed=seed)
        ctl = make_baseline("SafeRL", cfg, seed=seed)
        for _ in range(n_updates):
            b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
            ctl.update_actor_critic(b)
    else:
        ctl = make_baseline(name, cfg, seed=seed)
    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr_e, seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m.update(name=name, corr=corr, seed=seed)
    return m


def exp_correlation(seeds, pool, n_updates=400):
    """Reviewer 1, point 7: the submitted channel drew the Lumos5G
    multiplier i.i.d. per cell per slot. Here a Gaussian copula imposes
    temporal persistence, cross-cell correlation and negative
    load/rate coupling while preserving the empirical marginal."""
    print("[R8] Correlated channel (AR(1) 0.9, spatial 0.4, load coupling 0.6)")
    names = ["DriftPlusPenalty", "SleepAwareDrift", "SafeRL"]
    jobs = [(n, c, s, n_updates) for n in names for c in (False, True)
            for s in seeds]
    rows = pool.map(_corr_one, jobs)
    out = {}
    print(f"  {'controller':>18} {'channel':>12} {'P(W)':>8} {'CVaR':>7} {'p99(ms)':>9}")
    for n in names:
        for c in (False, True):
            rs = [r for r in rows if r["name"] == n and r["corr"] == c]
            a = _agg(rs)
            out[f"{n}_corr{c}"] = a
            print(f"  {n:>18} {('correlated' if c else 'i.i.d.'):>12} "
                  f"{a['avg_power_W']['mean']:8.1f} {a['cvar_beta']['mean']:7.3f} "
                  f"{a['p99_delay_ms']['mean']:9.1f}")
    _save(out, "rev_correlation.json")
    return out


# ===================================================================== R9
def exp_compute(seeds, pool):
    """Reviewer 1, point 8: account for the energy of running the
    controller itself, not just the radio."""
    print("[R9] xApp compute energy")
    import time as _t
    cfg = _cfg()
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seeds[0] + 9000)
    env = CellularEnv(cfg, arr, seed=seeds[0] + 9000)
    ctl = make_baseline("SafeRL", cfg, seed=seeds[0])
    ctl.bind(env)
    s = env.reset(seed=seeds[0] + 9000)
    n = 2000
    t0 = _t.perf_counter()
    for _ in range(n):
        a, _, _, _ = ctl.act(s, env.q, stochastic=False)
        s, c, done = env.step(a)
        ctl.observe(c)
        if done:
            s = env.reset(seed=seeds[0] + 9000)
    dt_us = (_t.perf_counter() - t0) / n * 1e6
    duty = dt_us / (cfg.dt_s * 1e6)
    # A near-RT RIC core is commonly budgeted at 15-25 W; take 20 W and
    # charge only the fraction of wall-clock the controller occupies.
    core_W = 20.0
    xapp_W = core_W * duty
    radio_W = 1149.9
    out = dict(inference_us_per_slot=dt_us, slot_duty_fraction=duty,
               assumed_core_W=core_W, xapp_power_W=xapp_W,
               radio_power_W=radio_W,
               overhead_pct=100.0 * xapp_W / radio_W)
    print(f"  inference {dt_us:.0f} us/slot -> {100*duty:.1f}% of a 10 ms slot")
    print(f"  xApp power {xapp_W:.2f} W on a {core_W:.0f} W core "
          f"= {out['overhead_pct']:.3f}% of {radio_W:.0f} W radio power")
    _save(out, "rev_compute.json")
    return out


# ===================================================================== R10
def _curve_one(args):
    """Train once per seed, evaluating on a DEEP COPY at each checkpoint.

    Evaluation is not side-effect free: run_episode calls ctl.observe(), which
    advances the controller's EWMA arrival estimate. Evaluating the live
    controller mid-training therefore perturbs the run it is measuring, so we
    snapshot instead.
    """
    import copy
    seed, ckpts = args
    cfg = _cfg()
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    out = {}
    for u in range(1, max(ckpts) + 1):
        b = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
        if u in ckpts:
            snap = copy.deepcopy(ctl)
            arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                         seed=seed + 9000)
            m = run_episode(snap, CellularEnv(cfg, arr_e, seed=seed + 9000),
                            cfg, seed=seed + 9000)
            out[u] = dict(cvar=m["cvar_beta"], power=m["avg_power_W"],
                          viol=m["viol_rate"], tog=m["toggles_per_min"],
                          lam=float(ctl.lam))
    return dict(seed=seed, ckpt=out)


def exp_curve(seeds, pool, n_updates):
    """Tail risk against training budget, used to justify the budget choice."""
    ck = [c for c in (200, 400, 800, 1600, 3200, 6400) if c <= n_updates]
    if n_updates not in ck:
        ck.append(n_updates)
    print(f"[R10] CVaR versus training budget (checkpoints {ck})")
    rows = pool.map(_curve_one, [(s, ck) for s in seeds])
    out = {}
    print(f"  {'updates':>8} {'CVaR':>8} {'[lo,hi]':>16} {'P(W)':>8} "
          f"{'viol':>7} {'lambda':>8}")
    for u in ck:
        cv = bootstrap_ci([r["ckpt"][u]["cvar"] for r in rows])
        pw = bootstrap_ci([r["ckpt"][u]["power"] for r in rows])
        vl = bootstrap_ci([r["ckpt"][u]["viol"] for r in rows])
        lm = bootstrap_ci([r["ckpt"][u]["lam"] for r in rows])
        out[str(u)] = dict(cvar=cv, power=pw, viol=vl, lam=lm)
        print(f"  {u:8d} {cv['mean']:8.3f} [{cv['lo']:6.3f},{cv['hi']:6.3f}] "
              f"{pw['mean']:8.1f} {vl['mean']*100:6.1f}% {lm['mean']:8.2f}")
    _save(out, "rev_curve.json")
    return out


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["all", "feasibility", "headline", "filter",
                             "sensitivity", "scalability", "stress",
                             "vsweep", "correlation", "compute", "curve"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    _TRAIN_UPDATES[0] = args.updates
    seeds = canonical_seeds(args.seeds)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        if args.which in ("all", "feasibility"):
            exp_feasibility(seeds, pool)
        if args.which in ("all", "headline"):
            exp_headline(seeds, pool, args.updates)
        if args.which in ("all", "filter"):
            exp_filter(seeds, pool, args.updates)
        if args.which in ("all", "sensitivity"):
            exp_sensitivity(seeds, pool, args.updates)
        if args.which in ("all", "scalability"):
            exp_scalability(seeds, pool, args.updates)
        if args.which in ("all", "stress"):
            exp_stress(seeds, pool)
        if args.which in ("all", "vsweep"):
            exp_vsweep(seeds, pool, args.updates)
        if args.which in ("all", "correlation"):
            exp_correlation(seeds, pool, args.updates)
        if args.which in ("all", "compute"):
            exp_compute(seeds, pool)
        if args.which in ("all", "curve"):
            exp_curve(seeds, pool, args.updates)
    print(f"\n[revision suite] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
