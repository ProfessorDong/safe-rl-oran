"""
r2.py
-----
Experiment suite for the second revision (TCCN-TP-26-0764.R1 -> R2).

Differences from `revision.py` (kept for the record of the R1 results):

  * Seeds are split. Everything reported is evaluated on the ten canonical TEST
    seeds (evaluation traces at seed + 9000). Every quantity that is tuned or
    calibrated (the classical controllers' parameters, the baselines' budget
    translation) is chosen on ten separate VALIDATION seeds.
  * Every learned controller is trained once per (configuration, seed); its
    checkpoint is saved, and all evaluations (headline, filter bypass,
    correlated channel, training-budget curve) load checkpoints, so the
    evaluation of one policy under several conditions is exactly paired.
  * Per-slot loss, power and backlog series are saved for every evaluation.
  * Training draws a fresh arrival realization for every training episode.

Stages (run all with `python3 -m sim.r2 all`):
    calibrate  validation-seed calibration and classical-parameter selection
    feasible   certified floor and always-on reference on the test seeds
    classical  classical controllers on the test seeds (selected + grids)
    train      train every learned configuration (checkpoints)
    evaluate   evaluate every checkpoint (+ filter bypass, curve)
    stress     queue-stability stress test (untrained and adversarial policy)
    corr       correlated-channel study with matched marginals
    timing     per-slot actor and filter timing
Outputs: sim/results/r2/*.json, series under sim/results/r2/series/,
checkpoints under sim/results/r2/ckpt/.
"""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import os
import platform
import subprocess
import time
import multiprocessing as mp
from typing import Dict, List

import numpy as np
import torch

from .config import SimCfg, default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv, EPS_SLEEP
from .baselines import make_baseline, run_episode
from .safe_baselines import make_safe_baseline, WCSAC_GS
from .metrics import empirical_cvar, bootstrap_ci

OUT = os.environ.get("R2_OUT", os.path.join("sim", "results", "r2"))
CKPT = os.path.join(OUT, "ckpt")
SERIES = os.path.join(OUT, "series")
GAMMA = 3.5
UPDATES = 3200
CURVE_CKPTS = (200, 400, 800, 1600, 3200)

TEST_SEEDS = canonical_seeds(10)
VAL_SEEDS = canonical_seeds(20)[10:]
if os.environ.get("R2_SMOKE"):
    TEST_SEEDS, VAL_SEEDS = TEST_SEEDS[:2], VAL_SEEDS[:2]

RHO_GRID = [0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
QS_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
V_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

# Learned configurations: name -> (controller, overrides)
LEARNED = {
    "SafeRL":        ("SafeRL", {}),
    "LagPPO":        ("LagrangianPPO", {}),
    "CRPO":          ("CRPO", {}),
    "WCSAC":         ("WCSAC", {}),
    "LagPPO+F":      ("LagrangianPPO+F", {}),
    "CRPO+F":        ("CRPO+F", {}),
    "WCSAC+F":       ("WCSAC+F", {}),
    # Filter study (filter realization and predictor bias used in training
    # and evaluation alike).
    "SafeRL-none":   ("SafeRL", {"algo.safety_filter_kind": "none"}),
    "SafeRL-aggr":   ("SafeRL", {"algo.safety_filter_kind": "aggregate"}),
    "SafeRL-b+0.25": ("SafeRL", {"algo.pred_bias": 0.25}),
    "SafeRL-b+0.5":  ("SafeRL", {"algo.pred_bias": 0.5}),
    "SafeRL-b+1.0":  ("SafeRL", {"algo.pred_bias": 1.0}),
    "SafeRL-b-0.25": ("SafeRL", {"algo.pred_bias": -0.25}),
    # Sensitivity.
    "SafeRL-beta0.90":  ("SafeRL", {"algo.beta": 0.90}),
    "SafeRL-beta0.99":  ("SafeRL", {"algo.beta": 0.99}),
    "SafeRL-kappa0":    ("SafeRL", {"algo.lcb_kappa": 0.0}),
    "SafeRL-kappa0.5":  ("SafeRL", {"algo.lcb_kappa": 0.5}),
    "SafeRL-kappa2":    ("SafeRL", {"algo.lcb_kappa": 2.0}),
    "SafeRL-q0.25":     ("SafeRL", {"algo.q0_cell_Mb": 0.25}),
    "SafeRL-q4":        ("SafeRL", {"algo.q0_cell_Mb": 4.0}),
    "SafeRL-lam10":     ("SafeRL", {"algo.lam_max": 10.0}),
    "SafeRL-lam200":    ("SafeRL", {"algo.lam_max": 200.0}),
    # Cluster size.
    "SafeRL-K19":    ("SafeRL", {"topo.B": 19}),
    "SafeRL-K37":    ("SafeRL", {"topo.B": 37}),
    "SafeRL-K61":    ("SafeRL", {"topo.B": 61}),
}
GROUPS = {
    "headline": ["SafeRL", "LagPPO", "CRPO", "WCSAC",
                 "LagPPO+F", "CRPO+F", "WCSAC+F"],
    "filter": ["SafeRL-none", "SafeRL-aggr", "SafeRL-b+0.25", "SafeRL-b+0.5",
               "SafeRL-b+1.0", "SafeRL-b-0.25"],
    "sens": ["SafeRL-beta0.90", "SafeRL-beta0.99", "SafeRL-kappa0",
             "SafeRL-kappa0.5", "SafeRL-kappa2", "SafeRL-q0.25", "SafeRL-q4",
             "SafeRL-lam10", "SafeRL-lam200"],
    "scale": ["SafeRL-K19", "SafeRL-K37", "SafeRL-K61"],
}

CORR = {
    "iid":      dict(ar1_rho=0.0, spatial_rho=0.0, load_coupling=0.0),
    "temporal": dict(ar1_rho=0.9, spatial_rho=0.0, load_coupling=0.0),
    "spatial":  dict(ar1_rho=0.0, spatial_rho=0.4, load_coupling=0.0),
    "load":     dict(ar1_rho=0.0, spatial_rho=0.0, load_coupling=0.6),
    "joint":    dict(ar1_rho=0.9, spatial_rho=0.4, load_coupling=0.6),
}


# ============================================================ utilities
def _cfg(over: Dict = None, Gamma: float = GAMMA) -> SimCfg:
    c = default_cfg()
    c.algo.Gamma = Gamma
    for k, v in (over or {}).items():
        head, tail = k.split(".", 1)
        setattr(getattr(c, head), tail, v)
    return c


def _save(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    obj = dict(obj)
    obj["_provenance"] = _provenance()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"  -> {path}", flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


_PROV = {}


def _provenance() -> dict:
    if not _PROV:
        cfg = default_cfg()
        arr, src = generate_arrivals(cfg, T=256, seed=0)
        env = CellularEnv(cfg, arr, seed=0)
        try:
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          text=True).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "sim"], text=True).strip())
        except Exception:
            rev, dirty = "unknown", True
        import hashlib
        h = hashlib.sha256()
        for fn in sorted(os.listdir("sim")):
            if fn.endswith(".py"):
                h.update(open(os.path.join("sim", fn), "rb").read())
        _PROV.update(arrival_source=src, channel_source=env.channel_source,
                     require_real_data=bool(cfg.require_real_data),
                     seed_sequence=20260601, test_seeds=TEST_SEEDS,
                     val_seeds=VAL_SEEDS, git_head=rev, sim_dirty=dirty,
                     sim_sha256=h.hexdigest(), numpy=np.__version__,
                     torch=torch.__version__, python=platform.python_version(),
                     updates=UPDATES)
    return dict(_PROV)


def _eval_arrivals(cfg, seed):
    return generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)[0]


def _train_env(cfg, seed):
    """Training environment that draws a fresh arrival realization per
    episode (episode k uses a seed derived from (seed, k))."""
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)

    def regen(k, cfg=cfg, seed=seed):
        s = int(np.random.SeedSequence([seed, k, 7]).generate_state(1)[0])
        return generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=s)[0]
    return CellularEnv(cfg, arr, seed=seed, arrival_regen=regen)


def _agg(rows, keys=("avg_power_W", "cvar_beta", "cvar_95", "viol_rate",
                     "p99_delay_ms", "toggles_per_min", "awake_mean",
                     "share_exec_mean", "stranded_pct", "offload_pct",
                     "mean_loss", "avg_backlog_Mb")):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = bootstrap_ci(vals)
            out[k]["per_seed"] = vals
    fs = [r["filter"] for r in rows if "filter" in r]
    if fs:
        tot = {k: sum(f[k] for f in fs) for k in fs[0]}
        out["filter"] = tot
        if tot["cell_slots"]:
            out["filter_active_pct"] = 100.0 * tot["active"] / tot["cell_slots"]
        if tot["active"]:
            out["filter_fallback_pct"] = 100.0 * tot["fallback"] / tot["active"]
            out["filter_mean_margin_ok_pct"] = (100.0 * tot["mean_margin_ok"]
                                                / tot["active"])
    out["_n"] = len(rows)
    return out


def _save_series(tag, seed, m):
    ser = m.pop("_series", None)
    if ser is None:
        return
    os.makedirs(SERIES, exist_ok=True)
    np.savez_compressed(os.path.join(SERIES, f"{tag}_{seed}.npz"), **ser)


def _pool(workers):
    return mp.get_context("fork").Pool(workers, initializer=_worker_init)


def _worker_init():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


# ======================================================= calibration
def _classical_eval(args):
    name, seed, over, keep = args
    cfg = _cfg(over)
    env = CellularEnv(cfg, _eval_arrivals(cfg, seed), seed=seed + 9000)
    m = run_episode(make_baseline(name, cfg, seed), env, cfg, seed + 9000,
                    keep_series=keep)
    m.update(controller=name, seed=seed)
    return m


def _wcsac_ao_stat(seed):
    """Fit the WCSAC safety critic to the always-on policy on a validation
    seed and return the mean of its statistic J + k sqrt(M)."""
    cfg = _cfg()
    env = _train_env(cfg, seed)
    ctl = WCSAC_GS(cfg, seed=seed)
    ones = np.ones(cfg.topo.B, dtype=np.float32)
    raw0 = np.zeros(cfg.raw_dim, dtype=np.float32)
    ctl.act = lambda s, q, stochastic=True: (ones, raw0, ctl.augment(s), 0.0)
    stats = []
    n_cal = 12 if os.environ.get("R2_SMOKE") else 400
    for u in range(n_cal):
        b = ctl.collect_rollout(env, cfg.algo.rollout_slots)
        ctl.update_actor_critic(b, critics_only=True)
        if u >= 3 * n_cal // 4:
            stats.append(ctl.last_gamma_pi)
    return float(np.mean(stats))


def stage_calibrate(pool):
    print("[calibrate] validation seeds", flush=True)
    rows = pool.map(_classical_eval, [("AlwaysOn", s, {}, False)
                                      for s in VAL_SEEDS])
    g_ao = float(np.mean([r["cvar_beta"] for r in rows]))
    ml_ao = float(np.mean([r["mean_loss"] for r in rows]))
    ratio = GAMMA / g_ao
    wc = pool.map(_wcsac_ao_stat, VAL_SEEDS)
    budgets = dict(gamma=GAMMA, gamma_ao_val=g_ao, tightness=ratio,
                   mean_loss_ao_val=ml_ao, d_mean=ratio * ml_ao,
                   wcsac_stat_ao_val=float(np.mean(wc)),
                   d_ret=ratio * float(np.mean(wc)), wcsac_stat_per_seed=wc)
    print(f"  Gamma_AO(val)={g_ao:.3f} tightness={ratio:.3f} "
          f"d_mean={budgets['d_mean']:.3f} d_ret={budgets['d_ret']:.4f}")

    fams = {"rho": [f"DriftPlusPenalty:{r}" for r in RHO_GRID],
            "qs": [f"SleepAwareDrift:{q}" for q in QS_GRID],
            "V": [f"TextbookDPP:{v}" for v in V_GRID]}
    names = [n for v in fams.values() for n in v] + ["Threshold"]
    rows = pool.map(_classical_eval, [(n, s, {}, False) for n in names
                                      for s in VAL_SEEDS])
    grid = {n: _agg([r for r in rows if r["controller"] == n]) for n in names}
    sel = {}
    for fam, ns in fams.items():
        ok = [n for n in ns if grid[n]["cvar_beta"]["mean"] <= GAMMA]
        sel[fam] = (min(ok, key=lambda n: grid[n]["avg_power_W"]["mean"])
                    if ok else None)
        print(f"  selected {fam}: {sel[fam]}")
    _save(dict(budgets=budgets, selection=sel, val_grid=grid), "calibration.json")


def _calib():
    with open(os.path.join(OUT, "calibration.json")) as f:
        return json.load(f)


# ======================================================= feasibility
def _floor_one(seed):
    cfg = _cfg()
    arr = _eval_arrivals(cfg, seed)
    abar = arr.mean(axis=1) + 1e-6
    eta = cfg.topo.offload_efficiency
    ratio = float(abar.max() / abar.min())
    cond = ratio <= 1.0 / eta
    ell0 = np.minimum((arr / abar[:, None]).mean(axis=0), cfg.algo.ell_max)
    out = dict(seed=seed, abar=abar.tolist(), abar_ratio=ratio,
               ratio_condition=bool(cond),
               gamma0=empirical_cvar(ell0, cfg.algo.beta),
               gamma0_uncapped=empirical_cvar(
                   (arr / abar[:, None]).mean(axis=0), cfg.algo.beta))
    # General certified bound: slotwise minimum over all 2^K sleep patterns of
    # the weighted effective arrivals (valid with or without the condition).
    env = CellularEnv(cfg, arr, seed=0)
    K = cfg.topo.B
    pats = np.array(list(itertools.product([0, 1], repeat=K)))
    best = np.full(arr.shape[1], np.inf)
    for s in pats:
        a_eff = _redistribute_all(env, arr, s)
        best = np.minimum(best, (a_eff / abar[:, None]).mean(axis=0))
    out["gamma_lb_patternmin"] = empirical_cvar(
        np.minimum(best, cfg.algo.ell_max), cfg.algo.beta)
    m = run_episode(make_baseline("AlwaysOn", cfg, seed),
                    CellularEnv(cfg, arr, seed=seed + 9000), cfg, seed + 9000)
    out.update(gamma_ao=m["cvar_beta"], ao_power=m["avg_power_W"],
               ao_viol=m["viol_rate"])
    for b in (0.90, 0.99):
        cb = _cfg({"algo.beta": b})
        mb = run_episode(make_baseline("AlwaysOn", cb, seed),
                         CellularEnv(cb, arr, seed=seed + 9000), cb, seed + 9000)
        out[f"gamma_ao_beta{b}"] = mb["cvar_beta"]
        out[f"gamma0_beta{b}"] = empirical_cvar(ell0, b)
    return out


def _redistribute_all(env, arr, s):
    """Vectorized redistribute over all slots for one sleep pattern."""
    cfg = env.cfg
    awake = (np.asarray(s) == 1).astype(np.float64)
    asleep = 1.0 - awake
    share = env.cov * awake[None, :]
    reach = share.sum(axis=1)
    covered = asleep * (reach > 0)
    frac = cfg.topo.overlap_fraction
    eta = cfg.topo.offload_efficiency
    W = np.where(reach[:, None] > 0,
                 share / np.where(reach[:, None] > 0, reach[:, None], 1.0), 0.0)
    moved = arr * (covered * frac)[:, None]
    a_eff = arr * awake[:, None] + (W.T @ moved) / eta
    a_eff += arr * asleep[:, None] - moved
    return a_eff


def stage_feasible(pool):
    print("[feasible] certified floor and always-on reference", flush=True)
    rows = pool.map(_floor_one, TEST_SEEDS)
    out = {k: bootstrap_ci([r[k] for r in rows])
           for k in ("gamma0", "gamma0_uncapped", "gamma_lb_patternmin",
                     "gamma_ao", "ao_power", "ao_viol", "abar_ratio",
                     "gamma_ao_beta0.9", "gamma_ao_beta0.99",
                     "gamma0_beta0.9", "gamma0_beta0.99")}
    out["ratio_condition_all"] = all(r["ratio_condition"] for r in rows)
    out["n_ao_at_or_below_3.0"] = int(sum(r["gamma_ao"] <= 3.0 for r in rows))
    out["n_ao_at_or_below_3.5"] = int(sum(r["gamma_ao"] <= 3.5 for r in rows))
    out["per_seed"] = rows
    print(f"  Gamma0={out['gamma0']['mean']:.3f} (pattern-min "
          f"{out['gamma_lb_patternmin']['mean']:.3f})  "
          f"Gamma_AO={out['gamma_ao']['mean']:.3f}  "
          f"ratio condition on all seeds: {out['ratio_condition_all']}")
    _save(out, "feasibility.json")


# ======================================================= classical
def stage_classical(pool):
    print("[classical] test seeds", flush=True)
    sel = _calib()["selection"]
    headline = ["AlwaysOn", "Threshold", sel["V"], sel["rho"],
                "DriftPlusPenalty:1.0", "DriftPlusPenalty:2.0", sel["qs"],
                "SleepAwareDrift:0.05"]
    headline = list(dict.fromkeys(n for n in headline if n))
    grid = ([f"DriftPlusPenalty:{r}" for r in RHO_GRID]
            + [f"SleepAwareDrift:{q}" for q in QS_GRID]
            + [f"TextbookDPP:{v}" for v in V_GRID])
    names = list(dict.fromkeys(headline + grid))
    rows = pool.map(_classical_eval, [(n, s, {}, n in headline)
                                      for n in names for s in TEST_SEEDS])
    out = {}
    for n in names:
        rs = [r for r in rows if r["controller"] == n]
        for r in rs:
            _save_series(n.replace(":", "_"), r["seed"], r)
        out[n] = _agg(rs)
    out["_headline"] = headline
    out["_selection"] = sel
    _save(out, "classical.json")


# ======================================================= training
def _build(name, cfg, seed, budgets):
    ctrl, _ = LEARNED[name]
    base = ctrl[:-2] if ctrl.endswith("+F") else ctrl
    if base == "SafeRL":
        ctl = make_baseline("SafeRL", cfg, seed)
        ctl.use_safety_filter = cfg.algo.safety_filter_kind != "none"
    else:
        ctl = make_safe_baseline(base, cfg, seed, budgets=budgets)
        if ctrl.endswith("+F"):
            ctl.use_safety_filter = True
    return ctl


def _ckpt_path(name, seed, u):
    return os.path.join(CKPT, name, f"{seed}_u{u}.pt")


def _state_of(ctl):
    return dict(actor=ctl.actor.state_dict(), lam=ctl.lam, tau=ctl.tau,
                a_hat=ctl._a_hat.copy(), a_nom_hat=ctl._a_nom_hat.copy(),
                on_constraint=getattr(ctl, "_on_constraint", False),
                pid_I=getattr(ctl, "_pid_I", 0.0))


def _train_one(args):
    name, seed, budgets = args
    _worker_init()
    _, over = LEARNED[name]
    cfg = _cfg(over)
    final = _ckpt_path(name, seed, UPDATES)
    if os.path.exists(final):
        return name, seed, "cached"
    env = _train_env(cfg, seed)
    ctl = _build(name, cfg, seed, budgets)
    log = dict(lam=[], tau=[], g=[], loss=[], power=[], viol=[])
    ckpts = ({c for c in CURVE_CKPTS if c < UPDATES} | {UPDATES}
             if name == "SafeRL" else {UPDATES})
    os.makedirs(os.path.join(CKPT, name), exist_ok=True)
    t0 = time.time()
    for u in range(1, UPDATES + 1):
        b = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)
        log["lam"].append(float(ctl.lam)); log["tau"].append(float(ctl.tau))
        log["g"].append(float(np.mean(b["g_tau"])))
        log["loss"].append(float(np.mean(b["loss"])))
        log["power"].append(float(np.mean(b["energy"])))
        log["viol"].append(float(np.mean(b["viol"])))
        if u in ckpts:
            torch.save(_state_of(ctl), _ckpt_path(name, seed, u))
    np.savez_compressed(os.path.join(CKPT, name, f"{seed}_log.npz"),
                        train_s=time.time() - t0,
                        **{k: np.array(v, dtype=np.float32)
                           for k, v in log.items()})
    return name, seed, time.time() - t0


def stage_train(pool, groups):
    budgets = _calib()["budgets"]
    names = [n for g in groups for n in GROUPS[g]]
    # Longest jobs first so the pool tail is short.
    names.sort(key=lambda n: -LEARNED[n][1].get("topo.B", 7))
    jobs = [(n, s, budgets) for n in names for s in TEST_SEEDS]
    print(f"[train] {len(jobs)} jobs x {UPDATES} updates", flush=True)
    t0 = time.time()
    for i, (n, s, dt) in enumerate(pool.imap_unordered(_train_one, jobs), 1):
        print(f"  [{i}/{len(jobs)}] {n} seed {s}: {dt if isinstance(dt, str) else f'{dt:.0f}s'}"
              f"  (elapsed {time.time()-t0:.0f}s)", flush=True)


# ======================================================= evaluation
def _load(name, seed, u, budgets, over_extra=None):
    _, over = LEARNED[name]
    over = dict(over)
    over.update(over_extra or {})
    cfg = _cfg(over)
    ctl = _build(name, cfg, seed, budgets)
    st = torch.load(_ckpt_path(name, seed, u), weights_only=False)
    ctl.actor.load_state_dict(st["actor"])
    ctl.lam, ctl.tau = st["lam"], st["tau"]
    ctl._a_hat, ctl._a_nom_hat = st["a_hat"], st["a_nom_hat"]
    if hasattr(ctl, "_on_constraint"):
        ctl._on_constraint = st["on_constraint"]
    return cfg, ctl


def _eval_one(args):
    name, seed, u, mode, budgets = args
    _worker_init()
    extra = CORR[mode[5:]] if mode.startswith("corr:") else {}
    extra = {f"chan.{k}": v for k, v in extra.items()}
    cfg, ctl = _load(name, seed, u, budgets, extra)
    if mode == "bypass":
        ctl.use_safety_filter = False
    env = CellularEnv(cfg, _eval_arrivals(cfg, seed), seed=seed + 9000)
    m = run_episode(ctl, env, cfg, seed + 9000, keep_series=True)
    tag = f"{name}_u{u}_{mode.replace(':', '-')}"
    _save_series(tag, seed, m)
    m.update(name=name, seed=seed, u=u, mode=mode)
    if mode in ("default", "bypass"):
        m["share_prop_mean"] = _proposed_share(ctl, cfg, seed)
    return m


def _proposed_share(ctl, cfg, seed):
    """Mean share the actor PROPOSES along the executed (filtered or not)
    trajectory, for comparison with the executed share."""
    c2 = copy.deepcopy(ctl)
    env = CellularEnv(cfg, _eval_arrivals(cfg, seed), seed=seed + 9000)
    s = env.reset(seed=seed + 9000)
    c2.bind(env)
    props = []
    done = False
    while not done:
        aug = c2.augment(s)
        with torch.no_grad():
            prop = c2.actor.deterministic(
                torch.from_numpy(aug).float().unsqueeze(0)).numpy().squeeze(0)
        props.append(float(prop.mean()))
        a, _, _, _ = c2.act(s, env.q, stochastic=False)
        s, c, done = env.step(a)
        c2.observe(c)
    return float(np.mean(props))


def stage_evaluate(pool, groups):
    budgets = _calib()["budgets"]
    jobs = []
    for g in groups:
        for n in GROUPS[g]:
            for s in TEST_SEEDS:
                jobs.append((n, s, UPDATES, "default", budgets))
    if "headline" in groups:
        for s in TEST_SEEDS:
            jobs.append(("SafeRL", s, UPDATES, "bypass", budgets))
            for u in (c for c in CURVE_CKPTS if c < UPDATES):
                jobs.append(("SafeRL", s, u, "default", budgets))
    print(f"[evaluate] {len(jobs)} evaluations", flush=True)
    rows = pool.map(_eval_one, jobs)
    out = {}
    keyset = sorted({(r["name"], r["u"], r["mode"]) for r in rows})
    for n, u, mode in keyset:
        rs = [r for r in rows if (r["name"], r["u"], r["mode"]) == (n, u, mode)]
        a = _agg(rs, keys=("avg_power_W", "cvar_beta", "cvar_95", "viol_rate",
                           "p99_delay_ms", "toggles_per_min", "awake_mean",
                           "share_exec_mean", "share_prop_mean", "stranded_pct",
                           "offload_pct", "mean_loss", "avg_backlog_Mb"))
        a["lam_final"] = bootstrap_ci([_lam_of(n, r["seed"], u) for r in rs])
        out[f"{n}|u{u}|{mode}"] = a
    tag = "_".join(groups)
    _save(out, f"learned_{tag}.json")


def _lam_of(name, seed, u):
    return float(torch.load(_ckpt_path(name, seed, u),
                            weights_only=False)["lam"])


# ======================================================= stress
class _AllSleep:
    """Adversarial proposal: every cell asks to sleep at every slot."""

    def __init__(self, inner):
        self.inner = inner
        self.filter_stats = inner.filter_stats

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def act(self, state, q, stochastic=False):
        from .safety_filter import lcb_project
        a = np.zeros(self.inner.cfg.topo.B, dtype=np.float32)
        if self.inner.use_safety_filter:
            a, force = lcb_project(a, self.inner._env, self.inner._a_nom_hat,
                                   self.inner.cfg, self.inner._chan_mean,
                                   self.inner._chan_std, self.filter_stats)
            self.inner._env._pending_force = force
        return a, None, state, 0.0


def _stress_one(args):
    rate, policy, use_filter, seed = args
    _worker_init()
    cfg = _cfg({"algo.fixed_lambda": 0.0}, Gamma=100.0)
    cfg.arr.base_rate_Mb_per_slot = rate
    env = CellularEnv(cfg, _eval_arrivals(cfg, seed), seed=seed + 9000)
    ctl = make_baseline("SafeRL", cfg, seed)
    ctl.use_safety_filter = use_filter
    if policy == "allsleep":
        ctl = _AllSleep(ctl)
    m = run_episode(ctl, env, cfg, seed + 9000)
    offered = float(_eval_arrivals(cfg, seed).mean() / cfg.dt_s)
    m.update(rate=rate, policy=policy, use_filter=use_filter, seed=seed,
             offered_Mbps_per_cell=offered)
    return m


def stage_stress(pool):
    print("[stress] untrained and all-sleep proposals, filter on/off", flush=True)
    rates = [0.3, 0.5, 0.7]
    jobs = [(r, p, f, s) for r in rates for p in ("untrained", "allsleep")
            for f in (True, False) for s in TEST_SEEDS]
    rows = pool.map(_stress_one, jobs)
    out = {}
    for r in rates:
        for p in ("untrained", "allsleep"):
            for f in (True, False):
                rs = [x for x in rows if (x["rate"], x["policy"],
                                          x["use_filter"]) == (r, p, f)]
                a = _agg(rs, keys=("avg_backlog_Mb", "avg_power_W",
                                   "p99_delay_ms", "cvar_95",
                                   "offered_Mbps_per_cell", "awake_mean"))
                out[f"rate{r}|{p}|filter{f}"] = a
                print(f"  rate {r} {p:9s} filter {str(f):5s} "
                      f"Qbar={a['avg_backlog_Mb']['mean']:9.2f} "
                      f"P={a['avg_power_W']['mean']:7.1f}", flush=True)
    _save(out, "stress.json")


# ======================================================= correlation
def _corr_classical(args):
    name, cond, seed = args
    over = {f"chan.{k}": v for k, v in CORR[cond].items()}
    m = _classical_eval((name, seed, over, False))
    m.update(cond=cond)
    return m


def _dependence(cond, seed):
    cfg = _cfg({f"chan.{k}": v for k, v in CORR[cond].items()})
    arr = _eval_arrivals(cfg, seed)
    env = CellularEnv(cfg, arr, seed=seed + 9000)
    env.reset(seed=seed + 9000)
    M = []
    for t in range(arr.shape[1]):
        env.t = t
        M.append(env._draw_channel().astype(float))
    M = np.array(M)
    A = arr.T
    K = cfg.topo.B
    pool_ = np.asarray(env.channel_pool, dtype=float)
    return dict(
        lag1=float(np.mean([np.corrcoef(M[:-1, b], M[1:, b])[0, 1]
                            for b in range(K)])),
        cross=float(np.mean([np.corrcoef(M[:, i], M[:, j])[0, 1]
                             for i in range(K) for j in range(i + 1, K)])),
        rate_load=float(np.mean([np.corrcoef(M[:, b], A[:, b])[0, 1]
                                 for b in range(K)])),
        mean=float(M.mean()), sd=float(M.std()),
        low_mass=float((M <= 0.1 + 1e-6).mean()),
        high_mass=float((M >= 2.0 - 1e-6).mean()),
        pool_mean=float(pool_.mean()), pool_sd=float(pool_.std()),
        pool_low=float((pool_ <= 0.1 + 1e-6).mean()),
        pool_high=float((pool_ >= 2.0 - 1e-6).mean()))


def stage_corr(pool):
    print("[corr] correlated channel with matched marginal", flush=True)
    budgets = _calib()["budgets"]
    sel = _calib()["selection"]
    classical = [n for n in ("AlwaysOn", sel["rho"], sel["qs"]) if n]
    conds = list(CORR)
    rows = pool.map(_corr_classical, [(n, c, s) for n in classical
                                      for c in conds for s in TEST_SEEDS])
    lrows = pool.map(_eval_one, [("SafeRL", s, UPDATES, f"corr:{c}", budgets)
                                 for c in conds for s in TEST_SEEDS])
    dep = {c: _dependence(c, TEST_SEEDS[0]) for c in conds}
    out = {"dependence": dep}
    for c in conds:
        for n in classical:
            out[f"{n}|{c}"] = _agg([r for r in rows
                                    if r["controller"] == n and r["cond"] == c])
        out[f"SafeRL|{c}"] = _agg([r for r in lrows if r["mode"] == f"corr:{c}"])
        print(f"  {c:9s} AO CVaR={out[f'AlwaysOn|{c}']['cvar_beta']['mean']:.3f} "
              f"SafeRL CVaR={out[f'SafeRL|{c}']['cvar_beta']['mean']:.3f} "
              f"marginal mean {dep[c]['mean']:.3f} (pool {dep[c]['pool_mean']:.3f})",
              flush=True)
    _save(out, "correlation.json")


# ======================================================= timing
def _timing_one(K):
    _worker_init()
    cfg = _cfg({"topo.B": K})
    seed = TEST_SEEDS[0]
    env = CellularEnv(cfg, _eval_arrivals(cfg, seed), seed=seed + 9000)
    name = "SafeRL" if K == 7 else f"SafeRL-K{K}"
    try:
        _, ctl = _load(name, seed, UPDATES, _calib()["budgets"])
    except Exception:
        ctl = make_baseline("SafeRL", cfg, seed)
    ctl.bind(env)
    s = env.reset(seed=seed + 9000)
    from .safety_filter import lcb_project
    t_actor, t_filter = [], []
    for t in range(2500):
        aug = ctl.augment(s)
        x = torch.from_numpy(aug).float().unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            a = ctl.actor.deterministic(x).numpy().squeeze(0)
        t1 = time.perf_counter()
        a2, force = lcb_project(a, env, ctl._a_nom_hat, cfg, ctl._chan_mean,
                                ctl._chan_std)
        t2 = time.perf_counter()
        env._pending_force = force
        s, c, done = env.step(a2)
        ctl.observe(c)
        if t >= 500:                      # warmup excluded
            t_actor.append((t1 - t0) * 1e6)
            t_filter.append((t2 - t1) * 1e6)
    def pct(x):
        x = np.array(x)
        return dict(mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                    p99=float(np.percentile(x, 99)),
                    p999=float(np.percentile(x, 99.9)), max=float(x.max()))
    return K, dict(actor_us=pct(t_actor), filter_us=pct(t_filter),
                   total_us=pct(np.array(t_actor) + np.array(t_filter)))


def stage_timing():
    print("[timing] sequential, one process, one thread", flush=True)
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    out = {"cpu": cpu}
    for K in (7, 19, 37, 61):
        k, d = _timing_one(K)
        out[f"K={k}"] = d
        print(f"  K={k}: actor p50 {d['actor_us']['p50']:.0f} us, "
              f"filter p50 {d['filter_us']['p50']:.0f} us, "
              f"total p99 {d['total_us']['p99']:.0f} us", flush=True)
    _save(out, "timing.json")


# ======================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "feasible", "classical",
                                      "train", "evaluate", "stress", "corr",
                                      "timing", "all"])
    ap.add_argument("--groups", default="headline,filter,sens,scale")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--updates", type=int, default=UPDATES)
    args = ap.parse_args()
    globals()["UPDATES"] = args.updates
    groups = args.groups.split(",")
    torch.set_num_threads(1)
    t0 = time.time()
    with _pool(args.workers) as pool:
        if args.stage in ("calibrate", "all"):
            stage_calibrate(pool)
        if args.stage in ("feasible", "all"):
            stage_feasible(pool)
        if args.stage in ("classical", "all"):
            stage_classical(pool)
        if args.stage in ("train", "all"):
            stage_train(pool, groups)
        if args.stage in ("evaluate", "all"):
            stage_evaluate(pool, groups)
        if args.stage in ("stress", "all"):
            stage_stress(pool)
        if args.stage in ("corr", "all"):
            stage_corr(pool)
    if args.stage in ("timing", "all"):
        stage_timing()
    print(f"[r2] {args.stage} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
