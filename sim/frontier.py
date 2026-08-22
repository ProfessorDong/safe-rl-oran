"""
frontier.py
-----------
Traces the achievable (power, tail-risk) frontier of the classical
drift-plus-penalty family, so the proposed controller can be judged against
the whole family rather than against one or two hand-picked settings.

The headline table compares the learned controller with drift-plus-penalty at
rho = 1 and rho = 2. Two points do not establish whether the learned
controller is better than the family: a single well-chosen rho may already
dominate it. Here rho is swept on a grid and the same sweep is run for the
sleep-aware variant, giving the empirical frontier of what these classical
rules can reach on this workload. A learned controller earns its complexity
only if its operating point lies strictly inside that frontier, i.e. lower
power at equal or lower CVaR.

The evaluation protocol is byte-for-byte the one `revision._headline_one`
applies to non-learning controllers (evaluation arrivals at seed + 9000,
environment at seed + 9000, `run_episode` at seed + 9000), so the numbers are
directly comparable with `rev_headline.json`.

Run with:
    OMP_NUM_THREADS=1 python3 -m sim.frontier --seeds 10 --workers 20
"""
from __future__ import annotations
import argparse
import json
import os
import multiprocessing as mp
from typing import List

import numpy as np

from .config import default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import make_baseline, run_episode
from .metrics import bootstrap_ci

GAMMA_HEADLINE = 3.5

# Grid chosen to bracket the two settings already in the headline table
# (rho = 1 and rho = 2) on both sides, so the frontier is traced rather than
# extrapolated.
RHO_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
            2.5, 3.0, 4.0, 6.0, 8.0]
QS_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _cfg(Gamma: float = GAMMA_HEADLINE):
    c = default_cfg()
    c.algo.Gamma = Gamma
    return c


def _one(args):
    name, seed = args
    cfg = _cfg()
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr, seed=seed + 9000)
    ctl = make_baseline(name, cfg, seed=seed)
    m = run_episode(ctl, env, cfg, seed=seed + 9000)
    m["controller"] = name
    m["seed"] = seed
    return m


def _agg(rows):
    keys = ["avg_power_W", "cvar_beta", "viol_rate", "p99_delay_ms",
            "toggles_per_min"]
    return {k: bootstrap_ci([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    seeds = canonical_seeds(args.seeds)

    names: List[str] = [f"DriftPlusPenalty:{r}" for r in RHO_GRID] + \
                       [f"SleepAwareDrift:{q}" for q in QS_GRID]
    jobs = [(n, s) for n in names for s in seeds]
    with mp.Pool(args.workers) as pool:
        rows = pool.map(_one, jobs)

    out = {}
    print(f"  {'controller':>28} {'P(W)':>8} {'CVaR':>7} {'viol':>7} "
          f"{'p99(ms)':>9} {'tog/min':>8}")
    for n in names:
        rs = [r for r in rows if r["controller"] == n]
        a = _agg(rs)
        out[n] = a
        print(f"  {n:>28} {a['avg_power_W']['mean']:8.1f} "
              f"{a['cvar_beta']['mean']:7.3f} "
              f"{a['viol_rate']['mean']*100:6.1f}% "
              f"{a['p99_delay_ms']['mean']:9.1f} "
              f"{a['toggles_per_min']['mean']:8.1f}")

    # Which of these classical settings actually respect the budget, and what
    # is the cheapest one that does? That is the number the learned controller
    # has to beat.
    feasible = [(n, out[n]["avg_power_W"]["mean"], out[n]["cvar_beta"]["mean"])
                for n in names if out[n]["cvar_beta"]["mean"] <= GAMMA_HEADLINE]
    if feasible:
        best = min(feasible, key=lambda t: t[1])
        out["_best_feasible_classical"] = dict(
            name=best[0], power_W=best[1], cvar=best[2])
        print(f"\n  cheapest classical setting meeting CVaR <= "
              f"{GAMMA_HEADLINE}: {best[0]}  "
              f"P={best[1]:.1f} W  CVaR={best[2]:.3f}")
    else:
        out["_best_feasible_classical"] = None
        print(f"\n  no classical setting meets CVaR <= {GAMMA_HEADLINE}")

    out["_gamma"] = GAMMA_HEADLINE
    out["_rho_grid"] = RHO_GRID
    out["_qs_grid"] = QS_GRID
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "rev_frontier.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  -> sim/results/rev_frontier.json")


if __name__ == "__main__":
    main()
