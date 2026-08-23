"""
resource_share.py
-----------------
Measures the mean executed resource share of each controller.

Section VI-C explains the energy gap between the learned controller and tuned
drift-plus-penalty by the actions each one takes: the classical rule serves in
proportion to backlog and holds a low mean share, while the learned policy
settles roughly seventy percent higher and sleeps too rarely to recover the
difference. The learned figure comes out of `attribution.py`, which records the
share the actor proposes and the share the filter executes. The classical
figures had no such record, so this experiment supplies them.

Share is averaged over executed actions, so a controller that never sleeps and
one that sleeps often are compared on what they actually asked the radios for.
Note the value depends on rho: the cheapest setting meeting the budget
(rho = 0.75) runs leaner than the rho = 1 default, and Section VI-C quotes the
former, since that is the point the comparison is drawn against.

The evaluation protocol is the one `revision._headline_one` applies to
non-learning controllers (arrivals, environment and episode all at seed + 9000),
so these numbers line up with `rev_headline.json` and `rev_frontier.json`.

Run:  OMP_NUM_THREADS=1 python3 -m sim.resource_share
"""
from __future__ import annotations
import argparse
import json
import os
import multiprocessing as mp

import numpy as np

from .config import default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import make_baseline
from .metrics import bootstrap_ci

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GAMMA_HEADLINE = 3.5

# The rho = 0.75 entry is the cheapest classical setting meeting the budget and
# is the one Section VI-C compares the learned policy against.
CONTROLLERS = ["AlwaysOn", "LyapunovOnly", "DriftPlusPenalty:0.75",
               "DriftPlusPenalty", "SleepAwareDrift"]


def _one(args):
    name, seed = args
    cfg = default_cfg()
    cfg.algo.Gamma = GAMMA_HEADLINE
    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)
    env = CellularEnv(cfg, arr, seed=seed + 9000)
    ctl = make_baseline(name, cfg, seed=seed)
    if hasattr(ctl, "bind"):
        ctl.bind(env)
    if hasattr(ctl, "reset"):
        ctl.reset(seed + 9000)
    s = env.reset(seed=seed + 9000)
    shares = []
    done = False
    while not done:
        a, _, _, _ = ctl.act(s, env.q, stochastic=False)
        shares.append(float(np.asarray(a, dtype=float).mean()))
        s, c, done = env.step(a)
        if hasattr(ctl, "observe"):
            ctl.observe(c)
    return name, float(np.mean(shares))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    seeds = canonical_seeds(args.seeds)

    jobs = [(n, s) for n in CONTROLLERS for s in seeds]
    with mp.Pool(args.workers) as pool:
        rows = pool.map(_one, jobs)

    out = {}
    print("[R11] Mean executed resource share")
    for n in CONTROLLERS:
        vals = [r[1] for r in rows if r[0] == n]
        out[n] = bootstrap_ci(vals)
        print(f"  {n:>24}  {out[n]['mean']:.4f}")
    out["_gamma"] = GAMMA_HEADLINE
    out["_n"] = len(seeds)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "rev_resource_share.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
