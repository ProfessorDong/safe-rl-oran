"""
check_dpp.py
------------
Exhaustive check of TextbookDPP against the environment's feasible actions.

For random states (queues, previous sleep pattern, dwell counters, load
estimates) of a K-cell cluster, every action in {0, 0.05, 1}^K is executed
through the environment's own hysteresis (predict_sleep) and offloading map,
and the drift-plus-penalty objective V P + sum_b Q_b (A_b - mu_b) is computed
with the controller's arrival estimate and mean channel. The controller's
choice must attain the minimum. With an affine objective in each share the
minimum over [0, 1]^K is attained on this endpoint set.

Run: PYTHONPATH=. python -m sim.check_dpp   (writes results/r2/check_dpp.json)
"""
from __future__ import annotations
import itertools
import json

import numpy as np

from .config import default_cfg
from .env import CellularEnv
from .baselines import TextbookDPP


def objective(env, ctl, a, q):
    c, e = env.cfg, env.cfg.energy
    s = env.predict_sleep(np.asarray(a, dtype=np.float32))
    A = env.redistribute(ctl._a_hat, s)[0].astype(np.float64)
    wf = np.where((s == 1) & (env.s_prev == 0), e.wake_service_frac, 1.0)
    phi = np.clip(np.asarray(a, dtype=np.float64), 0, 1)
    mu = s * wf * (c.chan.mu_min_mbps * c.dt_s
                   + (c.chan.mu_max_mbps - c.chan.mu_min_mbps) * c.dt_s * phi) * ctl._m
    P = (np.where(s == 1, e.p_on_W + e.p_dyn_W * phi, e.p_slp_W).sum()
         + e.p_sw_W * np.sum(s != env.s_prev))
    return ctl.V * P + float(np.dot(q, A - mu))


def main(n_cases=150, seed=0):
    rng = np.random.default_rng(seed)
    cfg = default_cfg()
    K = cfg.topo.B
    arr = np.full((K, 20), 0.3, dtype=np.float32)
    env = CellularEnv(cfg, arr)
    env.reset(seed=0)
    grid = np.array(list(itertools.product([0.0, 0.05, 1.0], repeat=K)))
    worst, fails = 0.0, 0
    for i in range(n_cases):
        V = float(rng.choice([1e-5, 1e-4, 4e-4, 1e-3, 1e-2]))
        ctl = TextbookDPP(cfg, V=V)
        ctl.bind(env)
        env.s_prev = rng.integers(0, 2, K).astype(env.s_prev.dtype)
        env.dwell = rng.integers(1, 8, K).astype(env.dwell.dtype)
        q = rng.choice([0.0, 0.05, 0.3, 1.0, 5.0], K) * rng.random(K)
        env.q = q.astype(np.float32)
        ctl._a_hat = rng.uniform(0.05, 0.6, K)
        env.t = 0
        a = ctl.act(None, env.q)[0]
        best = min(objective(env, ctl, g, q) for g in grid)
        gap = objective(env, ctl, a, q) - best
        worst = max(worst, gap)
        fails += gap > 1e-7
    out = dict(cases=n_cases, failures=int(fails), worst_gap=float(worst))
    print(json.dumps(out))
    with open("sim/results/r2/check_dpp.json", "w") as f:
        json.dump(out, f, indent=2)
    assert fails == 0


if __name__ == "__main__":
    main()
