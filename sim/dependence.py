"""
dependence.py
-------------
Measures the dependence the correlated-channel model actually realizes.

Section VI-H states the temporal, spatial and load coupling that the Gaussian
copula induces on the Lumos5G marginal. Those figures are a property of the
channel model alone and do not depend on the controller, so they are measured
here directly rather than inferred from a training run.

The multiplier is drawn once per slot with no environment step in between, so
the AR(1) chain is observed at its own rate; drawing it twice per slot (once
directly and once inside `step`) halves the observed persistence and is the
easy way to get this wrong.

Run:  OMP_NUM_THREADS=1 python3 -m sim.dependence
"""
from __future__ import annotations
import json
import os

import numpy as np

from .config import default_cfg
from .arrivals import generate_arrivals
from .env import CellularEnv

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def measure(T: int = 10000, seed: int = 9000) -> dict:
    cfg = default_cfg()
    cfg.chan.ar1_rho = 0.9
    cfg.chan.spatial_rho = 0.4
    cfg.chan.load_coupling = 0.6

    arr, _ = generate_arrivals(cfg, T=T, seed=seed)
    env = CellularEnv(cfg, arr, seed=seed)
    env.reset(seed=seed)
    B = cfg.topo.B

    mult, load = [], []
    for t in range(T):
        a = arr[:, t]
        mult.append(np.asarray(env._draw_channel(a), dtype=float).copy())
        load.append(np.asarray(a, dtype=float).copy())
    M = np.array(mult)
    A = np.array(load)

    lag1 = float(np.mean([np.corrcoef(M[:-1, b], M[1:, b])[0, 1]
                          for b in range(B)]))
    cross = float(np.mean([np.corrcoef(M[:, i], M[:, j])[0, 1]
                           for i in range(B) for j in range(B) if i < j]))
    rate_load = float(np.mean([np.corrcoef(M[:, b], A[:, b])[0, 1]
                               for b in range(B)]))
    return dict(lag1_autocorr=lag1, cross_cell_corr=cross,
                rate_load_corr=rate_load,
                ar1_rho=cfg.chan.ar1_rho, spatial_rho=cfg.chan.spatial_rho,
                load_coupling=cfg.chan.load_coupling,
                slots=T, cells=B, seed=seed)


def main():
    d = measure()
    print("[R8b] Realized dependence of the correlated channel")
    print(f"  lag-one autocorrelation : {d['lag1_autocorr']:+.3f}  "
          f"(AR(1) parameter {d['ar1_rho']})")
    print(f"  cross-cell correlation  : {d['cross_cell_corr']:+.3f}  "
          f"(shared innovation {d['spatial_rho']})")
    print(f"  rate-load correlation   : {d['rate_load_corr']:+.3f}  "
          f"(load coupling {d['load_coupling']})")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "rev_dependence.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
