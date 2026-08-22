"""
attribution.py
--------------
Attributes the safety filter's effect on a *fixed* policy.

Section VI-D compares filter realizations by training a separate controller
for each, which confounds the filter with the training run it produced. This
experiment removes that confound: it trains one controller at the headline
setting and then evaluates those same weights twice, once with the LCB
projection engaged and once with it bypassed. Because the policy is identical
in both arms, the difference is the filter's doing and nothing else.

It also records how often the projection actually binds, and by how much it
raises the requested resource share, which is what distinguishes "a few timely
interventions" from "a uniformly more conservative policy".

Run:  OMP_NUM_THREADS=1 python3 -m sim.attribution --seeds 5 --updates 3200
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from .config import default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import make_baseline, run_episode
from .safety_filter import lcb_project
from .metrics import bootstrap_ci

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GAMMA_HEADLINE = 3.5


def _one(seed: int, updates: int) -> dict:
    cfg = default_cfg()
    cfg.algo.Gamma = GAMMA_HEADLINE

    arr, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env_t = CellularEnv(cfg, arr, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    for _ in range(updates):
        b = ctl.collect_rollout(env_t, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(b)

    arr_e, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval, seed=seed + 9000)

    ctl.use_safety_filter = True
    on = run_episode(ctl, CellularEnv(cfg, arr_e, seed=seed + 9000), cfg,
                     seed=seed + 9000)
    ctl.use_safety_filter = False
    off = run_episode(ctl, CellularEnv(cfg, arr_e, seed=seed + 9000), cfg,
                      seed=seed + 9000)

    # How often, and how hard, does the projection bind?
    ctl.use_safety_filter = True
    ee = CellularEnv(cfg, arr_e, seed=seed + 9000)
    ctl.bind(ee)
    st = ee.reset(seed=seed + 9000)
    n_cell = n_bound = 0
    prop, exe = [], []
    done = False
    while not done:
        aug = ctl.augment(st)
        with torch.no_grad():
            raw = ctl.actor.deterministic(
                torch.from_numpy(aug).float().unsqueeze(0))
        a_prop = raw.numpy().squeeze(0)
        a_exec = lcb_project(a_prop, ee.q, ctl._a_hat, cfg,
                             ctl._chan_mean, ctl._chan_std)
        n_cell += len(a_prop)
        n_bound += int((a_exec > a_prop + 1e-6).sum())
        prop.append(a_prop.mean())
        exe.append(a_exec.mean())
        st, c, done = ee.step(a_exec)
        ctl.observe(c)

    return dict(seed=seed,
                power_on=on["avg_power_W"], cvar_on=on["cvar_beta"],
                power_off=off["avg_power_W"], cvar_off=off["cvar_beta"],
                bind_pct=100.0 * n_bound / n_cell,
                phi_proposed=float(np.mean(prop)),
                phi_executed=float(np.mean(exe)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--updates", type=int, default=3200)
    args = ap.parse_args()
    torch.set_num_threads(1)

    rows = []
    for seed in canonical_seeds(args.seeds):
        r = _one(seed, args.updates)
        rows.append(r)
        print(f"  seed {seed}: with filter P={r['power_on']:7.1f} "
              f"CVaR={r['cvar_on']:6.3f} | without P={r['power_off']:7.1f} "
              f"CVaR={r['cvar_off']:6.3f} | bind={r['bind_pct']:4.1f}%",
              flush=True)

    agg = {k: bootstrap_ci([r[k] for r in rows])
           for k in ("power_on", "cvar_on", "power_off", "cvar_off",
                     "bind_pct", "phi_proposed", "phi_executed")}
    agg["energy_cost_W"] = agg["power_on"]["mean"] - agg["power_off"]["mean"]
    agg["_updates"] = args.updates
    agg["_n"] = len(rows)
    agg["per_seed"] = rows

    print(f"\n  with filter : P={agg['power_on']['mean']:7.1f} W  "
          f"CVaR={agg['cvar_on']['mean']:6.3f}")
    print(f"  bypassed    : P={agg['power_off']['mean']:7.1f} W  "
          f"CVaR={agg['cvar_off']['mean']:6.3f}")
    print(f"  filter costs{agg['energy_cost_W']:+8.1f} W, binds on "
          f"{agg['bind_pct']['mean']:.1f}% of cell-slots, "
          f"phi {agg['phi_proposed']['mean']:.3f} -> "
          f"{agg['phi_executed']['mean']:.3f}")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "rev_attribution.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2, default=float)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
