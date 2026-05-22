"""
ablations.py
------------
Component-ablation driver for Paper 4. Runs four variants on the
canonical 10-seed protocol with the same training horizon as the
headline (250 updates), then a held-out 100s evaluation episode per
seed. Same metric set as Table II.

Variants
--------
  FullSafeRL          : safety filter + dual + tau-update (baseline)
  NoSafetyFilter      : dual + tau-update active, but no projection
                        onto the safe set (q_0 = inf)
  NoDual              : safety filter + tau-update active, but lambda
                        frozen at 10 (no dual ascent)
  NoTauUpdate         : safety filter + dual active, but tau frozen
                        at 2.0 (no CVaR-threshold subgradient descent)

Output
------
  sim/results/ablations.json  -- aggregated metrics + per-seed values
                                  for each variant.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import SafeRLController
from .metrics import summarize_episode, aggregate


VARIANTS = {
    "FullSafeRL":     dict(use_safety_filter=True,  enforce_risk=True,
                           fixed_lambda=-1.0, fixed_tau=-1.0),
    "NoSafetyFilter": dict(use_safety_filter=False, enforce_risk=True,
                           fixed_lambda=-1.0, fixed_tau=-1.0),
    "NoDual":         dict(use_safety_filter=True,  enforce_risk=True,
                           fixed_lambda=10.0, fixed_tau=-1.0),
    "NoTauUpdate":    dict(use_safety_filter=True,  enforce_risk=True,
                           fixed_lambda=-1.0, fixed_tau=2.0),
}


def make_cfg(variant_kwargs):
    cfg = default_cfg()
    cfg.algo.fixed_lambda = variant_kwargs["fixed_lambda"]
    cfg.algo.fixed_tau    = variant_kwargs["fixed_tau"]
    return cfg


def train_seed(variant_name, kwargs, seed, n_updates, verbose=False):
    cfg = make_cfg(kwargs)
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_train,
                                          seed=seed)
    env = CellularEnv(cfg, arrivals, seed=seed)
    ctl = SafeRLController(cfg, seed=seed,
                            use_safety_filter=kwargs["use_safety_filter"],
                            enforce_risk=kwargs["enforce_risk"])
    history = {"viol_rate": [], "avg_energy_W": [], "avg_loss": [],
               "lambda": [], "tau": [], "z": []}
    for u in range(n_updates):
        batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        info = ctl.update_actor_critic(batch)
        history["viol_rate"].append(float(batch["viol"].mean()))
        history["avg_energy_W"].append(float(batch["energy"].mean()))
        history["avg_loss"].append(float(batch["loss"].mean()))
        history["lambda"].append(info["lambda"])
        history["tau"].append(info["tau"])
        history["z"].append(info["z"])
        if verbose and u % 25 == 0:
            print(f"    [{variant_name}/seed={seed}] u={u:3d} "
                  f"viol={history['viol_rate'][-1]:.3f} "
                  f"E={history['avg_energy_W'][-1]:.0f}W "
                  f"lam={info['lambda']:.2f} tau={info['tau']:.2f}")
    return ctl, source


def eval_seed(variant_name, kwargs, seed, trained_ctl):
    cfg = make_cfg(kwargs)
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                          seed=seed + 9000)
    env = CellularEnv(cfg, arrivals, seed=seed + 9000)
    energy_s, loss_s, backlog_s, arr_s = [], [], [], []
    toggles = 0
    s = env.state()
    done = False
    while not done:
        a, _, _, _ = trained_ctl.act(s, env.q, stochastic=False)
        s_next, cost, done = env.step(a)
        energy_s.append(cost["energy_W"])
        loss_s.append(cost["loss"])
        backlog_s.append(cost["q_total_Mb"])
        arr_s.append(cost["arrived_Mb"])
        toggles += cost["n_toggles"]
        s = s_next
    metrics = summarize_episode(
        np.array(energy_s), np.array(loss_s),
        np.array(backlog_s), np.array(arr_s),
        toggles=toggles,
        beta=cfg.algo.beta, Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s,
    )
    metrics["source"] = source
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",   type=int, default=10)
    ap.add_argument("--updates", type=int, default=250)
    ap.add_argument("--variants", type=str, nargs="+",
                    default=list(VARIANTS.keys()))
    args = ap.parse_args()

    seeds = canonical_seeds(args.seeds)
    print(f"[ablations] {args.seeds} seeds x {args.updates} updates x "
          f"{len(args.variants)} variants")
    all_results = {}
    t_global = time.time()
    for v in args.variants:
        if v not in VARIANTS:
            print(f"  ! unknown variant {v}, skipping"); continue
        kwargs = VARIANTS[v]
        print(f"\n[{v}]  kwargs={kwargs}")
        rows = []
        for i, s in enumerate(seeds):
            t0 = time.time()
            ctl, src = train_seed(v, kwargs, s, args.updates, verbose=False)
            ev = eval_seed(v, kwargs, s, ctl)
            rows.append(ev)
            print(f"  [seed {i+1}/{args.seeds} = {s}] "
                  f"P={ev['avg_power_W']:7.1f}W  "
                  f"CVaR={ev['cvar_beta']:5.2f}  "
                  f"viol={ev['viol_rate']*100:5.1f}%  "
                  f"p99={ev['p99_delay_ms']:6.0f}ms  "
                  f"tog={ev['toggles_per_min']:7.1f}/min  "
                  f"({time.time()-t0:.0f}s)")
        all_results[v] = aggregate(rows)
        m = all_results[v]
        print(f"  >> {v}: P={m['avg_power_W']['mean']:.1f}W "
              f"CVaR={m['cvar_beta']['mean']:.3f} "
              f"viol={m['viol_rate']['mean']*100:.2f}% "
              f"tog={m['toggles_per_min']['mean']:.1f}/min")

    out_path = "sim/results/ablations.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[ablations] saved {out_path} "
          f"(total wall-clock {time.time()-t_global:.0f}s)")


if __name__ == "__main__":
    main()
