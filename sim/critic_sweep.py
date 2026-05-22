"""
critic_sweep.py
---------------
Critic-capacity sweep to disentangle Proposition 1's energy-gap
decomposition  J(pi-hat) - J(pi*) <= C1*eps_V + C2*beta-bar + C3/V.

Hypothesis: in Exp A (V-sweep) the observed power decay across
four orders of V was only ~8%, with a log-log slope of -0.010 vs
the -1 predicted by the C3/V term. This sweep tests whether the
C1*eps_V term (critic approximation error) dominates: shrinking
eps_V via a larger critic should either
  (a) reveal the underlying O(1/V) rate (Prop. 1 verified), or
  (b) show that eps_V is not the bottleneck (Prop. 1's 1/V
      rate is not observable in the practical regime).

Design
------
  Critic hidden widths : {32, 64, 128, 256}  (4 levels)
  V values             : {1e-3, 1e-2, 1e-1, 1.0}
  Seeds                : 5 canonical seeds
  Updates per run      : 200

Output: sim/results/critic_sweep.json
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


def _train_and_eval(cfg, seed, n_updates):
    arrivals, _ = generate_arrivals(cfg, T=cfg.time.T_slots_train, seed=seed)
    env = CellularEnv(cfg, arrivals, seed=seed)
    ctl = SafeRLController(cfg, seed=seed,
                            use_safety_filter=True, enforce_risk=True)
    for u in range(n_updates):
        batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        ctl.update_actor_critic(batch)
    # Eval
    arr_eval, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                     seed=seed + 9000)
    env_eval = CellularEnv(cfg, arr_eval, seed=seed + 9000)
    energy_s, loss_s, backlog_s, arr_s = [], [], [], []
    toggles = 0
    s = env_eval.state()
    done = False
    while not done:
        a, _, _, _ = ctl.act(s, env_eval.q, stochastic=False)
        s_next, cost, done = env_eval.step(a)
        energy_s.append(cost["energy_W"])
        loss_s.append(cost["loss"])
        backlog_s.append(cost["q_total_Mb"])
        arr_s.append(cost["arrived_Mb"])
        toggles += cost["n_toggles"]
        s = s_next
    return summarize_episode(np.array(energy_s), np.array(loss_s),
                              np.array(backlog_s), np.array(arr_s),
                              toggles=toggles, beta=cfg.algo.beta,
                              Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",   type=int, default=5)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--hidden",  type=int, nargs="+",
                    default=[32, 64, 128, 256])
    ap.add_argument("--Vs",      type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0])
    args = ap.parse_args()

    seeds = canonical_seeds(args.seeds)
    out = {"hidden": args.hidden, "Vs": args.Vs, "grid": {}}
    t_global = time.time()
    for h in args.hidden:
        for V in args.Vs:
            print(f"\n[hidden={h}  V={V:g}]")
            rows = []
            for i, s in enumerate(seeds):
                t0 = time.time()
                cfg = default_cfg()
                cfg.algo.hidden = h
                cfg.algo.V_energy_weight = V
                m = _train_and_eval(cfg, s, args.updates)
                rows.append(m)
                print(f"  seed {i+1}/{args.seeds} = {s}: "
                      f"P={m['avg_power_W']:7.1f}W  "
                      f"<Q>={m['avg_backlog_Mb']:5.1f}Mb  "
                      f"CVaR={m['cvar_beta']:5.2f}  "
                      f"viol={m['viol_rate']*100:4.1f}%  "
                      f"({time.time()-t0:.0f}s)")
            agg = aggregate(rows)
            out["grid"][f"h{h}_V{V:g}"] = agg
            print(f"  >> h={h} V={V:g}: P={agg['avg_power_W']['mean']:.1f}W "
                  f"<Q>={agg['avg_backlog_Mb']['mean']:.1f}Mb")

    out_path = "sim/results/critic_sweep.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[critic_sweep] saved {out_path} "
          f"(total wall-clock {time.time()-t_global:.0f}s)")


if __name__ == "__main__":
    main()
