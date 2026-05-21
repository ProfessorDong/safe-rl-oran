"""
gamma_sweep.py
--------------
After the main sweep finishes, retrain SafeRL with a small grid of risk
budgets Gamma to produce the energy-vs-violation Pareto frontier (figure E3).
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import default_cfg, canonical_seeds
from .run import train_one_seed_named, eval_one_seed
from .metrics import aggregate

RESULTS_DIR = "sim/results"
FIG_DIR = "fig"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1.5, 2.0, 3.0, 4.5, 7.0])
    ap.add_argument("--seeds",   type=int, default=4)
    ap.add_argument("--updates", type=int, default=150)
    args = ap.parse_args()

    seeds = canonical_seeds(args.seeds)
    grid = {}
    for G in args.gammas:
        cfg = default_cfg()
        cfg.algo.Gamma = float(G)
        rows = []
        print(f"\n[gamma={G:.2f}]")
        for s in seeds:
            t0 = time.time()
            rec = train_one_seed_named(cfg, s, "SafeRL", n_updates=args.updates)
            ev = eval_one_seed(cfg, s, "SafeRL",
                               trained_actor=rec["actor_state_dict"],
                               trained_critic=rec["critic_state_dict"])
            rows.append(ev)
            print(f"  seed {s}: P={ev['avg_power_W']:.0f}W "
                  f"viol={ev['viol_rate']*100:.1f}%  ({time.time()-t0:.0f}s)")
        grid[f"G_{G:.2f}"] = aggregate(rows)

    out = os.path.join(RESULTS_DIR, "gamma_sweep.json")
    with open(out, "w") as f:
        json.dump({"gammas": args.gammas, "results": grid}, f,
                   indent=2, default=float)
    print(f"\n[saved] {out}")

    # Plot
    Gs = []
    Ps = []
    Plo = []
    Phi = []
    Vs = []
    Vlo = []
    Vhi = []
    for G in args.gammas:
        d = grid[f"G_{G:.2f}"]
        Gs.append(G)
        Ps.append(d["avg_power_W"]["mean"])
        Plo.append(d["avg_power_W"]["lo"])
        Phi.append(d["avg_power_W"]["hi"])
        Vs.append(d["viol_rate"]["mean"] * 100)
        Vlo.append(d["viol_rate"]["lo"] * 100)
        Vhi.append(d["viol_rate"]["hi"] * 100)

    Gs = np.array(Gs)
    Ps = np.array(Ps); Plo = np.array(Plo); Phi = np.array(Phi)
    Vs = np.array(Vs); Vlo = np.array(Vlo); Vhi = np.array(Vhi)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4),
                             constrained_layout=True)
    ax = axes[0]
    ax.errorbar(Gs, Ps, yerr=[Ps - Plo, Phi - Ps],
                marker="o", color="#1f77b4", capsize=3, linewidth=1.2)
    ax.set_xlabel(r"Risk budget $\Gamma$")
    ax.set_ylabel("Average power (W)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(Gs, Vs, yerr=[Vs - Vlo, Vhi - Vs],
                marker="s", color="#d62728", capsize=3, linewidth=1.2)
    ax.axhline(5.0, color="k", linestyle=":", linewidth=0.7,
               label=r"$1{-}\beta=5\%$")
    ax.set_xlabel(r"Risk budget $\Gamma$")
    ax.set_ylabel(r"Loss-exceedance rate $\Pr\{\ell>\Gamma\}$ (\%)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, "e3_gamma_sweep.pdf"))
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR}/e3_gamma_sweep.pdf")


if __name__ == "__main__":
    main()
