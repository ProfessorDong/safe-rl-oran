"""
make_figures.py
---------------
Generate the four experimental figures for Paper 4 (TCCN) from the JSONs
in sim/results/.

  E1 -- Training-time violation curve (SafeRL vs UnconstrainedPPO),
        mean across seeds with shaded 95% CI, horizontal Gamma line.
  E2 -- Final operating points: two-panel bar plot (violation %, avg power W)
        for all four controllers with bootstrap CIs.
  E4 -- Two-timescale convergence: lambda(u), tau(u), z(u) over training.

E3 (Gamma sweep) is produced by gamma_sweep.py separately.
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "sim/results"
FIG_DIR = "fig"
GAMMA = 3.0  # must match cfg.algo.Gamma

COLORS = {
    "SafeRL":           "#1f77b4",
    "UnconstrainedPPO": "#d62728",
    "LyapunovOnly":     "#2ca02c",
    "Threshold":        "#7f7f7f",
}
LABELS = {
    "SafeRL":           "Safe-RL (proposed)",
    "UnconstrainedPPO": "Unconstrained PPO",
    "LyapunovOnly":     "Lyapunov drift",
    "Threshold":        "Threshold (hysteretic)",
}

plt.rcParams.update({
    "font.size":       9,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize":  9,
    "axes.linewidth":  0.6,
    "lines.linewidth": 1.2,
    "figure.dpi":      150,
})


def _ci_band(ax, x, y_seeds, color, label):
    """y_seeds: (n_seeds, n_steps). Plot mean line + 95% bootstrap band."""
    y_seeds = np.asarray(y_seeds, dtype=float)
    if y_seeds.size == 0:
        return
    mu = y_seeds.mean(axis=0)
    # Smooth the cross-seed mean with a wider moving-average window so that
    # the per-update sampling noise (which is large for safe-RL violation
    # rates) does not create misleading boundary spikes. We use reflect
    # padding so that boundary values are not pulled toward the edge.
    win = 15
    if mu.shape[0] > win:
        pad = win // 2
        padded = np.pad(mu, pad, mode="reflect")
        kernel = np.ones(win) / win
        mu = np.convolve(padded, kernel, mode="valid")
    if y_seeds.shape[0] >= 3:
        lo = np.percentile(y_seeds, 2.5, axis=0)
        hi = np.percentile(y_seeds, 97.5, axis=0)
        if mu.shape[0] == lo.shape[0]:
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mu, color=color, label=label)


def load_training_history(path: str) -> dict:
    """Returns {ctrl_name: {field: (n_seeds, n_updates) array}}."""
    with open(path) as f:
        t = json.load(f)
    out = {}
    for ctrl, seeds in t.items():
        if not isinstance(seeds, dict) or not seeds:
            continue
        # Each seed entry has a 'history' dict with lists.
        field_arrays = {}
        any_key = next(iter(seeds))
        fields = list(seeds[any_key]["history"].keys())
        for f_ in fields:
            arrs = []
            for sk in sorted(seeds.keys()):
                arrs.append(seeds[sk]["history"][f_])
            try:
                field_arrays[f_] = np.array(arrs, dtype=float)
            except Exception:
                continue
        out[ctrl] = field_arrays
    return out


def fig_e1_training_violation(train: dict, path_pdf: str):
    """Per-update violation rate, SafeRL vs UnconstrainedPPO."""
    fig, ax = plt.subplots(figsize=(3.8, 2.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.04)
    for ctrl in ("SafeRL", "UnconstrainedPPO"):
        if ctrl not in train or "viol_rate" not in train[ctrl]:
            continue
        y = train[ctrl]["viol_rate"] * 100.0  # %
        x = np.arange(y.shape[1])
        _ci_band(ax, x, y, COLORS[ctrl], LABELS[ctrl])
    ax.axhline((1.0 - 0.95) * 100, color="k", linestyle=":", linewidth=0.7,
               label=r"$1{-}\beta=5\%$ reference")
    ax.set_xlabel("Training update")
    ax.set_ylabel("Loss-exceedance rate (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"[fig] wrote {path_pdf}")


def fig_e2_final_ops(eval_json: dict, path_pdf: str):
    """Side-by-side bar plot: violation% and avg power across all 4 controllers."""
    ctrls = ["SafeRL", "UnconstrainedPPO", "LyapunovOnly", "Threshold"]
    # Bump font size for this figure only via rc_context.
    with plt.rc_context({
        "font.size":       14,
        "axes.labelsize":  14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                                  constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.02,
                                         wspace=0.06)

        # Panel A: violation rate
        ax = axes[0]
        bars = []
        for i, c in enumerate(ctrls):
            d = eval_json.get(c, {}).get("viol_rate", {})
            if not d:
                continue
            m = d["mean"] * 100.0
            lo = d["lo"] * 100.0
            hi = d["hi"] * 100.0
            bar = ax.bar(i, m, color=COLORS[c],
                         yerr=[[m - lo], [hi - m]], capsize=3,
                         edgecolor="k", linewidth=0.3, width=0.65)
            bars.append(bar)
            ax.text(i, m, f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=12)
        ax.axhline(5.0, color="k", linestyle=":", linewidth=0.7)
        ax.set_xticks(range(len(ctrls)))
        ax.set_xticklabels([LABELS[c].replace(" (proposed)", "")
                            for c in ctrls], rotation=18, ha="right")
        ax.set_ylabel("Loss-exceedance rate (%)")
        ax.set_ylim(0, max(115, ax.get_ylim()[1]))
        ax.grid(True, axis="y", alpha=0.3)

        # Panel B: avg power
        ax = axes[1]
        for i, c in enumerate(ctrls):
            d = eval_json.get(c, {}).get("avg_power_W", {})
            if not d:
                continue
            m = d["mean"]
            lo = d["lo"]
            hi = d["hi"]
            ax.bar(i, m, color=COLORS[c],
                   yerr=[[m - lo], [hi - m]], capsize=3,
                   edgecolor="k", linewidth=0.3, width=0.65)
            ax.text(i, m, f"{m:.0f} W", ha="center", va="bottom",
                    fontsize=12)
        ax.set_xticks(range(len(ctrls)))
        ax.set_xticklabels([LABELS[c].replace(" (proposed)", "")
                            for c in ctrls], rotation=18, ha="right")
        ax.set_ylabel("Average power (W)")
        ax.grid(True, axis="y", alpha=0.3)

        fig.savefig(path_pdf)
    plt.close(fig)
    print(f"[fig] wrote {path_pdf}")


def fig_e4_two_timescale(train: dict, path_pdf: str):
    """lambda(u), tau(u), and risk virtual queue z(u) over training, SafeRL only."""
    sr = train.get("SafeRL", {})
    if not sr or "lambda" not in sr:
        print("[fig] e4: no SafeRL data")
        return
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 3.6),
                             sharex=True, constrained_layout=True)
    x = np.arange(sr["lambda"].shape[1])
    fields = [("lambda", r"$\lambda(u)$  (dual)"),
              ("tau",    r"$\tau(u)$  (VaR estimate)"),
              ("z",      r"$z(u)$  (risk virtual queue)")]
    for ax, (key, ylab) in zip(axes, fields):
        if key not in sr:
            continue
        y = sr[key]
        _ci_band(ax, x, y, COLORS["SafeRL"], "Safe-RL")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Training update")
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"[fig] wrote {path_pdf}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    train = load_training_history(os.path.join(RESULTS_DIR,
                                                "train_safe_rl.json"))
    with open(os.path.join(RESULTS_DIR, "eval_baselines.json")) as f:
        evalj = json.load(f)
    fig_e1_training_violation(train, os.path.join(FIG_DIR, "e1_train_viol.pdf"))
    fig_e2_final_ops(evalj, os.path.join(FIG_DIR, "e2_final_ops.pdf"))
    fig_e4_two_timescale(train, os.path.join(FIG_DIR, "e4_twotimescale.pdf"))


if __name__ == "__main__":
    main()
