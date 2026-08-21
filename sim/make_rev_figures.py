"""
make_rev_figures.py
-------------------
Figures for the TCCN major revision, generated from sim/results/rev_*.json.

  r1_feasibility  Feasible/infeasible budget regions and the two floors.
  r2_pareto       Energy/tail-risk frontier: classical vs learned controllers.
  r3_filter       Safety-filter comparison and service-predictor robustness.
  r4_sensitivity  Sensitivity to beta, kappa, q0 and lam_max.
  r5_theory       Corollary 2 V-sweep and Theorem 1 stress test.
  r6_scalability  Per-cell power and per-slot inference time vs cluster size.

Run:  python3 -m sim.make_rev_figures
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

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

C_CLASSICAL = "#2166ac"
C_LEARNED = "#b2182b"
C_PROPOSED = "#1a9850"
C_REF = "#7f7f7f"


def _load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def _save(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------- r1
def fig_feasibility():
    d = _load("rev_feasibility.json")
    g0 = d["floor"]["mean"]
    gmin = d["always_on_cvar"]["mean"]
    gsub = d["gamma_submitted"]
    ghead = d["gamma_headline"]

    fig, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    hi = 5.0
    ax.axvspan(0, g0, color="#b2182b", alpha=0.28)
    ax.axvspan(g0, gmin, color="#f4a582", alpha=0.45)
    ax.axvspan(gmin, hi, color="#92c5de", alpha=0.40)

    ax.axvline(g0, color="#b2182b", lw=1.4)
    ax.axvline(gmin, color="#2166ac", lw=1.4)
    ax.axvline(gsub, color="k", ls=":", lw=1.5)
    ax.axvline(ghead, color=C_PROPOSED, ls="--", lw=1.5)

    ax.text(g0 / 2, 0.72, "infeasible\nfor any\npolicy", ha="center",
            va="center", fontsize=7)
    ax.text((g0 + gmin) / 2, 0.62, "no attainable policy", ha="center",
            va="center", fontsize=6.5, rotation=90)
    ax.text((gmin + hi) / 2, 0.72, "feasible", ha="center", va="center",
            fontsize=8)
    ax.annotate(f"$\\Gamma_0$={g0:.2f}", xy=(g0, 0.30), xytext=(g0 - 0.75, 0.20),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate(f"$\\Gamma_{{\\min}}$={gmin:.2f}", xy=(gmin, 0.42),
                xytext=(gmin + 0.16, 0.34), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("submitted\n$\\Gamma$=3.0", xy=(gsub, 0.10),
                xytext=(gsub - 1.25, 0.04), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("revised\n$\\Gamma$=3.5", xy=(ghead, 0.10),
                xytext=(ghead + 0.22, 0.04), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    ax.set_xlim(0, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(r"risk budget $\Gamma$")
    ax.grid(False)
    _save(fig, "r1_feasibility")


# ---------------------------------------------------------------- r2
CLASSICAL = ["AlwaysOn", "Threshold", "LyapunovOnly", "DriftPlusPenalty",
             "DriftPlusPenalty:2.0", "SleepAwareDrift", "SleepAwareDrift:0.2"]
LEARNED = ["LagrangianPPO", "CRPO", "WCSAC"]
SHORT = {
    "AlwaysOn": "always-on", "Threshold": "threshold",
    "LyapunovOnly": "Lyap. bang-bang",
    "DriftPlusPenalty": r"DPP $\rho$=1", "DriftPlusPenalty:2.0": r"DPP $\rho$=2",
    "SleepAwareDrift": r"sleep-aware $q_s$=0.05",
    "SleepAwareDrift:0.2": r"sleep-aware $q_s$=0.2",
    "LagrangianPPO": "PPO-Lagrangian", "CRPO": "CRPO", "WCSAC": "WCSAC-GS",
    "SafeRL": "proposed",
}


def fig_pareto():
    d = _load("rev_headline.json")

    def xy(k):
        return d[k]["avg_power_W"]["mean"], d[k]["cvar_beta"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)

    ax = axes[0]
    for k in CLASSICAL:
        x, y = xy(k)
        ax.scatter(x, y, s=34, marker="o", color=C_CLASSICAL, zorder=3)
    for k in LEARNED:
        x, y = xy(k)
        ax.scatter(x, y, s=34, marker="^", color=C_LEARNED, zorder=3)
    xp, yp = xy("SafeRL")
    ax.scatter(xp, yp, s=95, marker="*", color=C_PROPOSED, zorder=5,
               edgecolor="k", linewidth=0.4)
    ax.scatter([], [], s=34, marker="o", color=C_CLASSICAL, label="classical")
    ax.scatter([], [], s=34, marker="^", color=C_LEARNED, label="learned (prior)")
    ax.scatter([], [], s=95, marker="*", color=C_PROPOSED,
               edgecolor="k", linewidth=0.4, label="proposed")
    ax.axhline(d["_gamma"], color=C_REF, ls="--", lw=1.0)
    ax.text(1560, d["_gamma"] + 0.16, r"$\Gamma$", color=C_REF, fontsize=8)
    ax.set_xlabel("average cluster power (W)")
    ax.set_ylabel(r"$\mathrm{CVaR}_{0.95}(\ell)$")
    ax.set_title("energy vs tail risk")
    ax.legend(loc="upper right", framealpha=0.9)

    # Zoomed view of the useful region, annotated.
    ax = axes[1]
    keep = ["DriftPlusPenalty", "DriftPlusPenalty:2.0", "SleepAwareDrift",
            "SleepAwareDrift:0.2", "LyapunovOnly", "AlwaysOn", "SafeRL"]
    for k in keep:
        x, y = xy(k)
        col = C_PROPOSED if k == "SafeRL" else C_CLASSICAL
        mk = "*" if k == "SafeRL" else "o"
        sz = 110 if k == "SafeRL" else 34
        ax.scatter(x, y, s=sz, marker=mk, color=col, zorder=4,
                   edgecolor="k" if k == "SafeRL" else "none", linewidth=0.4)
        off = {"LyapunovOnly": (-58, 6), "SafeRL": (6, -11),
               "SleepAwareDrift": (5, -10), "SleepAwareDrift:0.2": (5, 4),
               "DriftPlusPenalty": (-16, 6), "DriftPlusPenalty:2.0": (-14, -12),
               "AlwaysOn": (-38, 6)}.get(k, (3, 4))
        ax.annotate(SHORT[k], xy=(x, y), xytext=off,
                    textcoords="offset points", fontsize=6.5)
    ax.axhline(d["_gamma"], color=C_REF, ls="--", lw=1.0)
    ax.set_xlabel("average cluster power (W)")
    ax.set_ylabel(r"$\mathrm{CVaR}_{0.95}(\ell)$")
    ax.set_title("detail (toggling not shown)")
    ax.set_xlim(880, 1700)
    ax.set_ylim(2.8, 5.6)
    _save(fig, "r2_pareto")


# ---------------------------------------------------------------- r3
def fig_filter():
    d = _load("rev_filter.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)

    ax = axes[0]
    kinds = [("none_bias0.0", "no filter"),
             ("aggregate_bias0.0", "aggregate\n(as submitted)"),
             ("lcb_bias0.0", "LCB\n(as specified)")]
    vals = [d[k]["cvar_beta"]["mean"] for k, _ in kinds]
    err = [[v - d[k]["cvar_beta"]["lo"] for v, (k, _) in zip(vals, kinds)],
           [d[k]["cvar_beta"]["hi"] - v for v, (k, _) in zip(vals, kinds)]]
    cols = [C_REF, C_LEARNED, C_PROPOSED]
    ax.bar(range(3), vals, yerr=err, capsize=3, color=cols, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lb for _, lb in kinds], fontsize=7.5)
    ax.set_ylabel(r"$\mathrm{CVaR}_{0.95}(\ell)$")
    ax.set_title("safety-filter realization")

    ax = axes[1]
    biases = [-0.25, 0.0, 0.25, 0.5, 1.0]
    keys = [f"lcb_bias{b}" for b in biases]
    cv = [d[k]["cvar_beta"]["mean"] for k in keys]
    lo = [d[k]["cvar_beta"]["lo"] for k in keys]
    hi = [d[k]["cvar_beta"]["hi"] for k in keys]
    ax.errorbar([b * 100 for b in biases], cv,
                yerr=[np.array(cv) - np.array(lo), np.array(hi) - np.array(cv)],
                marker="o", ms=4, lw=1.3, capsize=3, color=C_PROPOSED)
    ax.axhline(d["aggregate_bias0.0"]["cvar_beta"]["mean"], color=C_LEARNED,
               ls="--", lw=1.0, label="aggregate filter")
    ax.set_xlabel("service-predictor bias (%)")
    ax.set_ylabel(r"$\mathrm{CVaR}_{0.95}(\ell)$")
    ax.set_title("robustness to predictor error")
    ax.legend(loc="upper left")
    _save(fig, "r3_filter")


# ---------------------------------------------------------------- r4
def fig_sensitivity():
    d = _load("rev_sensitivity.json")
    panels = [("beta", [0.9, 0.95, 0.99], r"CVaR level $\beta$"),
              ("lcb_kappa", [0.0, 1.0, 2.0], r"LCB conservatism $\kappa$"),
              ("q0_cell_Mb", [0.25, 1.0, 4.0], r"filter threshold $q_0$ (Mb)"),
              ("lam_max", [10.0, 50.0, 200.0], r"dual cap $\lambda_{\max}$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), constrained_layout=True)
    for ax, (knob, vals, lab) in zip(axes, panels):
        cv = [d[f"{knob}={v}"]["cvar_beta"]["mean"] for v in vals]
        lo = [d[f"{knob}={v}"]["cvar_beta"]["lo"] for v in vals]
        hi = [d[f"{knob}={v}"]["cvar_beta"]["hi"] for v in vals]
        pw = [d[f"{knob}={v}"]["avg_power_W"]["mean"] for v in vals]
        x = np.arange(len(vals))
        ax.errorbar(x, cv, yerr=[np.array(cv) - np.array(lo),
                                 np.array(hi) - np.array(cv)],
                    marker="o", ms=4, lw=1.3, capsize=3, color=C_PROPOSED)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], fontsize=7.5)
        ax.set_xlabel(lab, fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(x, pw, marker="s", ms=3, lw=1.0, ls=":", color=C_CLASSICAL)
        ax2.tick_params(axis="y", labelsize=6.5, colors=C_CLASSICAL)
        ax2.grid(False)
        if ax is axes[0]:
            ax.set_ylabel(r"$\mathrm{CVaR}_{0.95}$")
        if ax is axes[-1]:
            ax2.set_ylabel("power (W)", fontsize=7, color=C_CLASSICAL)
    _save(fig, "r4_sensitivity")


# ---------------------------------------------------------------- r5
def fig_theory():
    v = _load("rev_vsweep.json")
    s = _load("rev_stress.json")
    sc = _load("rev_scalability.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35),
                             constrained_layout=True)

    ax = axes[0]
    Vs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    qb = [v[f"V={V}"]["backlog"]["mean"] for V in Vs]
    lo = [v[f"V={V}"]["backlog"]["lo"] for V in Vs]
    hi = [v[f"V={V}"]["backlog"]["hi"] for V in Vs]
    ax.errorbar(Vs, qb, yerr=[np.array(qb) - np.array(lo),
                              np.array(hi) - np.array(qb)],
                marker="o", ms=4, lw=1.4, capsize=3, color=C_PROPOSED,
                label=r"measured $\bar Q$")
    neely = [qb[0] * (V / Vs[0]) for V in Vs]
    ax.plot(Vs, neely, ls="--", lw=1.2, color=C_REF,
            label=r"canonical Neely $O(V)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lyapunov weight $V$")
    ax.set_ylabel(r"time-average backlog $\bar Q$ (Mb)")
    ax.set_title("Corollary 2")
    ax.legend(loc="upper left")

    ax = axes[1]
    rates = [0.3, 0.5, 0.7]
    on = [s[f"rate{r}_filterTrue"]["backlog"]["mean"] for r in rates]
    off = [s[f"rate{r}_filterFalse"]["backlog"]["mean"] for r in rates]
    x = np.arange(len(rates))
    ax.bar(x - 0.18, on, 0.36, label="filter on", color=C_PROPOSED)
    ax.bar(x + 0.18, off, 0.36, label="filter off", color=C_LEARNED)
    for i, (a, b) in enumerate(zip(on, off)):
        ax.text(i + 0.18, b * 1.25, f"{b/max(a,1e-9):.0f}$\\times$",
                ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}" for r in rates])
    ax.set_xlabel("per-cell arrival rate (Mb/slot)")
    ax.set_ylabel(r"$\bar Q$ (Mb)")
    ax.set_title("Theorem 1 stress test")
    ax.legend(loc="upper left")

    # Third panel: near-real-time timing margin against cluster size.
    ax = axes[2]
    Ks = [7, 19, 37, 61]
    it = [sc[f"K={K}"]["infer_us_per_slot"]["mean"] for K in Ks]
    ax.plot(Ks, it, marker="s", ms=4, lw=1.3, color=C_CLASSICAL)
    ax.axhline(10000, color=C_LEARNED, ls="--", lw=1.2)
    ax.text(9, 12000, "10 ms budget", color=C_LEARNED, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(100, 40000)
    ax.set_xlabel("cluster size $K$")
    ax.set_ylabel(r"inference ($\mu$s/slot)")
    ax.set_title("near-RT margin")
    _save(fig, "r5_theory")


# ---------------------------------------------------------------- r6
def fig_scalability():
    d = _load("rev_scalability.json")
    Ks = [7, 19, 37, 61]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4), constrained_layout=True)

    ax = axes[0]
    pc = [d[f"K={K}"]["power_per_cell"]["mean"] for K in Ks]
    lo = [d[f"K={K}"]["power_per_cell"]["lo"] for K in Ks]
    hi = [d[f"K={K}"]["power_per_cell"]["hi"] for K in Ks]
    ax.errorbar(Ks, pc, yerr=[np.array(pc) - np.array(lo),
                              np.array(hi) - np.array(pc)],
                marker="o", ms=4, lw=1.3, capsize=3, color=C_PROPOSED)
    ax.set_xlabel("cluster size $K$")
    ax.set_ylabel("power per cell (W)")
    ax.set_title("per-cell energy")

    ax = axes[1]
    it = [d[f"K={K}"]["infer_us_per_slot"]["mean"] for K in Ks]
    ax.plot(Ks, it, marker="s", ms=4, lw=1.3, color=C_CLASSICAL)
    ax.axhline(10000, color=C_LEARNED, ls="--", lw=1.2)
    ax.text(20, 11500, "10 ms near-RT budget", color=C_LEARNED, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(100, 40000)
    ax.set_xlabel("cluster size $K$")
    ax.set_ylabel(r"inference ($\mu$s/slot)")
    ax.set_title("near-RT timing margin")
    _save(fig, "r6_scalability")


def main():
    print("[figures] revision set")
    fig_feasibility()
    fig_pareto()
    fig_filter()
    fig_sensitivity()
    fig_theory()


if __name__ == "__main__":
    main()
