"""
make_rev_tables.py
------------------
Emit the LaTeX tables for the revision directly from sim/results/rev_*.json,
so every number in the manuscript traces to a stored experiment.

  tab/rev_headline.tex     controller comparison (Table I)
  tab/rev_theory_impl.tex  analysis / implementation correspondence

Run:  python3 -m sim.make_rev_tables
"""
from __future__ import annotations
import json
import os

RESULTS_DIR = "sim/results"
TAB_DIR = "tab"

ORDER = [
    ("AlwaysOn", "Always-on"),
    ("Threshold", "Threshold heuristic"),
    ("LyapunovOnly", "Lyapunov drift, bang-bang"),
    ("DriftPlusPenalty", r"Drift-plus-penalty, $\rho\!=\!1$"),
    ("DriftPlusPenalty:2.0", r"Drift-plus-penalty, $\rho\!=\!2$"),
    ("SleepAwareDrift", r"Sleep-aware drift, $q_s\!=\!0.05$"),
    ("SleepAwareDrift:0.2", r"Sleep-aware drift, $q_s\!=\!0.2$"),
    ("LagrangianPPO", "PPO-Lagrangian (expected cost)"),
    ("CRPO", "CRPO~\\cite{Xu_CRPO_ICML2021}"),
    ("WCSAC", "WCSAC-GS~\\cite{Yang_WCSAC_ML2023}"),
    ("LagrangianPPO+filter", "PPO-Lagrangian + LCB filter"),
    ("CRPO+filter", "CRPO + LCB filter"),
    ("WCSAC+filter", "WCSAC-GS + LCB filter"),
    ("SafeRL", r"\textbf{Proposed}"),
]


def _fmt(v, prec=1):
    return f"{v:.{prec}f}"


def headline():
    d = json.load(open(os.path.join(RESULTS_DIR, "rev_headline.json")))
    gamma = d["_gamma"]
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Controller comparison on the seven-cell cluster at "
        rf"$\Gamma={gamma}$, ten canonical seeds. Subscripts are $95\%$ "
        r"bootstrap confidence intervals. Rows above the rule are "
        r"non-learning controllers.}",
        r"  \label{tab:headline}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{8pt}",
        r"  \begin{tabular}{@{}lccccc@{}}",
        r"    \toprule",
        r"    Controller & $P$ (W) & $\CVaR_{\beta}$ & $\Pr\{\ell{>}\Gamma\}$"
        r" & $p_{99}$ (ms) & tog/min \\",
        r"    \midrule",
    ]
    for key, label in ORDER:
        a = d[key]
        if key == "LagrangianPPO":
            lines.append(r"    \midrule")
        p, c = a["avg_power_W"], a["cvar_beta"]
        v = a["viol_rate"]
        p99 = a["p99_delay_ms"]["mean"]
        tog = a["toggles_per_min"]["mean"]
        # Rendered inside an outer $...$, so no inner math delimiters here.
        p99s = (f"{p99:.0f}" if p99 < 1000
                else f"{p99/1000:.1f}\\!\\times\\!10^{{3}}")
        row = (f"    {label} & "
               f"${_fmt(p['mean'])}_{{[{_fmt(p['lo'],0)},{_fmt(p['hi'],0)}]}}$ & "
               f"${_fmt(c['mean'],3)}_{{[{_fmt(c['lo'],2)},{_fmt(c['hi'],2)}]}}$ & "
               f"${_fmt(v['mean']*100,1)}\\%$ & "
               f"${p99s}$ & "
               f"${_fmt(tog,0)}$ \\\\")
        lines.append(row)
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    _write("rev_headline.tex", lines)


THEORY_IMPL = [
    ("Step sizes", "diminishing", "constant"),
    ("Dual update", r"integral, \eqref{eq:dual_update}",
     r"PID, \eqref{eq:pid_dual}"),
    ("Dual cap", "none", r"$\lambda_{\max}\!=\!50$"),
    ("Critics", "single, augmented cost", "separate energy and risk"),
    ("Conditioning", r"$X(t)$", r"$X(t),\lambda,\tau$"),
    ("Safety filter", r"LCB projection, \eqref{eq:safe_set}",
     "LCB projection, per cell"),
    ("Advantage", "exact differential", r"GAE-$\lambda$"),
    # Assumption 4 puts tau on the slowest timescale; the implementation
    # updates it every slot while actor and dual update per rollout.
    ("Timescale order", r"critic $\gg$ actor $\gg$ dual,\,$\tau$",
     r"$\tau$ per slot, others per rollout"),
    ("Projection", r"smoothed $\Pi_\rho$", "hard"),
    ("CVaR result", r"$\le\Gamma$ (conditional)", None),   # filled from results
]


def theory_impl():
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Correspondence between the analysis of "
        r"Section~\ref{sec:theory} and the implementation evaluated in "
        r"Section~\ref{sec:eval}.}",
        r"  \label{tab:theory_impl}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{@{}lll@{}}",
        r"    \toprule",
        r"    Quantity & Analysis & Implementation \\",
        r"    \midrule",
    ]
    d = json.load(open(os.path.join(RESULTS_DIR, "rev_headline.json")))
    cv = d["SafeRL"]["cvar_beta"]["mean"]
    gam = d["_gamma"]
    rows = [(q, t, (i if i is not None
                    else rf"${cv:.3f}$ at $\Gamma\!=\!{gam}$"))
            for q, t, i in THEORY_IMPL]
    for q, t, i in rows:
        lines.append(f"    {q} & {t} & {i} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    _write("rev_theory_impl.tex", lines)


def _write(name, lines):
    os.makedirs(TAB_DIR, exist_ok=True)
    path = os.path.join(TAB_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


def main():
    print("[tables] revision set")
    headline()
    theory_impl()


if __name__ == "__main__":
    main()
