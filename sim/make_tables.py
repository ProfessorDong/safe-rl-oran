"""
make_tables.py
--------------
Generate LaTeX tables for Paper 4 from sim/results/eval_baselines.json.

  T1 -- Headline comparison: avg power, CVaR, viol%, p95/p99 delay, toggles/min
        across all four controllers, with bootstrap 95% CIs.
"""
from __future__ import annotations
import json
import os

RESULTS_DIR = "sim/results"
TAB_DIR = "tab"

CTRLS = ["SafeRL", "UnconstrainedPPO", "LyapunovOnly", "Threshold"]
CTRL_LABEL = {
    "SafeRL":           r"Safe-RL (proposed)",
    "UnconstrainedPPO": r"Unconstrained PPO",
    "LyapunovOnly":     r"Lyapunov drift",
    "Threshold":        r"Threshold heuristic",
}


def fmt(mci, scale=1.0, prec=1, pct=False):
    """Format a {mean,lo,hi} block with a [lo, hi] CI."""
    if not mci:
        return r"---"
    m = mci.get("mean", 0.0) * scale
    lo = mci.get("lo", 0.0) * scale
    hi = mci.get("hi", 0.0) * scale
    unit = "" if not pct else r"\%"
    return f"${m:.{prec}f}_{{[{lo:.{prec}f},\\,{hi:.{prec}f}]}}${unit}"


def table_headline(evalj: dict, path_tex: str):
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"  \centering")
    rows.append(r"  \caption{Headline comparison on 10 evaluation seeds "
                r"(100\,s episodes, $\beta=0.95$, $\Gamma=3.0$). Subscripts "
                r"are 95\% bootstrap CIs. Lower is better for all columns.}")
    rows.append(r"  \label{tab:headline}")
    rows.append(r"  \small")
    rows.append(r"  \begin{tabular}{lccccc}")
    rows.append(r"    \toprule")
    rows.append(r"    Controller & Avg.\ power (W) & CVaR$_{\beta}$ "
                r"& Viol.\ rate (\%) & $p99$ delay (ms) & Toggles/min \\")
    rows.append(r"    \midrule")
    for c in CTRLS:
        d = evalj.get(c, {})
        if not d:
            continue
        rows.append(
            f"    {CTRL_LABEL[c]} & "
            f"{fmt(d.get('avg_power_W', {}), prec=1)} & "
            f"{fmt(d.get('cvar_beta', {}), prec=2)} & "
            f"{fmt(d.get('viol_rate', {}), scale=100, prec=1, pct=False)} & "
            f"{fmt(d.get('p99_delay_ms', {}), prec=0)} & "
            f"{fmt(d.get('toggles_per_min', {}), prec=1)} \\\\"
        )
    rows.append(r"    \bottomrule")
    rows.append(r"  \end{tabular}")
    rows.append(r"\end{table*}")

    with open(path_tex, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[tab] wrote {path_tex}")


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eval_baselines.json")) as f:
        evalj = json.load(f)
    table_headline(evalj, os.path.join(TAB_DIR, "headline.tex"))


if __name__ == "__main__":
    main()
