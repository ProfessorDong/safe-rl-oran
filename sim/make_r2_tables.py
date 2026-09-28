"""
make_r2_tables.py
-----------------
LaTeX tables and a paired-statistics summary for the second revision.

Reads sim/results/r2/*.json and the saved per-slot series, writes
tab/r2_headline.tex and sim/results/r2/analysis.json (every number the text
quotes that is not read directly off a table).

Run:  python3 -m sim.make_r2_tables
"""
from __future__ import annotations
import json
import os

import numpy as np

from .metrics import bootstrap_ci

R2 = os.environ.get("R2_OUT", os.path.join("sim", "results", "r2"))
SERIES = os.path.join(R2, "series")
LEARNED_FILE = "learned_headline_filter_sens_scale.json"
U = int(os.environ.get("R2_U", 3200))


def _j(name):
    with open(os.path.join(R2, name)) as f:
        return json.load(f)


def _closest(val_grid, prefix, gamma=3.5):
    """Validation-selected member of a family: the cheapest meeting the
    budget on validation, or, if none does, the one closest to it."""
    ks = [k for k in val_grid if k.startswith(prefix + ":")]
    ok = [k for k in ks if val_grid[k]["cvar_beta"]["mean"] <= gamma]
    if ok:
        return min(ok, key=lambda k: val_grid[k]["avg_power_W"]["mean"]), True
    return min(ks, key=lambda k: val_grid[k]["cvar_beta"]["mean"]), False


def _fmt_ci(m, digits, scale=1.0):
    f = f"{{:.{digits}f}}"
    lo, hi = m["lo"] * scale, m["hi"] * scale
    d2 = max(digits - 1, 0) if digits > 1 else 0
    g = f"{{:.{d2}f}}"
    return f"${f.format(m['mean'] * scale)}_{{[{g.format(lo)},{g.format(hi)}]}}$"


def _p99(v):
    return (f"${v/1000:.1f}\\!\\times\\!10^{{3}}$" if v >= 1000 else f"${v:.0f}$")


def _row(label, m):
    return (f"    {label} & {_fmt_ci(m['avg_power_W'], 1)} & "
            f"{_fmt_ci(m['cvar_95'], 3)} & "
            f"${100*m['viol_rate']['mean']:.1f}\\%$ & "
            f"{_p99(m['p99_delay_ms']['mean'])} & "
            f"${m['toggles_per_min']['mean']:.0f}$ \\\\")


def _series(tag, seed):
    p = os.path.join(SERIES, f"{tag}_{seed}.npz")
    return np.load(p) if os.path.exists(p) else None


def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    ci = bootstrap_ci(d)
    return dict(mean=ci["mean"], lo=ci["lo"], hi=ci["hi"], per_seed=d)


def main():
    cal = _j("calibration.json")
    c = _j("classical.json")
    d = _j(LEARNED_FILE)
    f = _j("feasibility.json")
    L = lambda n, mode="default", u=U: d[f"{n}|u{u}|{mode}"]

    sel_rho, ok_rho = _closest(cal["val_grid"], "DriftPlusPenalty")
    sel_qs, ok_qs = _closest(cal["val_grid"], "SleepAwareDrift")
    sel_V, ok_V = _closest(cal["val_grid"], "TextbookDPP")

    def lab_rho(k, dag=False):
        return (f"Backlog-proportional, $\\rho\\!=\\!{float(k.split(':')[1]):g}"
                + ("^\\dagger$" if dag else "$"))

    rows = [("Always-on", c["AlwaysOn"]),
            ("Threshold heuristic", c["Threshold"]),
            (f"Textbook DPP, $V\\!=\\!{{{_sci(sel_V)}}}" + ("^\\dagger$" if ok_V else "$"),
             c[sel_V]),
            (lab_rho(sel_rho, dag=True), c[sel_rho]),
            (lab_rho("x:1.0"), c["DriftPlusPenalty:1.0"]),
            (lab_rho("x:2.0"), c["DriftPlusPenalty:2.0"]),
            (f"Sleep-aware, $q_s\\!=\\!{float(sel_qs.split(':')[1]):g}$",
             c[sel_qs])]
    learned = [("PPO-Lagrangian (expected cost)", L("LagPPO")),
               ("CRPO~\\cite{Xu_CRPO_ICML2021}", L("CRPO")),
               ("WCSAC-GS~\\cite{Yang_WCSAC_ML2023}, PPO adaptation", L("WCSAC")),
               ("PPO-Lagrangian + LCB filter", L("LagPPO+F")),
               ("CRPO + LCB filter", L("CRPO+F")),
               ("WCSAC-GS + LCB filter", L("WCSAC+F")),
               ("\\textbf{Proposed}", L("SafeRL"))]
    lines = [r"\begin{table*}[t]", r"  \centering",
             r"  \caption{Controller comparison on the seven-cell cluster at "
             r"$\Gamma=3.5$, ten test seeds (evaluation traces disjoint from "
             r"training and from the validation seeds used for tuning). "
             r"Subscripts are $95\%$ bootstrap intervals over seeds of the "
             r"per-episode value. $p_{99}$ is the backlog-based delay proxy. "
             r"$^\dagger$Selected on validation seeds as the lowest-power "
             r"setting with mean $\mathrm{CVaR}_{0.95}\le\Gamma$; "
             + ("the sleep-aware row is the setting closest to the budget, "
                "since none meets it. " if not ok_qs else "")
             + r"Rows above the learned methods are non-learning "
             r"controllers.}",
             r"  \label{tab:headline}", r"  \footnotesize",
             r"  \setlength{\tabcolsep}{7pt}",
             r"  \begin{tabular}{@{}lccccc@{}}", r"    \toprule",
             r"    Controller & $P$ (W) & $\CVaR_{0.95}$ & "
             r"$\Pr\{\ell{>}\Gamma\}$ & $p_{99}$ (ms) & tog/min \\",
             r"    \midrule"]
    lines += [_row(a, m) for a, m in rows]
    lines.append(r"    \midrule")
    lines += [_row(a, m) for a, m in learned]
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    os.makedirs("tab", exist_ok=True)
    with open(os.path.join("tab", "r2_headline.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  -> tab/r2_headline.tex")

    # ------------------------------------------------ paired statistics
    P = L("SafeRL")
    an = dict(selection=dict(rho=sel_rho, rho_ok=ok_rho, qs=sel_qs,
                             qs_ok=ok_qs, V=sel_V, V_ok=ok_V))
    for name, m in (("rho_sel", c[sel_rho]), ("rho1", c["DriftPlusPenalty:1.0"]),
                    ("dpp_sel", c[sel_V]),
                    ("wcsac_f", L("WCSAC+F")), ("crpo_f", L("CRPO+F")),
                    ("lagppo_f", L("LagPPO+F"))):
        an[f"proposed_minus_{name}"] = dict(
            power=_paired(P["avg_power_W"]["per_seed"],
                          m["avg_power_W"]["per_seed"]),
            cvar=_paired(P["cvar_95"]["per_seed"], m["cvar_95"]["per_seed"]),
            p99=_paired(P["p99_delay_ms"]["per_seed"],
                        m["p99_delay_ms"]["per_seed"]))
    B = L("SafeRL", "bypass")
    an["filter_on_minus_bypass"] = dict(
        power=_paired(P["avg_power_W"]["per_seed"], B["avg_power_W"]["per_seed"]),
        cvar=_paired(P["cvar_95"]["per_seed"], B["cvar_95"]["per_seed"]),
        share_prop_on=P.get("share_prop_mean"),
        share_exec_on=P.get("share_exec_mean"))
    an["share_exec"] = {"proposed": P["share_exec_mean"]["mean"],
                        "rho_sel": c[sel_rho]["share_exec_mean"]["mean"]}
    # Budget met with confidence? One-sided: upper 95% bootstrap bound.
    an["budget_check"] = {k: dict(mean=m["cvar_95"]["mean"],
                                  hi=m["cvar_95"]["hi"])
                          for k, m in (("proposed", P), ("rho_sel", c[sel_rho]))}
    # Always-on as the slotwise loss minimizer, from the saved series.
    seeds = f["_provenance"]["test_seeds"]
    frac = {}
    for tag in ("SafeRL_u%d_default" % U, sel_rho.replace(":", "_"),
                sel_V.replace(":", "_"),
                "LagPPO+F_u%d_default" % U, "CRPO+F_u%d_default" % U,
                "WCSAC+F_u%d_default" % U):
        vals = []
        for s in seeds:
            a, b = _series("AlwaysOn", s), _series(tag, s)
            if a is not None and b is not None:
                vals.append(float(np.mean(a["loss"] <= b["loss"] + 1e-9)))
        if vals:
            frac[tag] = float(np.mean(vals))
    an["ao_loss_le_controller_frac"] = frac
    with open(os.path.join(R2, "analysis.json"), "w") as fh:
        json.dump(an, fh, indent=2)
    print(f"  -> {R2}/analysis.json")
    print(json.dumps({k: v for k, v in an.items() if k != "selection"},
                     indent=1)[:3000])


def _sci(k):
    v = float(k.split(":")[1])
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    return (f"10^{{{e}}}" if abs(m - 1) < 1e-9 else f"{m:g}\\!\\cdot\\!10^{{{e}}}")


if __name__ == "__main__":
    main()
