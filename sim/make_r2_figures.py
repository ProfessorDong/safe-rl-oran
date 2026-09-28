"""
make_r2_figures.py
------------------
TikZ/pgfplots figures for the second revision, drawn at the 252 pt column
width, generated from sim/results/r2/*.json and written to fig/r2f_*.pdf.
Shares the preamble and helpers of make_tikz_figures.py.

Run:  python3 -m sim.make_r2_figures
"""
from __future__ import annotations
import json
import os

from . import make_tikz_figures as T

R2 = os.environ.get("R2_OUT", os.path.join("sim", "results", "r2"))
LEARNED_FILE = "learned_headline_filter_sens_scale.json"
U = int(os.environ.get("R2_U", 3200))


def _j(name):
    with open(os.path.join(R2, name)) as f:
        return json.load(f)


def _L(d, name, u=None, mode="default"):
    u = U if u is None else u
    return d[f"{name}|u{u}|{mode}"]


def _pt(x, y):
    return "(" + T._num(x) + "," + T._num(y) + ")"


# ------------------------------------------------------------ feasibility
def fig_feasibility():
    f = _j("feasibility.json")
    g0 = f["gamma0"]["mean"]
    gao, lo, hi = (f["gamma_ao"][k] for k in ("mean", "lo", "hi"))
    top = 5.0
    b = [r"\begin{tikzpicture}", r"\begin{axis}[",
         r"  width=210pt, height=44pt,",
         r"  xmin=0, xmax=" + T._num(top) + r", ymin=0, ymax=1,",
         r"  ytick=\empty, axis y line=none, xlabel={risk budget $\Gamma$},",
         r"  xtick={0,1,2,3,4,5}, grid=none, axis on top]"]
    b.append(r"\fill[clearned, opacity=0.28] (axis cs:0,0) rectangle (axis cs:"
             + T._num(g0) + ",1);")
    b.append(r"\fill[cbandlo, opacity=0.45] (axis cs:" + T._num(g0)
             + ",0) rectangle (axis cs:" + T._num(gao) + ",1);")
    b.append(r"\fill[cbandhi, opacity=0.40] (axis cs:" + T._num(gao)
             + ",0) rectangle (axis cs:" + T._num(top) + ",1);")
    b.append(r"\draw[clearned, line width=1.0pt] (axis cs:" + T._num(g0)
             + ",0) -- (axis cs:" + T._num(g0) + ",1);")
    b.append(r"\fill[cclassical, opacity=0.22] (axis cs:" + T._num(lo)
             + ",0) rectangle (axis cs:" + T._num(hi) + ",1);")
    b.append(r"\draw[cclassical, line width=1.0pt] (axis cs:" + T._num(gao)
             + ",0) -- (axis cs:" + T._num(gao) + ",1);")
    b.append(r"\draw[black, densely dotted, line width=1.0pt] (axis cs:3,0)"
             r" -- (axis cs:3,1);")
    b.append(r"\draw[cproposed, dashed, line width=1.0pt] (axis cs:3.5,0)"
             r" -- (axis cs:3.5,1);")
    b.append(r"\node[font=\figlab, align=center] at (axis cs:" + T._num(g0 / 2)
             + r",0.60) {certified\\infeasible};")
    b.append(r"\node[font=\fignote, align=center] at (axis cs:"
             + T._num((g0 + gao) / 2 - 0.12) + r",0.58) {undeter-\\mined};")
    b.append(r"\node[font=\figlab, align=center] at (axis cs:"
             + T._num((gao + top) / 2 + 0.2) + r",0.60) {feasible\\(witness)};")
    b.append(r"\node[font=\figlab, anchor=east, inner sep=1pt] at (axis cs:"
             + T._num(g0 - 0.08) + r",0.14) {$\Gamma_0{=}" + f"{g0:.2f}"
             + r"$};")
    b.append(r"\node[font=\figlab, anchor=west, inner sep=1pt] at (axis cs:"
             + T._num(hi + 0.04) + r",0.14) {$\Gamma_{\mathrm{AO}}{=}"
             + f"{gao:.2f}" + r"$};")
    b.append(r"\node[font=\fignote, anchor=north east] at (axis cs:2.97,0.99)"
             r" {$\Gamma{=}3$};")
    b.append(r"\node[font=\fignote, anchor=north west, text=cproposed] at"
             r" (axis cs:3.55,0.99) {adopted};")
    b += [r"\end{axis}", r"\end{tikzpicture}"]
    T._write_and_compile("r2f_feasibility", "\n".join(b))


# ------------------------------------------------------------ pareto
def fig_pareto():
    c = _j("classical.json")
    d = _j(LEARNED_FILE)
    sel = c["_selection"]

    def cxy(n):
        return c[n]["avg_power_W"]["mean"], c[n]["cvar_beta"]["mean"]

    def lxy(n):
        m = _L(d, n)
        return m["avg_power_W"]["mean"], m["cvar_beta"]["mean"]

    def pts(xy):
        return " ".join(_pt(x, y) for x, y in xy)

    def curve(prefix, keys):
        return pts(sorted(cxy(f"{prefix}:{k}") for k in keys
                          if f"{prefix}:{k}" in c))

    from .r2 import RHO_GRID, QS_GRID, V_GRID
    # The same classical rows as Table II: validation-selected members of each
    # family (c["_headline"] predates validation selection and omits DPP).
    from .make_r2_tables import _closest
    cal = _j("calibration.json")
    heads = ["AlwaysOn", "Threshold",
             _closest(cal["val_grid"], "TextbookDPP")[0],
             _closest(cal["val_grid"], "DriftPlusPenalty")[0],
             "DriftPlusPenalty:1.0", "DriftPlusPenalty:2.0",
             _closest(cal["val_grid"], "SleepAwareDrift")[0]]
    classical = [cxy(n) for n in heads if n in c]
    assert len(classical) == len(heads), heads
    raw = [lxy(n) for n in ("LagPPO", "CRPO", "WCSAC")]
    filt = [lxy(n) for n in ("LagPPO+F", "CRPO+F", "WCSAC+F")]
    prop = [lxy("SafeRL")]
    xs = [p[0] for p in classical + raw + filt + prop]
    ys = [p[1] for p in classical + raw + filt + prop]
    b = [r"\begin{tikzpicture}", r"\begin{groupplot}[",
         r"  group style={group size=2 by 1, horizontal sep=30pt},",
         r"  width=88pt, height=42pt, xlabel={cluster power (W)},",
         r"  ylabel={$\mathrm{CVaR}_{0.95}(\ell)$}, legend columns=2,",
         r"  legend style={at={(0.5,-0.62)}, anchor=north, draw=none,"
         r" fill=none, font=\figtick,"
         r" /tikz/every even column/.append style={column sep=5pt}}]"]
    b.append(r"\nextgroupplot[title={all controllers}, xmin="
             + T._num(min(xs) - 80) + ", xmax=" + T._num(max(xs) + 80)
             + ", ymin=2.6, ymax=10.4, ytick={4,6,8,10}]")
    b.append(r"\addplot[only marks, mark=*, cclassical] coordinates {"
             + pts(classical) + "};")
    b.append(r"\addlegendentry{classical}")
    b.append(r"\addplot[only marks, mark=triangle*, clearned, mark size=1.7pt]"
             r" coordinates {" + pts(raw) + "};")
    b.append(r"\addlegendentry{prior safe RL}")
    b.append(r"\addplot[only marks, mark=triangle, clearned, mark size=1.7pt,"
             r" line width=0.6pt] coordinates {" + pts(filt) + "};")
    b.append(r"\addlegendentry{${+}$filter}")
    b.append(r"\addplot[only marks, mark=star, cproposed, mark size=2.6pt,"
             r" line width=0.7pt] coordinates {" + pts(prop) + "};")
    b.append(r"\addlegendentry{proposed}")
    b.append(r"\addplot[cref, dashed, line width=0.6pt, forget plot] coordinates {"
             + _pt(min(xs) - 80, 3.5) + " " + _pt(max(xs) + 80, 3.5) + "};")
    zoom = [p for p in classical + filt + prop if p[1] < 6.5]
    zx = [p[0] for p in zoom]
    b.append(r"\nextgroupplot[title={vs.\ classical families}, xmin="
             + T._num(min(zx) - 60) + ", xmax=" + T._num(max(zx) + 60)
             + ", ymin=2.95, ymax=6.2, ytick={3,4,5,6}]")
    b.append(r"\addplot[cclassical, mark=*, mark size=0.9pt] coordinates {"
             + curve("DriftPlusPenalty", RHO_GRID) + "};")
    b.append(r"\addlegendentry{backlog-prop., $\rho$}")
    b.append(r"\addplot[clight, mark=square*, mark size=0.9pt] coordinates {"
             + curve("SleepAwareDrift", QS_GRID) + "};")
    b.append(r"\addlegendentry{sleep-aware, $q_s$}")
    b.append(r"\addplot[cref, mark=diamond*, mark size=1.1pt] coordinates {"
             + curve("TextbookDPP", V_GRID) + "};")
    b.append(r"\addlegendentry{textbook DPP, $V$}")
    b.append(r"\addplot[only marks, mark=triangle, clearned, mark size=1.7pt,"
             r" line width=0.6pt] coordinates {" + pts(filt) + "};")
    b.append(r"\addlegendentry{prior RL${+}$filt.}")
    b.append(r"\addplot[only marks, mark=star, cproposed, mark size=2.8pt,"
             r" line width=0.7pt] coordinates {" + pts(prop) + "};")
    b.append(r"\addlegendentry{proposed}")
    b.append(r"\addplot[cref, dashed, line width=0.6pt, forget plot] coordinates {"
             + _pt(min(zx) - 60, 3.5) + " " + _pt(max(zx) + 60, 3.5) + "};")
    b += [r"\end{groupplot}", r"\end{tikzpicture}"]
    T._write_and_compile("r2f_pareto", "\n".join(b))


# ------------------------------------------------------------ filter
def fig_filter():
    d = _j(LEARNED_FILE)
    bars = [("none", _L(d, "SafeRL-none")), ("aggr.", _L(d, "SafeRL-aggr")),
            ("LCB", _L(d, "SafeRL")),
            ("bypass", _L(d, "SafeRL", mode="bypass"))]
    cols = ["cref", "clearned", "cproposed", "clight"]
    b = [r"\begin{tikzpicture}", r"\begin{groupplot}[",
         r"  group style={group size=2 by 1, horizontal sep=30pt},",
         r"  width=88pt, height=45pt, ylabel={$\mathrm{CVaR}_{0.95}(\ell)$}]"]
    b.append(r"\nextgroupplot[title={filter realization}, ybar, bar width=8pt,"
             r" xmin=-0.6, xmax=3.6, ymin=0, ymax=11, xtick={0,1,2,3},"
             r" xticklabels={" + ",".join(n for n, _ in bars) + r"},"
             r" xticklabel style={font=\fignote}, ytick={0,2,4,6,8,10}]")
    for i, ((_, m), col) in enumerate(zip(bars, cols)):
        cv = m["cvar_beta"]
        b.append("\\addplot[fill=" + col + ", draw=" + col + ", bar shift=0pt,"
                 " error bars/.cd, y dir=both, y explicit,"
                 " error bar style={line width=0.4pt, black},"
                 " error mark options={line width=0.4pt, mark size=1pt, black}]\n"
                 "  table[x=x, y=y, y error plus=ep, y error minus=em] {\n"
                 + T._table(["x", "y", "ep", "em"],
                            [(i, cv["mean"], cv["hi"] - cv["mean"],
                              cv["mean"] - cv["lo"])]) + "\n};")
    names = [("SafeRL-b-0.25", -25), ("SafeRL", 0), ("SafeRL-b+0.25", 25),
             ("SafeRL-b+0.5", 50), ("SafeRL-b+1.0", 100)]
    cv = [_L(d, n)["cvar_beta"] for n, _ in names]
    none = _L(d, "SafeRL-none")["cvar_beta"]["mean"]
    ymax = max(max(c["hi"] for c in cv), none) + 0.6
    b.append(r"\nextgroupplot[title={predictor error}, xmin=-35, xmax=110,"
             r" ymin=3.0, ymax=" + T._num(ymax) + r", xtick={-25,0,25,50,100},"
             r" xlabel={service-predictor bias (\%)}]")
    b.append(T._errplot("cproposed, mark=*", [x for _, x in names],
                        [c["mean"] for c in cv], [c["lo"] for c in cv],
                        [c["hi"] for c in cv]))
    b.append(r"\addplot[cref, dashed, line width=0.6pt, forget plot]"
             r" coordinates {" + _pt(-35, none) + " " + _pt(110, none) + "};")
    b.append(r"\node[font=\fignote, text=cref, anchor=north east] at (axis cs:108,"
             + T._num(none) + r") {no filter};")
    b += [r"\end{groupplot}", r"\end{tikzpicture}"]
    T._write_and_compile("r2f_filter", "\n".join(b))


# ------------------------------------------------------------ sensitivity
def fig_sensitivity():
    d = _j(LEARNED_FILE)
    panels = [("beta", [("SafeRL-beta0.90", "0.90"), ("SafeRL", "0.95"),
                        ("SafeRL-beta0.99", "0.99")], r"CVaR level $\beta$"),
              ("kappa", [("SafeRL-kappa0", "0"), ("SafeRL-kappa0.5", "0.5"),
                         ("SafeRL", "1"), ("SafeRL-kappa2", "2")],
               r"LCB conservatism $\kappa$"),
              ("q0", [("SafeRL-q0.25", "0.25"), ("SafeRL", "1"),
                      ("SafeRL-q4", "4")], r"threshold $q_0$ (Mb)"),
              ("lam", [("SafeRL-lam10", "10"), ("SafeRL", "50"),
                       ("SafeRL-lam200", "200")], r"dual cap $\lambda_{\max}$")]
    b = [r"\begin{tikzpicture}", r"\begin{groupplot}[",
         r"  group style={group size=2 by 2, horizontal sep=30pt,"
         r" vertical sep=30pt}, width=82pt, height=25pt]"]
    for idx, (knob, pts, lab) in enumerate(panels):
        xs = list(range(len(pts)))
        c95 = [_L(d, n)["cvar_95"] for n, _ in pts]
        allv = [v for c in c95 for v in (c["lo"], c["hi"])]
        cb = None
        if knob == "beta":
            cb = [_L(d, n)["cvar_beta"] for n, _ in pts]
            allv += [v for c in cb for v in (c["lo"], c["hi"])]
        pad = 0.12 * (max(allv) - min(allv) + 1e-6)
        b.append(r"\nextgroupplot[title={" + lab + r"},")
        if idx % 2 == 0:
            b.append(r"  ylabel={$\mathrm{CVaR}$},")
        b.append(r"  xmin=-0.35, xmax=" + T._num(len(pts) - 0.65)
                 + r", ymin=" + T._num(min(allv) - pad) + r", ymax="
                 + T._num(max(allv) + pad) + r", xtick={"
                 + ",".join(map(str, xs)) + r"}, xticklabels={"
                 + ",".join(v for _, v in pts) + r"},"
                 r" xticklabel style={font=\fignote},"
                 r" yticklabel style={font=\fignote},"
                 r" ylabel style={font=\fignote}]")
        b.append(T._errplot("cproposed, mark=*", xs, [c["mean"] for c in c95],
                            [c["lo"] for c in c95], [c["hi"] for c in c95]))
        if cb:
            b.append(T._errplot("clearned, mark=triangle*, dashed", xs,
                                [c["mean"] for c in cb], [c["lo"] for c in cb],
                                [c["hi"] for c in cb]))
    b += [r"\end{groupplot}", r"\end{tikzpicture}"]
    T._write_and_compile("r2f_sensitivity", "\n".join(b))


# ------------------------------------------------------------ theory panel
def fig_theory():
    d = _j(LEARNED_FILE)
    s = _j("stress.json")
    tm = _j("timing.json")
    us = [u for u in (200, 400, 800, 1600, 3200, U)
          if f"SafeRL|u{u}|default" in d]
    us = sorted(set(us))
    cv = [_L(d, "SafeRL", u=u)["cvar_beta"] for u in us]
    b = [r"\begin{tikzpicture}", r"\begin{groupplot}[",
         r"  group style={group size=3 by 1, horizontal sep=34pt},",
         r"  width=48pt, height=40pt]"]
    b.append(r"\nextgroupplot[title={training budget}, xmode=log,"
             r" xlabel={updates}, ylabel={$\mathrm{CVaR}_{0.95}$},"
             r" xtick={200,800,3200}, xticklabels={200,800,3200}]")
    b.append(T._errplot("cproposed, mark=*", us, [c["mean"] for c in cv],
                        [c["lo"] for c in cv], [c["hi"] for c in cv]))
    b.append(r"\addplot[cref, dashed, forget plot] coordinates {"
             + _pt(150, 3.5) + " " + _pt(4200, 3.5) + "};")
    rates = [0.3, 0.5, 0.7]
    series = [("untrained", "True", "cproposed"),
              ("allsleep", "True", "cclassical"),
              ("untrained", "False", "clearned"),
              ("allsleep", "False", "cref")]
    b.append(r"\nextgroupplot[title={stress test}, ymode=log, xlabel={base rate},"
             r" ylabel={$\bar Q$ (Mb)}, xmin=0.25, xmax=0.75,"
             r" xtick={0.3,0.5,0.7}, ymin=1, ymax=5e4, ytick={1e0,1e2,1e4},"
             r" yminorticks=false,"
             r" legend style={at={(1.02,1.0)}, anchor=north west, draw=none,"
             r" fill=none, font=\fignote}]")
    labels = {("untrained", "True"): "untrained, filter",
              ("allsleep", "True"): "all-sleep, filter",
              ("untrained", "False"): "untrained, none",
              ("allsleep", "False"): "all-sleep, none"}
    for p, f, col in series:
        ys = [max(s[f"rate{r}|{p}|filter{f}"]["avg_backlog_Mb"]["mean"], 1.0)
              for r in rates]
        mark = "*" if f == "True" else "o"
        b.append(r"\addplot[" + col + ", mark=" + mark + "] coordinates {"
                 + " ".join(_pt(r, y) for r, y in zip(rates, ys)) + "};")
    Ks = [7, 19, 37, 61]
    p50 = [tm[f"K={K}"]["total_us"]["p50"] for K in Ks]
    p99 = [tm[f"K={K}"]["total_us"]["p99"] for K in Ks]
    b.append(r"\nextgroupplot[title={per-slot cost}, ymode=log,"
             r" xlabel={cluster size $K$}, ylabel={actor${+}$filter ($\mu$s)},"
             r" xmin=0, xmax=68, ymin=10, ymax=40000, xtick={7,37,61},"
             r" ytick={1e1,1e2,1e3,1e4}, minor tick num=0,"
             r" log basis y=10, yminorticks=false]")
    b.append(r"\addplot[cclassical, mark=square*] coordinates {"
             + " ".join(_pt(k, y) for k, y in zip(Ks, p50)) + "};")
    b.append(r"\addplot[cclassical, mark=square, dashed] coordinates {"
             + " ".join(_pt(k, y) for k, y in zip(Ks, p99)) + "};")
    b.append(r"\addplot[clearned, dashed, forget plot] coordinates"
             r" {(0,10000) (68,10000)};")
    b.append(r"\node[font=\fignote, text=clearned, anchor=south east] at"
             r" (axis cs:66,10000) {10\,ms};")
    b += [r"\end{groupplot}", r"\end{tikzpicture}"]
    T._write_and_compile("r2f_theory", "\n".join(b))


def main():
    print("[r2 figures]")
    fig_feasibility()
    fig_pareto()
    fig_filter()
    fig_sensitivity()
    fig_theory()


if __name__ == "__main__":
    main()
