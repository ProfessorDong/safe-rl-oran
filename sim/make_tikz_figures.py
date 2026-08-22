"""
make_tikz_figures.py
--------------------
TikZ/pgfplots versions of the revision figures, generated from the same
sim/results/rev_*.json files as the matplotlib set and written to fig/*.pdf
under the same names, so main.tex needs no change.

Why TikZ rather than matplotlib. The matplotlib figures were drawn 7.1 in
wide and then scaled into a 252 pt (3.5 in) IEEE column by
\\includegraphics[width=\\linewidth], so their 9 pt labels reached the page at
about 4.4 pt, well under the size IEEE expects in artwork. These are drawn at
the final column width instead, so a 7 pt tick label is 7 pt on the page, and
they are typeset with newtx (Times) so figure text matches the body text in
both face and size. Everything is vector, and every axis in the set shares one
style block, so the figures are visually uniform.

Run:  python3 -m sim.make_tikz_figures
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile

RESULTS_DIR = "sim/results"
FIG_DIR = "fig"
TEX_DIR = os.path.join(FIG_DIR, "tikz")

# Same palette as the matplotlib set, so the two remain comparable.
C_CLASSICAL = "2166ac"
C_LEARNED = "b2182b"
C_PROPOSED = "1a9850"
C_REF = "7f7f7f"
C_LIGHT = "67a9cf"
C_BAND_LO = "f4a582"
C_BAND_HI = "92c5de"

PREAMBLE = r"""
\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{newtxmath}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta}
\pgfplotsset{compat=1.18}

\definecolor{cclassical}{HTML}{2166ac}
\definecolor{clearned}{HTML}{b2182b}
\definecolor{cproposed}{HTML}{1a9850}
\definecolor{cref}{HTML}{7f7f7f}
\definecolor{clight}{HTML}{67a9cf}
\definecolor{cbandlo}{HTML}{f4a582}
\definecolor{cbandhi}{HTML}{92c5de}

% Figure text is sized for a 252 pt column: axis labels a touch under the
% 8 pt caption size, tick labels one step below that.
\newcommand{\figlab}{\fontsize{7.5}{8.5}\selectfont}
\newcommand{\figtick}{\fontsize{6.5}{7.5}\selectfont}
\newcommand{\fignote}{\fontsize{6}{7}\selectfont}

\pgfplotsset{
  every axis/.append style={
    line width=0.5pt,
    scale only axis,
    tick style={line width=0.4pt, black},
    tick pos=left,
    grid=major,
    grid style={gray!30, line width=0.3pt},
    label style={font=\figlab},
    tick label style={font=\figtick},
    title style={font=\figlab, yshift=-2pt},
    legend cell align=left,
    legend style={
      font=\fignote, draw=black!35, fill=white, fill opacity=0.92,
      text opacity=1, inner sep=1.2pt, row sep=-1.5pt, legend image post style={scale=0.7}},
  },
  every axis plot/.append style={line width=0.7pt, mark size=1.3pt},
}
\begin{document}
"""

POSTAMBLE = "\n\\end{document}\n"


def _load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def _num(v):
    return f"{v:.6g}"


def _table(cols, rows):
    """Inline pgfplots table: `cols` header names, `rows` list of tuples."""
    head = " ".join(cols)
    body = "\n".join(" ".join(_num(v) for v in r) for r in rows)
    return head + "\n" + body


def _errplot(style, xs, ys, los, his, xcol_symbolic=False):
    """Asymmetric-error-bar line plot as an inline table."""
    rows = [(x, y, h - y, y - l) for x, y, l, h in zip(xs, ys, los, his)]
    tbl = _table(["x", "y", "ep", "em"], rows)
    return (
        "\\addplot[" + style + ", error bars/.cd, y dir=both, y explicit,\n"
        "  error bar style={line width=0.4pt}, error mark options={line width=0.4pt, mark size=1pt}]\n"
        "  table[x=x, y=y, y error plus=ep, y error minus=em] {\n" + tbl + "\n};\n")


def _write_and_compile(stem, body):
    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    tex_path = os.path.join(TEX_DIR, f"{stem}.tex")
    with open(tex_path, "w") as f:
        f.write(PREAMBLE + body + POSTAMBLE)
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", td, tex_path],
            capture_output=True, text=True)
        out = os.path.join(td, f"{stem}.pdf")
        if r.returncode != 0 or not os.path.exists(out):
            tail = "\n".join(r.stdout.strip().splitlines()[-25:])
            raise RuntimeError(f"pdflatex failed for {stem}:\n{tail}")
        shutil.copy(out, os.path.join(FIG_DIR, f"{stem}.pdf"))
    # Report the natural width so it can be checked against the 252 pt column.
    try:
        info = subprocess.run(["pdfinfo", os.path.join(FIG_DIR, f"{stem}.pdf")],
                              capture_output=True, text=True).stdout
        size = [l for l in info.splitlines() if l.startswith("Page size")]
        note = size[0].split(":")[1].strip() if size else ""
    except Exception:
        note = ""
    print(f"  -> {FIG_DIR}/{stem}.pdf   {note}")


# ================================================================== r1
def fig_feasibility():
    d = _load("rev_feasibility.json")
    g0 = d["floor"]["mean"]
    gmin = d["always_on_cvar"]["mean"]
    gmin_lo = d["always_on_cvar"]["lo"]
    gmin_hi = d["always_on_cvar"]["hi"]
    gsub = d["gamma_submitted"]
    ghead = d["gamma_headline"]
    hi = 5.0

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{axis}[")
    b.append(r"  width=210pt, height=44pt,")
    b.append(r"  xmin=0, xmax=" + _num(hi) + r", ymin=0, ymax=1,")
    b.append(r"  ytick=\empty, axis y line=none,")
    b.append(r"  xlabel={risk budget $\Gamma$},")
    b.append(r"  xtick={0,1,2,3,4,5}, grid=none,")
    b.append(r"  axis on top,")
    b.append(r"]")
    # Three regions.
    b.append(r"\fill[clearned, opacity=0.28] (axis cs:0,0) rectangle (axis cs:"
             + _num(g0) + ",1);")
    b.append(r"\fill[cbandlo, opacity=0.45] (axis cs:" + _num(g0)
             + ",0) rectangle (axis cs:" + _num(gmin) + ",1);")
    b.append(r"\fill[cbandhi, opacity=0.40] (axis cs:" + _num(gmin)
             + ",0) rectangle (axis cs:" + _num(hi) + ",1);")
    # Boundaries and the two budgets.
    b.append(r"\draw[clearned, line width=1.0pt] (axis cs:" + _num(g0)
             + ",0) -- (axis cs:" + _num(g0) + ",1);")
    # The attainable floor is an estimate, and its 95% interval straddles the
    # 3.0 budget. Drawing the interval keeps the figure from asserting a
    # sharper separation than ten traces support, and it is the only way to
    # make the 3.0-versus-floor relation visible at all: the two markers are
    # 0.05 apart, about 2pt on this axis.
    b.append(r"\fill[cclassical, opacity=0.22] (axis cs:" + _num(gmin_lo)
             + ",0) rectangle (axis cs:" + _num(gmin_hi) + ",1);")
    b.append(r"\draw[cclassical, line width=1.0pt] (axis cs:" + _num(gmin)
             + ",0) -- (axis cs:" + _num(gmin) + ",1);")
    b.append(r"\draw[cclassical, line width=0.4pt, densely dashed] (axis cs:"
             + _num(gmin_lo) + ",0) -- (axis cs:" + _num(gmin_lo) + ",1);")
    b.append(r"\draw[cclassical, line width=0.4pt, densely dashed] (axis cs:"
             + _num(gmin_hi) + ",0) -- (axis cs:" + _num(gmin_hi) + ",1);")
    b.append(r"\draw[black, densely dotted, line width=1.0pt] (axis cs:"
             + _num(gsub) + ",0) -- (axis cs:" + _num(gsub) + ",1);")
    b.append(r"\draw[cproposed, dashed, line width=1.0pt] (axis cs:"
             + _num(ghead) + ",0) -- (axis cs:" + _num(ghead) + ",1);")
    # Region labels.
    b.append(r"\node[font=\figlab, align=center, text=black] at (axis cs:"
             + _num(g0 / 2) + r",0.60) {infeasible for\\any policy};")
    b.append(r"\node[font=\figlab, rotate=90, text=black] at (axis cs:"
             + _num(g0 + 0.28) + r",0.42) {unattainable};")
    b.append(r"\node[font=\figlab, text=black] at (axis cs:"
             + _num((gmin + hi) / 2) + r",0.62) {feasible};")
    # Annotations.
    b.append(r"\node[font=\figlab, anchor=east, inner sep=1pt] at (axis cs:"
             + _num(g0 - 0.10) + r",0.16) {$\Gamma_0{=}" + f"{g0:.2f}" + r"$};")
    b.append(r"\node[font=\figlab, anchor=west, inner sep=1pt] at (axis cs:"
             + _num(gmin + 0.06) + r",0.16) {$\Gamma_{\min}$};")
    b.append(r"\node[font=\figlab, anchor=north east, text=black] at (axis cs:"
             + _num(gsub - 0.04) + r",0.99) {$\Gamma{=}3$};")
    b.append(r"\node[font=\figlab, anchor=north west, text=cproposed] at (axis cs:"
             + _num(ghead + 0.06) + r",0.99) {adopted};")
    b.append(r"\end{axis}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r1_feasibility", "\n".join(b))


# ================================================================== r2
CLASSICAL = ["AlwaysOn", "Threshold", "LyapunovOnly", "DriftPlusPenalty",
             "DriftPlusPenalty:2.0", "SleepAwareDrift", "SleepAwareDrift:0.2"]
LEARNED = ["LagrangianPPO", "CRPO", "WCSAC"]
LEARNED_FILTERED = ["LagrangianPPO+filter", "CRPO+filter", "WCSAC+filter"]


def fig_pareto():
    d = _load("rev_headline.json")
    fr = _load("rev_frontier.json")
    gam = d["_gamma"]

    def xy(k):
        return (d[k]["avg_power_W"]["mean"], d[k]["cvar_beta"]["mean"])

    def pts(keys):
        return " ".join("(" + _num(x) + "," + _num(y) + ")"
                        for x, y in (xy(k) for k in keys))

    def curve(prefix, grid):
        p = sorted((fr[f"{prefix}:{g}"]["avg_power_W"]["mean"],
                    fr[f"{prefix}:{g}"]["cvar_beta"]["mean"]) for g in grid)
        return " ".join("(" + _num(x) + "," + _num(y) + ")" for x, y in p)

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{groupplot}[")
    b.append(r"  group style={group size=2 by 1, horizontal sep=30pt},")
    b.append(r"  width=88pt, height=42pt,")
    b.append(r"  xlabel={cluster power (W)},")
    b.append(r"  ylabel={$\mathrm{CVaR}_{0.95}(\ell)$},")
    # Legends sit under each panel, clear of the data. Because they no longer
    # overlap anything, the y ranges can be tightened back to the data.
    b.append(r"  legend columns=2,")
    b.append(r"  legend style={at={(0.5,-0.62)}, anchor=north, draw=none,"
             r" fill=none, font=\figtick,"
             r" /tikz/every even column/.append style={column sep=5pt}},")
    b.append(r"]")

    # --- left: every controller ---
    b.append(r"\nextgroupplot[title={all controllers}, xmin=350, xmax=1750,"
             r" ymin=2.6, ymax=10.4, xtick={600,1000,1400}, ytick={4,6,8,10}]")
    b.append(r"\addplot[only marks, mark=*, cclassical, mark size=1.3pt]"
             r" coordinates {" + pts(CLASSICAL) + r"};")
    b.append(r"\addlegendentry{classical}")
    b.append(r"\addplot[only marks, mark=triangle*, clearned, mark size=1.7pt]"
             r" coordinates {" + pts(LEARNED) + r"};")
    b.append(r"\addlegendentry{prior safe RL}")
    b.append(r"\addplot[only marks, mark=triangle, clearned, mark size=1.7pt,"
             r" line width=0.6pt] coordinates {" + pts(LEARNED_FILTERED) + r"};")
    b.append(r"\addlegendentry{${+}$filter}")
    b.append(r"\addplot[only marks, mark=star, cproposed, mark size=2.6pt,"
             r" line width=0.7pt] coordinates {" + pts(["SafeRL"]) + r"};")
    b.append(r"\addlegendentry{proposed}")
    b.append(r"\draw[cref, dashed, line width=0.6pt] (axis cs:350," + _num(gam)
             + r") -- (axis cs:1750," + _num(gam) + r");")
    b.append(r"\node[font=\fignote, text=cref, anchor=south east] at"
             r" (axis cs:1740," + _num(gam) + r") {$\Gamma$};")

    # --- right: against the swept classical frontier ---
    b.append(r"\nextgroupplot[title={vs.\ swept classical frontier},"
             r" xmin=880, xmax=1560, ymin=2.95, ymax=5.85,"
             r" xtick={1000,1200,1400}, ytick={3,4,5}]")
    b.append(r"\addplot[cclassical, mark=*, mark size=0.9pt] coordinates {"
             + curve("DriftPlusPenalty", fr["_rho_grid"]) + r"};")
    b.append(r"\addlegendentry{DPP, $\rho$}")
    b.append(r"\addplot[clight, mark=square*, mark size=0.9pt] coordinates {"
             + curve("SleepAwareDrift", fr["_qs_grid"]) + r"};")
    b.append(r"\addlegendentry{sleep-aware, $q_s$}")
    b.append(r"\addplot[only marks, mark=triangle, clearned, mark size=1.7pt,"
             r" line width=0.6pt] coordinates {" + pts(LEARNED_FILTERED) + r"};")
    b.append(r"\addlegendentry{prior RL${+}$filt.}")
    b.append(r"\addplot[only marks, mark=star, cproposed, mark size=2.8pt,"
             r" line width=0.7pt] coordinates {" + pts(["SafeRL"]) + r"};")
    b.append(r"\addlegendentry{proposed}")
    b.append(r"\draw[cref, dashed, line width=0.6pt] (axis cs:880," + _num(gam)
             + r") -- (axis cs:1560," + _num(gam) + r");")
    b.append(r"\end{groupplot}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r2_pareto", "\n".join(b))


# ================================================================== r3
def fig_filter():
    d = _load("rev_filter.json")

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{groupplot}[")
    b.append(r"  group style={group size=2 by 1, horizontal sep=30pt},")
    b.append(r"  width=88pt, height=45pt,")
    b.append(r"  ylabel={$\mathrm{CVaR}_{0.95}(\ell)$},")
    b.append(r"]")

    kinds = [("none_bias0.0", "none"),
             ("aggregate_bias0.0", "aggregate"),
             ("lcb_bias0.0", "LCB")]
    rows = []
    for i, (k, _) in enumerate(kinds):
        c = d[k]["cvar_beta"]
        rows.append((i, c["mean"], c["hi"] - c["mean"], c["mean"] - c["lo"]))
    b.append(r"\nextgroupplot[title={filter realization}, ybar, bar width=9pt,")
    b.append(r"  xmin=-0.6, xmax=2.6, ymin=0, ymax=11,")
    b.append(r"  xtick={0,1,2}, xticklabels={none, aggregate, LCB},")
    b.append(r"  xticklabel style={font=\figtick}, ytick={0,2,4,6,8,10},")
    b.append(r"  enlarge x limits=0.14]")
    # One bar per realization, so each can carry its own color.
    cols = ["cref", "clearned", "cproposed"]
    for (i, m, ep, em), col in zip(rows, cols):
        b.append("\\addplot[fill=" + col + ", draw=" + col
                 + ", bar shift=0pt"
                 + ", error bars/.cd, y dir=both, y explicit,"
                 " error bar style={line width=0.4pt, black},"
                 " error mark options={line width=0.4pt, mark size=1pt, black}]\n"
                 "  table[x=x, y=y, y error plus=ep, y error minus=em] {\n"
                 + _table(["x", "y", "ep", "em"], [(i, m, ep, em)]) + "\n};")

    biases = [-0.25, 0.0, 0.25, 0.5, 1.0]
    keys = [f"lcb_bias{x}" for x in biases]
    cv = [d[k]["cvar_beta"]["mean"] for k in keys]
    lo = [d[k]["cvar_beta"]["lo"] for k in keys]
    hi = [d[k]["cvar_beta"]["hi"] for k in keys]
    agg = d["aggregate_bias0.0"]["cvar_beta"]["mean"]
    b.append(r"\nextgroupplot[title={predictor error}, xmin=-35, xmax=110,")
    b.append(r"  ymin=3.0, ymax=8.6, ytick={4,6,8},"
             r" xtick={-25,0,25,50,100},")
    b.append(r"  xlabel={service-predictor bias (\%)},")
    b.append(r"  legend style={at={(0.02,0.98)}, anchor=north west}]")
    b.append(_errplot("cproposed, mark=*", [x * 100 for x in biases],
                      cv, lo, hi))
    b.append(r"\addlegendentry{LCB filter}")
    b.append(r"\addplot[clearned, dashed, line width=0.6pt, forget plot]"
             r" coordinates {(-35," + _num(agg) + r") (110," + _num(agg) + r")};")
    b.append(r"\node[font=\fignote, text=clearned, anchor=south east] at"
             r" (axis cs:108," + _num(agg) + r") {aggregate};")
    b.append(r"\end{groupplot}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r3_filter", "\n".join(b))


# ================================================================== r4
def fig_sensitivity():
    d = _load("rev_sensitivity.json")
    panels = [("beta", [0.9, 0.95, 0.99], r"CVaR level $\beta$"),
              ("lcb_kappa", [0.0, 0.5, 1.0, 2.0], r"LCB conservatism $\kappa$"),
              ("q0_cell_Mb", [0.25, 1.0, 4.0], r"threshold $q_0$ (Mb)"),
              ("lam_max", [10.0, 50.0, 200.0], r"dual cap $\lambda_{\max}$")]

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{groupplot}[")
    b.append(r"  group style={group size=2 by 2, horizontal sep=30pt,"
             r" vertical sep=30pt},")
    b.append(r"  width=82pt, height=25pt,")
    b.append(r"]")
    for idx, (knob, vals, lab) in enumerate(panels):
        cv = [d[f"{knob}={v}"]["cvar_beta"]["mean"] for v in vals]
        lo = [d[f"{knob}={v}"]["cvar_beta"]["lo"] for v in vals]
        hi = [d[f"{knob}={v}"]["cvar_beta"]["hi"] for v in vals]
        xs = list(range(len(vals)))
        ticks = ",".join(str(i) for i in xs)
        labs = ",".join(str(v) for v in vals)
        ymin = min(lo) - 0.12 * (max(hi) - min(lo) + 1e-6)
        ymax = max(hi) + 0.12 * (max(hi) - min(lo) + 1e-6)
        b.append(r"\nextgroupplot[title={" + lab + r"},")
        if idx % 2 == 0:
            b.append(r"  ylabel={$\mathrm{CVaR}_{0.95}$},")
        b.append(r"  xmin=-0.35, xmax=" + _num(len(vals) - 0.65) + ",")
        b.append(r"  ymin=" + _num(ymin) + r", ymax=" + _num(ymax) + ",")
        b.append(r"  xtick={" + ticks + r"}, xticklabels={" + labs + r"},")
        b.append(r"  xticklabel style={font=\fignote},"
                 r" yticklabel style={font=\fignote},")
        b.append(r"  ylabel style={font=\fignote}]")
        b.append(_errplot("cproposed, mark=*", xs, cv, lo, hi))
    b.append(r"\end{groupplot}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r4_sensitivity", "\n".join(b))


# ================================================================== r5
def fig_theory():
    v = _load("rev_vsweep.json")
    s = _load("rev_stress.json")
    sc = _load("rev_scalability.json")

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{groupplot}[")
    b.append(r"  group style={group size=3 by 1, horizontal sep=34pt},")
    b.append(r"  width=48pt, height=40pt,")
    b.append(r"]")

    Vs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    qb = [v[f"V={V}"]["backlog"]["mean"] for V in Vs]
    neely = [qb[0] * (V / Vs[0]) for V in Vs]
    b.append(r"\nextgroupplot[title={Corollary 2}, xmode=log, ymode=log,")
    b.append(r"  xlabel={$V$}, ylabel={$\bar Q$ (Mb)},")
    b.append(r"  xtick={1e-4,1e-2,1e0}, ymin=1, ymax=1e5,")
    b.append(r"  ytick={1e0,1e2,1e4},")
    b.append(r"  legend style={at={(0.03,0.99)}, anchor=north west, draw=none, fill=none}]")
    b.append(r"\addplot[cproposed, mark=*] coordinates {"
             + " ".join("(" + _num(x) + "," + _num(y) + ")"
                        for x, y in zip(Vs, qb)) + r"};")
    b.append(r"\addlegendentry{measured}")
    b.append(r"\addplot[cref, dashed] coordinates {"
             + " ".join("(" + _num(x) + "," + _num(y) + ")"
                        for x, y in zip(Vs, neely)) + r"};")
    b.append(r"\addlegendentry{Neely $O(V)$}")

    rates = [0.3, 0.5, 0.7]
    on = [s[f"rate{r}_filterTrue"]["backlog"]["mean"] for r in rates]
    off = [s[f"rate{r}_filterFalse"]["backlog"]["mean"] for r in rates]
    b.append(r"\nextgroupplot[title={Theorem 1}, ybar, bar width=4pt,"
             r" ymode=log,")
    b.append(r"  xlabel={arrival rate}, ylabel={$\bar Q$ (Mb)},")
    b.append(r"  xmin=-0.5, xmax=2.5, ymin=1, ymax=2e4,")
    b.append(r"  xtick={0,1,2}, xticklabels={0.3,0.5,0.7},")
    b.append(r"  ytick={1e0,1e2,1e4},")
    # Without a frame the default ybar legend sample is a full-height bar and
    # reads as extra data; a small square keys the colour without doing that.
    b.append(r"  legend image code/.code={\filldraw[#1]"
             r" (0cm,-0.045cm) rectangle (0.16cm,0.045cm);},")
    b.append(r"  legend style={at={(0.03,0.99)}, anchor=north west, draw=none, fill=none}]")
    b.append(r"\addplot[fill=cproposed, draw=cproposed] coordinates {"
             + " ".join("(" + str(i) + "," + _num(y) + ")"
                        for i, y in enumerate(on)) + r"};")
    b.append(r"\addlegendentry{filter on}")
    b.append(r"\addplot[fill=clearned, draw=clearned] coordinates {"
             + " ".join("(" + str(i) + "," + _num(y) + ")"
                        for i, y in enumerate(off)) + r"};")
    b.append(r"\addlegendentry{off}")

    Ks = [7, 19, 37, 61]
    it = [sc[f"K={K}"]["infer_us_per_slot"]["mean"] for K in Ks]
    b.append(r"\nextgroupplot[title={near-RT margin}, ymode=log,")
    b.append(r"  xlabel={cluster size $K$}, ylabel={inference ($\mu$s)},")
    b.append(r"  xmin=0, xmax=68, ymin=100, ymax=40000,")
    b.append(r"  xtick={7,37,61}, ytick={1e3,1e4}]")
    b.append(r"\addplot[cclassical, mark=square*] coordinates {"
             + " ".join("(" + str(k) + "," + _num(y) + ")"
                        for k, y in zip(Ks, it)) + r"};")
    b.append(r"\addplot[clearned, dashed, forget plot] coordinates"
             r" {(0,10000) (68,10000)};")
    b.append(r"\node[font=\fignote, text=clearned, anchor=south east] at"
             r" (axis cs:66,10000) {10\,ms};")
    b.append(r"\end{groupplot}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r5_theory", "\n".join(b))


# ================================================================== r6
def fig_scalability():
    d = _load("rev_scalability.json")
    Ks = [7, 19, 37, 61]
    pc = [d[f"K={K}"]["power_per_cell"]["mean"] for K in Ks]
    lo = [d[f"K={K}"]["power_per_cell"]["lo"] for K in Ks]
    hi = [d[f"K={K}"]["power_per_cell"]["hi"] for K in Ks]
    it = [d[f"K={K}"]["infer_us_per_slot"]["mean"] for K in Ks]

    b = []
    b.append(r"\begin{tikzpicture}")
    b.append(r"\begin{groupplot}[")
    b.append(r"  group style={group size=2 by 1, horizontal sep=32pt},")
    b.append(r"  width=88pt, height=68pt, xlabel={cluster size $K$},")
    b.append(r"]")
    b.append(r"\nextgroupplot[title={per-cell energy},"
             r" ylabel={power per cell (W)}, xmin=0, xmax=68, xtick={7,37,61}]")
    b.append(_errplot("cproposed, mark=*", Ks, pc, lo, hi))
    b.append(r"\nextgroupplot[title={near-RT margin}, ymode=log,"
             r" ylabel={inference ($\mu$s/slot)}, xmin=0, xmax=68,"
             r" xtick={7,37,61}, ymin=100, ymax=40000, ytick={1e3,1e4}]")
    b.append(r"\addplot[cclassical, mark=square*] coordinates {"
             + " ".join("(" + str(k) + "," + _num(y) + ")"
                        for k, y in zip(Ks, it)) + r"};")
    b.append(r"\addplot[clearned, dashed, forget plot] coordinates"
             r" {(0,10000) (68,10000)};")
    b.append(r"\node[font=\fignote, text=clearned, anchor=south east] at"
             r" (axis cs:66,10000) {10\,ms budget};")
    b.append(r"\end{groupplot}")
    b.append(r"\end{tikzpicture}")
    _write_and_compile("r6_scalability", "\n".join(b))


def main():
    print("[tikz figures] revision set (drawn at 252 pt column width)")
    fig_feasibility()
    fig_pareto()
    fig_filter()
    fig_sensitivity()
    fig_theory()
    fig_scalability()


if __name__ == "__main__":
    main()
