"""
r2_rerun.py
-----------
Targeted rerun after the second R2 audit (audit/r2-2026-09-27):

  F03  WCSAC-GS reimplemented (state-action Gaussian safety critic with the
       source's Bellman targets; the source's actor step against the fixed
       critic). Its budget d_ret is recalibrated on the validation seeds, and
       WCSAC and WCSAC+F are retrained and re-evaluated on the test seeds.
  F04  TextbookDPP reimplemented as a joint minimizer over admissible sleep
       patterns. Its V grid is re-evaluated on validation seeds (selection)
       and on test seeds.

Everything else in sim/results/r2 is unchanged; the updated entries are
written back into calibration.json, classical.json and
learned_headline_filter_sens_scale.json, and the replaced entries are kept
under "_r2draft" keys for the record. Old WCSAC checkpoints are moved to
ckpt/_r2draft/.

Run from the paper directory:
  PYTHONPATH=. OMP_NUM_THREADS=1 python -m sim.r2_rerun all --workers 20
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import time

import numpy as np

from . import r2
from .r2 import (VAL_SEEDS, TEST_SEEDS, UPDATES, CKPT, OUT, GAMMA,
                 _pool, _classical_eval, _wcsac_ao_both, _agg, _save,
                 _save_series, _train_one, _eval_one, _calib)

from .r2 import V_GRID   # refined grid, shared with the figures
WC = ["WCSAC", "WCSAC+F"]


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def stage_calibrate(pool):
    cal = _load("calibration.json")
    b = cal["budgets"]
    wcd = pool.map(_wcsac_ao_both, VAL_SEEDS)
    wc = [d["mc"]["stat"] for d in wcd]
    hist = cal.setdefault("_r2draft", {}).setdefault("budgets_wcsac_history", [])
    hist.append({k: b.get(k) for k in ("wcsac_stat_ao_val", "d_ret",
                                       "wcsac_stat_per_seed")})
    b.update(wcsac_stat_ao_val=float(np.mean(wc)),
             d_ret=b["tightness"] * float(np.mean(wc)),
             wcsac_stat_per_seed=wc, wcsac_calibration="monte_carlo",
             wcsac_calibration_check=wcd)
    for d in wcd:
        print(f"  MC: Q {d['mc']['q_mean']:.4f} vs G {d['mc']['mc_mean']:.4f}; "
              f"V {d['mc']['v_mean']:.5f} vs resid^2 {d['mc']['sq_resid_mean']:.5f}"
              f" | TD: Q {d['td']['mc']['q_mean']:.4f}, V {d['td']['mc']['v_mean']:.5f},"
              f" stat {d['td']['stat']:.4f} vs MC stat {d['mc']['stat']:.4f}",
              flush=True)
    print(f"  WCSAC always-on statistic (val) {np.mean(wc):.4f} "
          f"-> d_ret {b['d_ret']:.4f}", flush=True)
    _save(cal, "calibration.json")


def stage_dpp_val(pool):
    cal = _load("calibration.json")
    cal.setdefault("_r2draft", {})
    names = [f"TextbookDPP:{v}" for v in V_GRID]
    rows = pool.map(_classical_eval, [(n, s, {}, False) for n in names
                                      for s in VAL_SEEDS])
    old = {k: v for k, v in cal["val_grid"].items()
           if k.startswith("TextbookDPP")}
    cal["_r2draft"].setdefault("val_grid_textbookdpp", old)
    for k in old:
        del cal["val_grid"][k]
    for n in names:
        cal["val_grid"][n] = _agg([r for r in rows if r["controller"] == n])
        g = cal["val_grid"][n]
        print(f"  val {n:20s} P={g['avg_power_W']['mean']:7.1f} "
              f"CVaR={g['cvar_beta']['mean']:.3f}", flush=True)
    ok = [n for n in names if cal["val_grid"][n]["cvar_beta"]["mean"] <= GAMMA]
    cal["selection"]["V"] = (min(ok, key=lambda n: cal["val_grid"][n]
                                 ["avg_power_W"]["mean"]) if ok else None)
    print(f"  selected V: {cal['selection']['V']}", flush=True)
    _save(cal, "calibration.json")


def stage_classical(pool):
    cal = _load("calibration.json")
    grid = cal["val_grid"]
    names = [f"TextbookDPP:{v}" for v in V_GRID]
    # Validation-selected member (cheapest meeting the budget, else closest).
    sel = cal["selection"]["V"] or min(
        names, key=lambda n: grid[n]["cvar_beta"]["mean"])
    rows = pool.map(_classical_eval, [(n, s, {}, n == sel) for n in names
                                      for s in TEST_SEEDS])
    c = _load("classical.json")
    c.setdefault("_r2draft", {}).setdefault("textbookdpp", {
        k: v for k, v in c.items() if k.startswith("TextbookDPP")})
    for k in list(c):
        if k.startswith("TextbookDPP"):
            del c[k]
    for n in names:
        rs = [r for r in rows if r["controller"] == n]
        if n == sel:
            for r in rs:
                _save_series(n.replace(":", "_"), r["seed"], r)
        c[n] = _agg(rs)
        print(f"  test {n:20s} P={c[n]['avg_power_W']['mean']:7.1f} "
              f"CVaR={c[n]['cvar_beta']['mean']:.3f} "
              f"tog={c[n]['toggles_per_min']['mean']:.0f}", flush=True)
    c["_selection"] = dict(cal["selection"])
    c["_headline"] = ["AlwaysOn", "Threshold", sel, cal["selection"]["rho"],
                      "DriftPlusPenalty:1.0", "DriftPlusPenalty:2.0",
                      cal["selection"]["qs"] or min(
                          (n for n in grid if n.startswith("SleepAwareDrift")),
                          key=lambda n: grid[n]["cvar_beta"]["mean"])]
    _save(c, "classical.json")


def stage_train(pool):
    budgets = _calib()["budgets"]
    # Checkpoints are reused by filename, so any earlier WCSAC checkpoints
    # (trained under another implementation or budget) are moved aside.
    stamp = time.strftime("%Y%m%d%H%M%S")
    for n in WC:
        src = os.path.join(CKPT, n)
        if os.path.isdir(src):
            dst = os.path.join(CKPT, "_superseded", f"{n}_{stamp}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    jobs = [(n, s, budgets) for n in WC for s in TEST_SEEDS]
    t0 = time.time()
    for i, (n, s, dt) in enumerate(pool.imap_unordered(_train_one, jobs), 1):
        print(f"  [{i}/{len(jobs)}] {n} seed {s}: "
              f"{dt if isinstance(dt, str) else f'{dt:.0f}s'} "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)


def stage_evaluate(pool):
    budgets = _calib()["budgets"]
    rows = pool.map(_eval_one, [(n, s, UPDATES, "default", budgets)
                                for n in WC for s in TEST_SEEDS])
    fn = "learned_headline_filter_sens_scale.json"
    d = _load(fn)
    d.setdefault("_r2draft", {})
    for n in WC:
        key = f"{n}|u{UPDATES}|default"
        if key in d:
            d["_r2draft"].setdefault(key + "_history", []).append(d[key])
        rs = [r for r in rows if r["name"] == n]
        a = _agg(rs, keys=("avg_power_W", "cvar_beta", "cvar_95", "viol_rate",
                           "p99_delay_ms", "toggles_per_min", "awake_mean",
                           "share_exec_mean", "share_prop_mean", "stranded_pct",
                           "offload_pct", "mean_loss", "avg_backlog_Mb"))
        a["lam_final"] = r2.bootstrap_ci([r2._lam_of(n, x["seed"], UPDATES)
                                         for x in rs])
        d[key] = a
        print(f"  {n:8s} P={a['avg_power_W']['mean']:7.1f} "
              f"CVaR={a['cvar_beta']['mean']:.3f} "
              f"awake={a['awake_mean']['mean']:.2f}", flush=True)
    _save(d, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "dpp", "classical",
                                      "train", "evaluate", "all"])
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    with _pool(args.workers) as pool:
        if args.stage in ("calibrate", "all"):
            stage_calibrate(pool)
        if args.stage in ("dpp", "all"):
            stage_dpp_val(pool)
        if args.stage in ("dpp", "classical", "all"):
            stage_classical(pool)
        if args.stage in ("train", "all"):
            stage_train(pool)
        if args.stage in ("evaluate", "all"):
            stage_evaluate(pool)
    print(f"[r2_rerun] {args.stage} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
