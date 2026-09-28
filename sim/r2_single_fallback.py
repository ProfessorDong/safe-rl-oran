"""Ablation of the filter's joint fallback under the adversarial all-sleep
proposal (Sec. VI-F). Same protocol as r2.stage_stress, with the fallback
restricted to the backlogged cell's own share (the pre-R2 behavior).
Run:  python3 -m sim.r2_single_fallback"""
import multiprocessing as mp
from .r2 import _stress_one, _agg, _save, TEST_SEEDS, _worker_init, _cfg
from . import r2


def _one(args):
    rate, seed = args
    orig = r2._cfg

    def cfg_single(over=None, Gamma=3.5):
        over = dict(over or {})
        over["algo.filter_joint_fallback"] = False
        return orig(over, Gamma)
    r2._cfg = cfg_single
    try:
        return _stress_one((rate, "allsleep", True, seed))
    finally:
        r2._cfg = orig


def main():
    rates = [0.3, 0.5, 0.7]
    jobs = [(r, s) for r in rates for s in TEST_SEEDS]
    with mp.get_context("fork").Pool(20, initializer=_worker_init) as p:
        rows = p.map(_one, jobs)
    out = {}
    for r in rates:
        a = _agg([x for x in rows if x["rate"] == r],
                 keys=("avg_backlog_Mb", "avg_power_W", "p99_delay_ms"))
        out[f"rate{r}|allsleep|single_fallback"] = a
        print(r, round(a["avg_backlog_Mb"]["mean"], 2),
              [round(v, 1) for v in a["avg_backlog_Mb"]["per_seed"]])
    _save(out, "stress_single_fallback.json")


if __name__ == "__main__":
    main()
