"""Classical references (always-on, selected backlog-proportional rule) on the
hexagonal clusters used by the scalability study. Test seeds only.
Run:  python3 -m sim.r2_scale_classical"""
import json, os
import multiprocessing as mp
from .r2 import _classical_eval, _agg, _save, TEST_SEEDS, _calib, _worker_init


def main():
    sel = _calib()["selection"]["rho"]
    jobs = [(n, s, {"topo.B": K}, False) for K in (19, 37, 61)
            for n in ("AlwaysOn", sel) for s in TEST_SEEDS]
    with mp.get_context("fork").Pool(20, initializer=_worker_init) as p:
        rows = p.map(_classical_eval, jobs)
    out = {}
    for (n, s, over, _), r in zip(jobs, rows):
        r["K"] = over["topo.B"]
    for K in (19, 37, 61):
        for n in ("AlwaysOn", sel):
            a = _agg([r for r in rows if r["K"] == K and r["controller"] == n])
            out[f"{n}|K={K}"] = a
            print(K, n, round(a["avg_power_W"]["mean"], 1),
                  round(a["cvar_95"]["mean"], 3))
    _save(out, "scale_classical.json")


if __name__ == "__main__":
    main()
