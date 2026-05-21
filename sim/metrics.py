"""
metrics.py
----------
Per-episode summary metrics with bootstrap confidence intervals.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Sequence


def empirical_cvar(loss: np.ndarray, beta: float) -> float:
    if len(loss) == 0:
        return 0.0
    q = np.quantile(loss, beta)
    tail = loss[loss >= q]
    return float(tail.mean()) if len(tail) else float(q)


def bootstrap_ci(samples: Sequence[float], n_boot: int = 2000,
                 ci: float = 0.95,
                 rng: np.random.Generator = None) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(samples, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "std": 0.0}
    if n == 1:
        return {"mean": float(x[0]), "lo": float(x[0]),
                "hi": float(x[0]), "std": 0.0}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100,
                                   (1 + ci) / 2 * 100])
    return {"mean": float(x.mean()), "lo": float(lo),
            "hi": float(hi), "std": float(x.std(ddof=1))}


def summarize_episode(power_series: np.ndarray,
                      loss_series: np.ndarray,
                      backlog_series: np.ndarray,
                      arrivals_series: np.ndarray,
                      toggles: int,
                      beta: float,
                      Gamma: float,
                      dt_s: float) -> Dict[str, float]:
    T = len(power_series)
    avg_power_W = float(power_series.mean())
    energy_J = float(power_series.sum() * dt_s)
    avg_backlog_Mb = float(backlog_series.mean())
    avg_arrival_Mbps = float(arrivals_series.mean()) / dt_s
    # Little's law -- approximate delay (s).
    if avg_arrival_Mbps > 0:
        avg_delay_ms = (avg_backlog_Mb / avg_arrival_Mbps) * 1000.0
    else:
        avg_delay_ms = 0.0
    inst_delay_ms = (backlog_series / max(avg_arrival_Mbps, 1.0)) * 1000.0
    p95_delay_ms = float(np.percentile(inst_delay_ms, 95))
    p99_delay_ms = float(np.percentile(inst_delay_ms, 99))
    cvar = empirical_cvar(loss_series, beta)
    viol_rate = float((loss_series > Gamma).mean())
    return {
        "avg_power_W": avg_power_W,
        "energy_J_total": energy_J,
        "avg_backlog_Mb": avg_backlog_Mb,
        "avg_delay_ms": avg_delay_ms,
        "p95_delay_ms": p95_delay_ms,
        "p99_delay_ms": p99_delay_ms,
        "cvar_beta": cvar,
        "viol_rate": viol_rate,
        "toggles_per_min": toggles / (T * dt_s / 60.0) if T > 0 else 0.0,
    }


def aggregate(metric_list: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not metric_list:
        return {}
    keys = list(metric_list[0].keys())
    rng = np.random.default_rng(0)
    out = {}
    for k in keys:
        vals = []
        for m in metric_list:
            if k in m:
                try:
                    vals.append(float(m[k]))
                except (TypeError, ValueError):
                    pass
        if vals:
            out[k] = bootstrap_ci(vals, rng=rng)
            out[k]["_per_seed"] = vals
    out["_n_seeds"] = len(metric_list)
    return out
