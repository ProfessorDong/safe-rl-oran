"""
safety_filter.py
----------------
Lyapunov-guided safety filter: projects the actor's proposed action onto the
backlog-aware safe set of Eq. (safe_set), with a certified fallback.

`lcb_project` is the filter the analysis specifies. For every cell whose
backlog is at or above q0 it

  1. keeps the cell awake, waking it if it sleeps (a safety wake overrides the
     minimum off-dwell of the hysteresis);
  2. predicts the cell's effective arrivals conditional on the sleep pattern
     that will actually execute, i.e. including traffic offloaded onto it by
     sleeping neighbors, from a causal estimate of every cell's nominal load;
  3. charges the wake-up service loss when the cell wakes this slot;
  4. raises its share to the smallest phi with
         mu_LCB(phi) = mu_hat(phi) - kappa * sigma_hat(phi) >= a_hat + delta,
     where mu_hat and sigma_hat are the mean and standard deviation of the
     per-slot service; if no share satisfies it (the LCB-safe set is empty),
     the cell falls back to full share, the action a-dagger of the feasible-
     service assumption.

Every activation is logged as `feasible` (a certified projection) or
`fallback`, and the mean-margin check E[mu | x, a] >= a_hat + delta, with the
conditional mean service in place of the LCB, is recorded for the executed
action. With sigma_hat the per-slot spread of service, kappa * sigma_hat is a
per-slot service margin, stricter than the conditional-mean drift condition
Theorem 1 needs; the fallback executes the maximal-service action.

Up to R2 the filter clipped an infeasible share to one without recording it,
ignored the sleep state (a filter-woken cell could stay asleep under the dwell
lock), ignored the wake-up loss, and predicted arrivals from the previously
executed pattern.

`safe_project` is the coarse aggregate rule of the original submission: once
the cluster backlog crosses a threshold, force the most-backlogged half of the
cells to full service. It is retained only as the comparison.
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg

EPS_SLEEP = 0.05


def predict_service(cfg: SimCfg, phi: np.ndarray, chan_mean: float,
                    chan_std: float) -> tuple:
    """Predicted per-slot service mean and standard deviation (Mb/slot) of an
    awake cell at share phi, with the predictor-error knobs applied."""
    mu_cap = cfg.chan.mu_max_mbps * cfg.dt_s
    mu_floor = cfg.chan.mu_min_mbps * cfg.dt_s
    nominal = mu_floor + (mu_cap - mu_floor) * np.clip(phi, 0.0, 1.0)
    bias = getattr(cfg.algo, "pred_bias", 0.0)
    scale = getattr(cfg.algo, "pred_scale", 1.0)
    return nominal * chan_mean * (1.0 + bias), nominal * chan_std * scale


def new_filter_stats() -> dict:
    return dict(cell_slots=0, active=0, feasible=0, fallback=0, forced_wake=0,
                raised=0, mean_margin_ok=0, neighbor_wake=0)


def lcb_project(action: np.ndarray, env, a_nom_hat: np.ndarray, cfg: SimCfg,
                chan_mean: float, chan_std: float, stats: dict = None):
    """Project onto the LCB safe set. Returns (executed action, force mask).

    `env` supplies the current backlog, sleep state and dwell, the hysteresis
    predictor and the offload map; nothing in it is mutated.
    """
    algo = cfg.algo
    mu_cap = cfg.chan.mu_max_mbps * cfg.dt_s
    mu_floor = cfg.chan.mu_min_mbps * cfg.dt_s
    kappa = algo.lcb_kappa
    delta = getattr(algo, "delta_margin_Mb", 0.0)
    q0 = getattr(algo, "q0_cell_Mb", 1.0)
    bias = getattr(algo, "pred_bias", 0.0)
    scale = getattr(algo, "pred_scale", 1.0)
    wake_frac = getattr(cfg.energy, "wake_service_frac", 1.0)

    q = np.asarray(env.q)
    out = np.array(action, dtype=np.float32, copy=True)
    active = q >= q0
    if stats is not None:
        stats["cell_slots"] += len(out)
    if not active.any():
        return out, None

    # Pattern that will execute: hysteresis for inactive cells, active cells
    # held or woken by the filter. If even full share cannot cover an active
    # cell's expected arrivals (its own plus traffic offloaded onto it by
    # sleeping neighbors), the fallback is the JOINT action of the feasible-
    # service assumption: the cell at full share with the sleeping neighbors
    # that offload onto it woken, so that nothing is offloaded onto it.
    # Up to R2 the fallback raised only the cell's own share, which an
    # all-sleep proposal defeats by parking every neighbor asleep.
    force = active.copy()
    nbr = np.asarray(env.cov) > 0
    full_nom = mu_cap
    for _ in range(3):
        s_exec = env.predict_sleep(out, force_awake=force, commit=False)
        a_exp, _, _ = env.redistribute(a_nom_hat, s_exec)
        woke = (s_exec == 1) & (np.asarray(env.s_prev) == 0)
        wf = np.where(woke, wake_frac, 1.0)
        mean_full = wf * full_nom * chan_mean * (1.0 + bias)
        short = active & (mean_full < np.asarray(a_exp) + delta)
        wake_nb = np.zeros(len(out), dtype=bool)
        for b in np.flatnonzero(short):
            wake_nb |= nbr[:, b] & (s_exec == 0)   # sleeping cells offloading onto b
        wake_nb &= ~force
        if not wake_nb.any():
            break
        force |= wake_nb
    nb_woken = force & ~active

    eff = chan_mean * (1.0 + bias) - kappa * chan_std * scale
    need = np.asarray(a_exp, dtype=np.float64) + delta
    if eff > 1e-9:
        req_nominal = need / (eff * wf)
        phi_min = (req_nominal - mu_floor) / max(mu_cap - mu_floor, 1e-9)
        feasible = phi_min <= 1.0
    else:
        phi_min = np.full(len(out), np.inf)
        feasible = np.zeros(len(out), dtype=bool)
    phi_req = np.where(feasible, np.clip(phi_min, EPS_SLEEP, 1.0), 1.0)
    new = np.where(active, np.maximum(out, phi_req), out).astype(np.float32)
    # Neighbors woken only to stop offloading run at least the minimum share.
    new = np.where(nb_woken, np.maximum(new, EPS_SLEEP), new).astype(np.float32)

    if stats is not None:
        stats["active"] += int(active.sum())
        stats["feasible"] += int((active & feasible).sum())
        stats["fallback"] += int((active & ~feasible).sum())
        stats["forced_wake"] += int((active & woke).sum())
        stats["raised"] += int((active & (new > out + 1e-6)).sum())
        stats["neighbor_wake"] += int(nb_woken.sum())
        mean_serv = (wf * (mu_floor + (mu_cap - mu_floor) * new) * chan_mean)
        stats["mean_margin_ok"] += int((active & (mean_serv >= need)).sum())
    return new, force


def safe_project(action: np.ndarray, env, cfg: SimCfg, stats: dict = None):
    """Coarse aggregate-backlog filter of the original submission.

    If total backlog exceeds the cluster threshold, force the most-backlogged
    half of the cells to full service (and awake). Returns (action, force)."""
    q = np.asarray(env.q)
    out = np.array(action, dtype=np.float32, copy=True)
    if stats is not None:
        stats["cell_slots"] += len(out)
    if float(q.sum()) <= cfg.algo.q_safety_threshold_Mb:
        return out, None
    B = cfg.topo.B
    n_force = max(1, B // 2)
    idx = np.argpartition(-q, n_force - 1)[:n_force]
    force = np.zeros(B, dtype=bool)
    force[idx] = True
    out[idx] = 1.0
    if stats is not None:
        stats["active"] += int(n_force)
        stats["fallback"] += int(n_force)
    return out, force
