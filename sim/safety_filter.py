"""
safety_filter.py
----------------
Lyapunov-guided safety filter: projects the actor's proposed action onto the
backlog-aware safe set of Eq. (9).

Two implementations are provided.

`lcb_project` is the filter the theory actually specifies. For every cell
whose backlog exceeds the activation threshold q0, it admits only actions
whose *conservative* predicted service covers the estimated arrival rate plus
a margin,

    mu_LCB(x, a) = mu_hat(x, a) - kappa * sigma_hat(x, a) >= a_hat + delta,

and projects by raising phi_b to the smallest value satisfying that
inequality (waking the cell if necessary). This is the minimal modification
of the proposed action, i.e. the projection of Eq. (10), and it makes the
executed action satisfy the hypothesis of Theorem 1 by construction.

`safe_project` is the coarse rule used in the submitted version: once the
*aggregate* backlog crosses a threshold, force the most-backlogged half of
the cells to full service. It is retained so the revision can quantify what
the coarse filter costs relative to the theory-specified one, and because
Reviewer 3 asked for exactly that comparison.

The service predictor is deliberately imperfect and its error is
parameterized (`pred_bias`, `pred_scale`) so that robustness to prediction
error and channel-model mismatch can be swept.
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg


def predict_service(cfg: SimCfg, phi: np.ndarray, chan_mean: float,
                    chan_std: float) -> tuple:
    """Predicted mean and standard deviation of per-cell service (Mb/slot).

    Returns (mu_hat, sigma_hat) for an awake cell operating at share phi.
    `pred_bias` and `pred_scale` inject controlled predictor error: a bias of
    +0.1 means the predictor is 10% optimistic about the channel, which is the
    failure mode the LCB is meant to absorb.
    """
    mu_cap = cfg.chan.mu_max_mbps * cfg.dt_s
    mu_floor = cfg.chan.mu_min_mbps * cfg.dt_s
    nominal = mu_floor + (mu_cap - mu_floor) * np.clip(phi, 0.0, 1.0)
    bias = getattr(cfg.algo, "pred_bias", 0.0)
    scale = getattr(cfg.algo, "pred_scale", 1.0)
    mu_hat = nominal * chan_mean * (1.0 + bias)
    sigma_hat = nominal * chan_std * scale
    return mu_hat, sigma_hat


def lcb_project(action: np.ndarray, q: np.ndarray, a_hat: np.ndarray,
                cfg: SimCfg, chan_mean: float, chan_std: float) -> np.ndarray:
    """Project onto the safe set of Eq. (9) using the LCB service predictor.

    For a cell with q_b >= q0 the admissible shares are those with
        (mu_floor + (mu_cap - mu_floor) phi) * (chan_mean(1+bias)
            - kappa * chan_std * scale)  >=  a_hat_b + delta,
    which inverts to a lower bound phi_b^min. The projection raises phi_b to
    that bound and leaves everything else untouched, so it is the minimal
    modification of the actor's proposal.
    """
    algo = cfg.algo
    mu_cap = cfg.chan.mu_max_mbps * cfg.dt_s
    mu_floor = cfg.chan.mu_min_mbps * cfg.dt_s
    kappa = algo.lcb_kappa
    delta = getattr(algo, "delta_margin_Mb", 0.0)
    q0 = getattr(algo, "q0_cell_Mb", 1.0)

    bias = getattr(algo, "pred_bias", 0.0)
    scale = getattr(algo, "pred_scale", 1.0)
    eff = chan_mean * (1.0 + bias) - kappa * chan_std * scale
    out = np.array(action, dtype=np.float32, copy=True)
    if eff <= 1e-9:
        # The conservative predictor cannot certify any service level; fall
        # back to full service on the backlogged cells.
        out[q >= q0] = 1.0
        return out

    need = (a_hat + delta) / eff                      # required Mb/slot
    phi_min = (need - mu_floor) / max(mu_cap - mu_floor, 1e-9)
    phi_min = np.clip(phi_min, 0.0, 1.0)
    active = q >= q0
    out = np.where(active, np.maximum(out, phi_min), out).astype(np.float32)
    return out


def safe_project(action: np.ndarray, q: np.ndarray, cfg: SimCfg) -> np.ndarray:
    """Coarse aggregate-backlog filter used in the submitted version.

    If total backlog exceeds the cluster threshold, force `action[b] = 1.0`
    for the most-backlogged half of the cells; otherwise pass through.
    """
    q_total = float(q.sum())
    if q_total <= cfg.algo.q_safety_threshold_Mb:
        return action  # safe -- pass through

    B = cfg.topo.B
    n_force = max(1, B // 2)
    idx = np.argpartition(-q, n_force - 1)[:n_force]
    out = action.copy()
    out[idx] = 1.0
    return out
