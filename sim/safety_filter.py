"""
safety_filter.py
----------------
Lyapunov-guided safety filter that projects the actor's proposed action
onto a backlog-aware safe set.

Idea: if total backlog exceeds q_safety_threshold_Mb, force the
most-backlogged half (floor(B/2)) of cells awake at maximum service
rate (phi_b = 1.0). This guarantees aggregate negative drift outside
a bounded region under the conditions of the paper's Corollary 1
(vector-queue extension of Theorem 1).

If the controller had access to a conservative service predictor
(LCB), we could be smarter (project onto the smallest action set with
positive drift). The current implementation is the simplest version
that delivers the stability guarantee.
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg


def safe_project(action: np.ndarray, q: np.ndarray, cfg: SimCfg) -> np.ndarray:
    """Project the actor's proposed `action` (B-vector in [0, 1]) onto the
    safe set given current backlog `q` (Mbit).

    Rule: if total backlog exceeds the threshold, force `action[b] = 1.0`
    for the most-backlogged half of cells; otherwise pass through the
    actor's choice.
    """
    q_total = float(q.sum())
    if q_total <= cfg.algo.q_safety_threshold_Mb:
        return action  # safe -- pass through

    # Force the most-backlogged cells to full service.
    B = cfg.topo.B
    n_force = max(1, B // 2)
    idx = np.argpartition(-q, n_force - 1)[:n_force]
    out = action.copy()
    out[idx] = 1.0
    return out
