"""
check_wcsac.py
--------------
Exact check of the WCSAC-GS safety critic and actor step on the two-step MDP
of the R2 audit (audit/r2-2026-09-27, finding F03).

MDP: s0 moves to A or B with probability 1/2 at zero cost. At A the policy
takes a = 1 with probability p and incurs normalized loss h = 1 - gamma, or
a = 0 at zero loss; B incurs zero loss; both then terminate.

Checks
  1. Iterating the Bellman targets of sim.safe_baselines.wcsac_targets (the
     source's Eqs. (14)-(15)) with exact expectations converges to the true
     mean and variance of the loss-return at every state-action pair.
  2. The sampled target is unbiased: its average over sampled next states
     equals the exact target.
  3. The actor step (likelihood-ratio estimate of the derivative of
     E_{a~pi} Gamma(s, a) at the visited states with the critics fixed, with
     a per-state baseline) has the same sign as the true derivative of the
     rollout-average statistic used in the audit, whereas the R2-draft
     local-variance estimator had the opposite sign.

Run: PYTHONPATH=. python -m sim.check_wcsac   (writes results/r2/check_wcsac.json)
"""
from __future__ import annotations
import json
import math
import os

import numpy as np
import torch

from .safe_baselines import wcsac_targets, wcsac_lambda_targets, gaussian_cvar_coeff

P, GAMMA = 0.75, 0.99
H = 1.0 - GAMMA
K = gaussian_cvar_coeff(0.05)


def exact_moments(p):
    """True mean/variance of the discounted normalized loss-return."""
    out = {("A", 1): (H, 0.0), ("A", 0): (0.0, 0.0), ("B", 0): (0.0, 0.0)}
    # s0 (single dummy action): G = gamma * G_next.
    m1 = 0.5 * GAMMA * p * H
    m2 = 0.5 * GAMMA ** 2 * p * H ** 2
    out[("s0", 0)] = (m1, m2 - m1 ** 2)
    return out


def iterate_targets(p, n_iter=50):
    """Tabular fixed-point iteration of wcsac_targets with exact E over s', a'."""
    Q = {k: 0.0 for k in exact_moments(p)}
    V = {k: 0.0 for k in Q}
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    for _ in range(n_iter):
        Qn, Vn = dict(Q), dict(V)
        # Terminal pairs: c = h a at A, 0 at B; not_done = 0.
        for key, c in ((("A", 1), H), (("A", 0), 0.0), (("B", 0), 0.0)):
            qt, vt = wcsac_targets(t(c), t(Q[key]), torch.zeros(1, 1),
                                   torch.zeros(1, 1), t(0.0), GAMMA)
            Qn[key], Vn[key] = float(qt), float(vt)
        # s0: next state A (a' ~ Bernoulli(p)) or B, each w.p. 1/2. The target
        # is linear in the per-next-state terms, so the exact expectation is
        # the average of the targets over next states with a' weights.
        tq, tv = 0.0, 0.0
        for s_next, acts in (("A", ((1, p), (0, 1 - p))), ("B", ((0, 1.0),))):
            qn = torch.tensor([[Q[(s_next, a)] for a, _ in acts]], dtype=torch.float64)
            vn = torch.tensor([[V[(s_next, a)] for a, _ in acts]], dtype=torch.float64)
            w = torch.tensor([[wa for _, wa in acts]], dtype=torch.float64)
            # weighted mean over a' == mean over K equally likely draws
            eq = (qn * w).sum(-1); eq2 = (qn.pow(2) * w).sum(-1); ev = (vn * w).sum(-1)
            qt = 0.0 + GAMMA * eq
            vt = (0.0 - Q[("s0", 0)] ** 2
                  + (2 * GAMMA * 0.0 * eq + GAMMA ** 2 * ev + GAMMA ** 2 * eq2))
            tq += 0.5 * float(qt); tv += 0.5 * float(vt)
        Qn[("s0", 0)], Vn[("s0", 0)] = tq, tv
        Q, V = Qn, Vn
    return Q, V


def sampled_target_bias(p, Q, V, n=200_000, seed=0):
    """Average of wcsac_targets over sampled s' (and K=4 sampled a') at s0."""
    rng = np.random.default_rng(seed)
    to_a = rng.random(n) < 0.5
    acts = (rng.random((n, 4)) < p).astype(int)
    qn = np.where(to_a[:, None], np.where(acts == 1, Q[("A", 1)], Q[("A", 0)]), 0.0)
    vn = np.zeros_like(qn)
    qt, vt = wcsac_targets(torch.zeros(n, dtype=torch.float64),
                           torch.full((n,), Q[("s0", 0)], dtype=torch.float64),
                           torch.from_numpy(qn), torch.from_numpy(vn),
                           torch.ones(n, dtype=torch.float64), GAMMA)
    return float(qt.mean()), float(vt.mean())


def lambda_target_bias(p, Q, V, lam=0.95, n=200_000, seed=1):
    """Average lambda-return targets at s0 over sampled two-step trajectories
    (s0 -> A or B -> terminal) with exact critics; must equal the exact s0
    moments for any lambda."""
    rng = np.random.default_rng(seed)
    to_a = rng.random(n) < 0.5
    a_taken = (rng.random(n) < p) & to_a
    acts = (rng.random((n, 4)) < p).astype(int)
    qn0 = np.where(to_a[:, None], np.where(acts == 1, Q[("A", 1)], Q[("A", 0)]), 0.0)
    G = np.zeros(n); G2 = np.zeros(n)
    for i in range(0, n, 20000):
        sl = slice(i, i + 20000); m = min(20000, n - i)
        # rows: [s0, second state] per trajectory, interleaved
        c = torch.zeros(2 * m, dtype=torch.float64)
        c[1::2] = torch.from_numpy(np.where(a_taken[sl], H, 0.0))
        q_sa = torch.zeros(2 * m, dtype=torch.float64)
        q_sa[0::2] = Q[("s0", 0)]
        q_sa[1::2] = torch.from_numpy(np.where(to_a[sl], np.where(a_taken[sl], Q[("A", 1)], Q[("A", 0)]), 0.0))
        qn = torch.zeros(2 * m, 4, dtype=torch.float64)
        qn[0::2] = torch.from_numpy(qn0[sl])
        vn = torch.zeros_like(qn)
        nd = torch.zeros(2 * m, dtype=torch.float64); nd[0::2] = 1.0
        q_taken = torch.cat([q_sa[1:], q_sa[-1:]])
        g, v = wcsac_lambda_targets(c, q_sa, qn, vn, q_taken, nd, GAMMA, lam)
        G[sl] = g[0::2].numpy(); G2[sl] = v[0::2].numpy()
    return float(G.mean()), float(G2.mean())


def audit_statistic(p):
    J0 = GAMMA * H * p / 2; M0 = GAMMA ** 2 * H * H * (p / 2 - p * p / 4)
    JA = H * p; MA = H * H * p * (1 - p)
    return .5 * (J0 + K * math.sqrt(M0)) + .25 * (JA + K * math.sqrt(MA))


def main():
    ex = exact_moments(P)
    Q, V = iterate_targets(P)
    err = max(max(abs(Q[k] - ex[k][0]), abs(V[k] - ex[k][1])) for k in ex)
    q_mc, v_mc = sampled_target_bias(P, Q, V)
    q_l0, v_l0 = lambda_target_bias(P, Q, V, lam=0.0)
    q_l, v_l = lambda_target_bias(P, Q, V, lam=0.95)
    gam = {k: Q[k] + K * math.sqrt(max(V[k], 0.0)) for k in Q}
    # Actor step: rollout visits s0, A or B; only A has a policy-dependent
    # action, visited with weight 1/4 in the rollout average of the audit.
    # d/dp E_{a~Bern(p)} Gamma(A, a) = Gamma(A, 1) - Gamma(A, 0).
    actor_step = 0.25 * (gam[("A", 1)] - gam[("A", 0)])
    # Likelihood-ratio form with the per-state baseline, in expectation:
    base = P * gam[("A", 1)] + (1 - P) * gam[("A", 0)]
    lr = 0.25 * (P * (gam[("A", 1)] - base) / P
                 + (1 - P) * (gam[("A", 0)] - base) * (-1.0 / (1 - P)))
    eps = 1e-6
    true_deriv = (audit_statistic(P + eps) - audit_statistic(P - eps)) / (2 * eps)
    old_estimator = .25 * H * (1 + K * (1 - 2 * P) / (2 * math.sqrt(P * (1 - P))))
    out = dict(
        p=P, gamma=GAMMA, k_alpha=K,
        max_abs_error_fixed_point_vs_exact=err,
        s0_exact_mean_var=ex[("s0", 0)], s0_iterated_mean_var=(Q[("s0", 0)], V[("s0", 0)]),
        s0_sampled_target_mean_var=(q_mc, v_mc),
        s0_lambda0_target_mean_var=(q_l0, v_l0),
        s0_lambda095_target_mean_var=(q_l, v_l),
        actor_step_derivative=actor_step, actor_step_lr_expectation=lr,
        audit_true_statistic_derivative=true_deriv,
        r2_draft_estimator_derivative=old_estimator,
        sign_agrees_with_true=bool(np.sign(lr) == np.sign(true_deriv)),
        r2_draft_sign_agrees=bool(np.sign(old_estimator) == np.sign(true_deriv)))
    print(json.dumps(out, indent=2))
    os.makedirs("sim/results/r2", exist_ok=True)
    with open("sim/results/r2/check_wcsac.json", "w") as f:
        json.dump(out, f, indent=2)
    assert err < 1e-12 and abs(q_mc - ex[("s0", 0)][0]) < 1e-5
    assert abs(v_mc - ex[("s0", 0)][1]) < 1e-7 and out["sign_agrees_with_true"]
    for qq, vv in ((q_l0, v_l0), (q_l, v_l)):
        assert abs(qq - ex[("s0", 0)][0]) < 5e-5 and abs(vv - ex[("s0", 0)][1]) < 5e-7


if __name__ == "__main__":
    main()
