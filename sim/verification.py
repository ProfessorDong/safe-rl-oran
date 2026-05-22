"""
verification.py
---------------
Three rigorous theorem-verification experiments for Paper 4.

A. V-sweep      : sweep the energy weight V across 5 values (4 orders
                  of magnitude) to verify Corollary 2's asymptotic
                  [O(1/V), O(1)] energy/backlog tradeoff. Tests:
                    - power scales approximately as 1/V (Prop. 1)
                    - mean backlog stays *uniformly bounded* in V
                      (the safety-filter-induced stricter-than-Neely
                      claim, distinct from canonical Neely's O(V))

B. Stress test  : push the cluster into a high-load regime
                  (base_rate * util) and compare Full Safe-RL (filter
                  active) vs NoSafetyFilter, with risk pressure
                  removed (Gamma = 100, so the dual stays at 0). This
                  isolates the safety filter as the queue-stabilizing
                  mechanism (Theorem 1).

C. Theory-aligned : Robbins-Monro diminishing stepsize schedule plus
                    uncapped dual, longer training horizon. Tests
                    whether the idealized hypotheses of Theorem 2
                    (mean-rate-stable Z, projected tau converging)
                    are achievable in a practical PPO implementation
                    and whether CVaR approaches Gamma.

Output: sim/results/verification.json (aggregated per experiment).
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import time
import numpy as np

from .config import default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import SafeRLController
from .metrics import summarize_episode, aggregate


# ---------------------------------------------------------------------------
def _train_and_eval(cfg, seed, n_updates, use_safety_filter=True,
                    enforce_risk=True):
    """Helper: train a SafeRLController on seed and evaluate on seed+9000."""
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_train,
                                          seed=seed)
    env = CellularEnv(cfg, arrivals, seed=seed)
    ctl = SafeRLController(cfg, seed=seed,
                            use_safety_filter=use_safety_filter,
                            enforce_risk=enforce_risk)
    history = {"viol_rate": [], "avg_energy_W": [], "avg_loss": [],
               "lambda": [], "tau": [], "z": [],
               "avg_backlog_Mb": []}
    for u in range(n_updates):
        batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        info = ctl.update_actor_critic(batch)
        history["viol_rate"].append(float(batch["viol"].mean()))
        history["avg_energy_W"].append(float(batch["energy"].mean()))
        history["avg_loss"].append(float(batch["loss"].mean()))
        history["lambda"].append(info["lambda"])
        history["tau"].append(info["tau"])
        history["z"].append(info["z"])
    # Eval on held-out arrivals.
    arr_eval, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                     seed=seed + 9000)
    env_eval = CellularEnv(cfg, arr_eval, seed=seed + 9000)
    energy_s, loss_s, backlog_s, arr_s = [], [], [], []
    toggles = 0
    s = env_eval.state()
    done = False
    while not done:
        a, _, _, _ = ctl.act(s, env_eval.q, stochastic=False)
        s_next, cost, done = env_eval.step(a)
        energy_s.append(cost["energy_W"])
        loss_s.append(cost["loss"])
        backlog_s.append(cost["q_total_Mb"])
        arr_s.append(cost["arrived_Mb"])
        toggles += cost["n_toggles"]
        s = s_next
    metrics = summarize_episode(
        np.array(energy_s), np.array(loss_s),
        np.array(backlog_s), np.array(arr_s),
        toggles=toggles,
        beta=cfg.algo.beta, Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s,
    )
    metrics["history_tail"] = {
        "lambda": history["lambda"][-10:],
        "tau":    history["tau"][-10:],
        "z":      history["z"][-10:],
    }
    return metrics


# ---------------------------------------------------------------------------
def exp_A_v_sweep(seeds, n_updates):
    """Experiment A: V-sweep over the energy weight in c_lambda."""
    print("\n" + "="*72)
    print("[EXP A] V-sweep (Corollary 2 verification)")
    print("="*72)
    Vs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    out = {"Vs": Vs, "per_V": {}}
    for V in Vs:
        print(f"\n  [V = {V:g}]")
        rows = []
        for i, s in enumerate(seeds):
            t0 = time.time()
            cfg = default_cfg()
            cfg.algo.V_energy_weight = V
            m = _train_and_eval(cfg, s, n_updates)
            rows.append(m)
            print(f"    seed {i+1}/{len(seeds)} = {s}: "
                  f"P={m['avg_power_W']:7.1f}W  "
                  f"<Q>={m['avg_backlog_Mb']:5.1f}Mb  "
                  f"CVaR={m['cvar_beta']:5.2f}  "
                  f"viol={m['viol_rate']*100:4.1f}%  ({time.time()-t0:.0f}s)")
        out["per_V"][f"V_{V:g}"] = aggregate(rows)
        agg = out["per_V"][f"V_{V:g}"]
        print(f"  >> V={V:g}: P={agg['avg_power_W']['mean']:.1f}W "
              f"<Q>={agg['avg_backlog_Mb']['mean']:.1f}Mb "
              f"CVaR={agg['cvar_beta']['mean']:.2f}")
    return out


def exp_B_stress(seeds, n_updates):
    """Experiment B: arrival-rate stress test of the safety filter."""
    print("\n" + "="*72)
    print("[EXP B] Stress test (Theorem 1 verification)")
    print("="*72)
    # base_rate < per-cell service capacity (~0.8 Mb/slot at full phi) to
    # satisfy Slater (Assumption 3); 0.9 violates Slater and is excluded.
    rates = [0.30, 0.50, 0.70]
    out = {"rates": rates, "per_rate": {}}
    for r in rates:
        for variant, kwargs in [("WithFilter",    dict(use_safety_filter=True,
                                                        enforce_risk=False)),
                                 ("NoFilter",      dict(use_safety_filter=False,
                                                        enforce_risk=False))]:
            print(f"\n  [base_rate={r}  {variant}]")
            rows = []
            for i, s in enumerate(seeds):
                t0 = time.time()
                cfg = default_cfg()
                cfg.arr.base_rate_Mb_per_slot = r
                cfg.algo.Gamma = 100.0     # loose: removes risk pressure
                cfg.algo.fixed_lambda = 0.0  # no dual ascent during training
                m = _train_and_eval(cfg, s, n_updates,
                                     use_safety_filter=kwargs["use_safety_filter"],
                                     enforce_risk=False)
                rows.append(m)
                print(f"    seed {i+1}/{len(seeds)} = {s}: "
                      f"P={m['avg_power_W']:7.1f}W  "
                      f"<Q>={m['avg_backlog_Mb']:6.1f}Mb  "
                      f"p99-Q-delay={m['p99_delay_ms']:7.0f}ms  "
                      f"({time.time()-t0:.0f}s)")
            key = f"rate_{r:.2f}_{variant}"
            out["per_rate"][key] = aggregate(rows)
            agg = out["per_rate"][key]
            print(f"  >> {key}: P={agg['avg_power_W']['mean']:.1f}W "
                  f"<Q>={agg['avg_backlog_Mb']['mean']:.1f}Mb "
                  f"p99-delay={agg['p99_delay_ms']['mean']:.0f}ms")
    return out


def exp_C_theory_aligned(seeds, n_updates):
    """Experiment C: theory-aligned diminishing-stepsize + uncapped dual."""
    print("\n" + "="*72)
    print(f"[EXP C] Theory-aligned schedule (Theorem 2 verification, "
          f"{n_updates} updates)")
    print("="*72)
    rows = []
    history_per_seed = []
    for i, s in enumerate(seeds):
        t0 = time.time()
        cfg = default_cfg()
        cfg.algo.diminishing_schedule = True
        cfg.algo.lam_max = 1e6           # effectively uncapped
        cfg.algo.risk_warmup_slots = 100 # short warmup
        # Full SafeRL with theory-aligned schedules.
        arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_train,
                                              seed=s)
        env = CellularEnv(cfg, arrivals, seed=s)
        ctl = SafeRLController(cfg, seed=s,
                                use_safety_filter=True,
                                enforce_risk=True)
        # Track CVaR over training (per-update rollout CVaR on the loss).
        cvar_traj = []
        lam_traj  = []
        tau_traj  = []
        z_traj    = []
        for u in range(n_updates):
            batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
            ctl.update_actor_critic(batch)
            # Compute empirical CVaR on this rollout's losses.
            losses = batch["loss"]
            q = np.quantile(losses, cfg.algo.beta)
            tail = losses[losses >= q]
            cvar = float(tail.mean()) if len(tail) else float(q)
            cvar_traj.append(cvar)
            lam_traj.append(float(ctl.lam))
            tau_traj.append(float(ctl.tau))
            z_traj.append(float(ctl.z))
        history_per_seed.append({
            "cvar_traj": cvar_traj, "lam_traj": lam_traj,
            "tau_traj": tau_traj,   "z_traj":   z_traj,
        })
        # Final eval.
        arr_eval, _ = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                         seed=s + 9000)
        env_eval = CellularEnv(cfg, arr_eval, seed=s + 9000)
        energy_s, loss_s, backlog_s, arr_s = [], [], [], []
        toggles = 0
        st = env_eval.state()
        done = False
        while not done:
            a, _, _, _ = ctl.act(st, env_eval.q, stochastic=False)
            sn, cost, done = env_eval.step(a)
            energy_s.append(cost["energy_W"])
            loss_s.append(cost["loss"])
            backlog_s.append(cost["q_total_Mb"])
            arr_s.append(cost["arrived_Mb"])
            toggles += cost["n_toggles"]
            st = sn
        m = summarize_episode(np.array(energy_s), np.array(loss_s),
                              np.array(backlog_s), np.array(arr_s),
                              toggles=toggles, beta=cfg.algo.beta,
                              Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s)
        rows.append(m)
        print(f"  seed {i+1}/{len(seeds)} = {s}: "
              f"P={m['avg_power_W']:7.1f}W  "
              f"CVaR={m['cvar_beta']:5.2f}  "
              f"viol={m['viol_rate']*100:4.1f}%  "
              f"final_lam={lam_traj[-1]:.2f}  "
              f"final_z={z_traj[-1]:.0f}  "
              f"({time.time()-t0:.0f}s)")
    out = aggregate(rows)
    out["history_per_seed"] = history_per_seed
    print(f"  >> theory-aligned: P={out['avg_power_W']['mean']:.1f}W "
          f"CVaR={out['cvar_beta']['mean']:.2f} "
          f"viol={out['viol_rate']['mean']*100:.2f}%")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_A", type=int, default=5)
    ap.add_argument("--seeds_B", type=int, default=5)
    ap.add_argument("--seeds_C", type=int, default=10)
    ap.add_argument("--updates_A", type=int, default=200)
    ap.add_argument("--updates_B", type=int, default=200)
    ap.add_argument("--updates_C", type=int, default=600)
    ap.add_argument("--only", type=str, default="ABC",
                    help="subset of {A,B,C} to run")
    args = ap.parse_args()

    seeds_A = canonical_seeds(args.seeds_A)
    seeds_B = canonical_seeds(args.seeds_B)
    seeds_C = canonical_seeds(args.seeds_C)

    out = {}
    t0 = time.time()
    if "A" in args.only:
        out["exp_A_v_sweep"] = exp_A_v_sweep(seeds_A, args.updates_A)
    if "B" in args.only:
        out["exp_B_stress"]  = exp_B_stress(seeds_B,  args.updates_B)
    if "C" in args.only:
        out["exp_C_theory_aligned"] = exp_C_theory_aligned(
            seeds_C, args.updates_C)

    out_path = "sim/results/verification.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[verification] saved {out_path} "
          f"(total wall-clock {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
