"""
run.py
------
End-to-end driver for Paper 4 simulator. CLI:

    python3 -m sim.run train       # train SafeRL on canonical seeds
    python3 -m sim.run eval        # evaluate trained policies + baselines
    python3 -m sim.run all         # train + eval, save results
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List
import numpy as np

from .config import SimCfg, default_cfg, canonical_seeds
from .arrivals import generate_arrivals
from .env import CellularEnv
from .baselines import make_baseline
from .metrics import summarize_episode, aggregate


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def train_one_seed(cfg: SimCfg, seed: int, n_updates: int = 50,
                   verbose: bool = False) -> dict:
    """Train SafeRL on one seed, log per-update diagnostics."""
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_train,
                                          seed=seed)
    env = CellularEnv(cfg, arrivals, seed=seed)
    ctl = make_baseline("SafeRL", cfg, seed=seed)
    history = {"viol_rate": [], "avg_energy_W": [], "avg_loss": [],
               "lambda": [], "tau": [], "z": [],
               "actor_loss": [], "critic_loss": []}
    for u in range(n_updates):
        batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        info = ctl.update_actor_critic(batch)
        history["viol_rate"].append(float(batch["viol"].mean()))
        history["avg_energy_W"].append(float(batch["energy"].mean()))
        history["avg_loss"].append(float(batch["loss"].mean()))
        history["lambda"].append(info["lambda"])
        history["tau"].append(info["tau"])
        history["z"].append(info["z"])
        history["actor_loss"].append(info["actor_loss"])
        history["critic_loss"].append(info["critic_loss"])
        if verbose and (u % max(1, n_updates // 10) == 0):
            print(f"  [seed={seed}] update {u:3d}: "
                  f"viol={history['viol_rate'][-1]:.3f} "
                  f"E={history['avg_energy_W'][-1]:.1f} "
                  f"L={history['avg_loss'][-1]:.3f} "
                  f"lam={info['lambda']:.2f} tau={info['tau']:.2f}")
    return {"history": history, "source": source,
            "controller_state": {"lambda": ctl.lam, "tau": ctl.tau,
                                  "z": ctl.z},
            "actor_state_dict": {k: v.detach().cpu().numpy().tolist()
                                  for k, v in ctl.actor.state_dict().items()},
            "critic_state_dict": {k: v.detach().cpu().numpy().tolist()
                                   for k, v in ctl.critic.state_dict().items()},
            }


def eval_one_seed(cfg: SimCfg, seed: int, ctrl_name: str,
                  trained_actor: dict = None,
                  trained_critic: dict = None) -> dict:
    """Run a single deterministic evaluation episode for a named controller."""
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_eval,
                                          seed=seed + 9000)
    env = CellularEnv(cfg, arrivals, seed=seed + 9000)
    ctl = make_baseline(ctrl_name, cfg, seed=seed)
    # If we have a trained policy, load it.
    if trained_actor is not None and hasattr(ctl, "actor"):
        import torch
        ctl.actor.load_state_dict({k: torch.tensor(np.array(v))
                                    for k, v in trained_actor.items()})
        if trained_critic is not None:
            ctl.critic.load_state_dict({k: torch.tensor(np.array(v))
                                         for k, v in trained_critic.items()})

    # Pure rollout (no learning).
    energy_s = []
    loss_s = []
    backlog_s = []
    arr_s = []
    toggles = 0
    s = env.state()
    done = False
    while not done:
        if hasattr(ctl, "act"):
            a, _, _, _ = ctl.act(s, env.q, stochastic=False)
        else:
            a = np.ones(cfg.action_dim, dtype=np.float32)
        s_next, cost, done = env.step(a)
        energy_s.append(cost["energy_W"])
        loss_s.append(cost["loss"])
        backlog_s.append(cost["q_total_Mb"])
        arr_s.append(cost["arrived_Mb"])
        toggles += cost["n_toggles"]
        s = s_next
    metrics = summarize_episode(
        np.array(energy_s), np.array(loss_s), np.array(backlog_s),
        np.array(arr_s), toggles=toggles,
        beta=cfg.algo.beta, Gamma=cfg.algo.Gamma, dt_s=cfg.dt_s,
    )
    metrics["source"] = source
    metrics["controller"] = ctrl_name
    return metrics


# ---------------------------------------------------------------------------
def train_one_seed_named(cfg: SimCfg, seed: int, ctrl_name: str,
                          n_updates: int) -> dict:
    """Train a named controller on one seed."""
    arrivals, source = generate_arrivals(cfg, T=cfg.time.T_slots_train,
                                          seed=seed)
    env = CellularEnv(cfg, arrivals, seed=seed)
    ctl = make_baseline(ctrl_name, cfg, seed=seed)
    history = {"viol_rate": [], "avg_energy_W": [], "avg_loss": [],
               "lambda": [], "tau": [], "z": []}
    for u in range(n_updates):
        batch = ctl.collect_rollout(env, n_slots=cfg.algo.rollout_slots)
        info = ctl.update_actor_critic(batch)
        history["viol_rate"].append(float(batch["viol"].mean()))
        history["avg_energy_W"].append(float(batch["energy"].mean()))
        history["avg_loss"].append(float(batch["loss"].mean()))
        history["lambda"].append(info.get("lambda", 0.0))
        history["tau"].append(info.get("tau", 0.0))
        history["z"].append(info.get("z", 0.0))
    out = {"history": history, "source": source,
           "controller_state": {"lambda": getattr(ctl, "lam", 0.0),
                                 "tau": getattr(ctl, "tau", 0.0),
                                 "z": getattr(ctl, "z", 0.0)}}
    if hasattr(ctl, "actor"):
        out["actor_state_dict"] = {k: v.detach().cpu().numpy().tolist()
                                    for k, v in ctl.actor.state_dict().items()}
        out["critic_state_dict"] = {k: v.detach().cpu().numpy().tolist()
                                     for k, v in ctl.critic.state_dict().items()}
    return out


def exp_train_all(cfg: SimCfg, n_seeds: int, n_updates: int) -> dict:
    """Train SafeRL AND UnconstrainedPPO on each seed (separately!).
    LyapunovOnly and Threshold need no training; we just stash a placeholder."""
    seeds = canonical_seeds(n_seeds)
    print(f"[train] {n_seeds} seeds x {n_updates} updates of "
          f"{cfg.algo.rollout_slots} slots (SafeRL + UnconstrainedPPO)")
    results = {"SafeRL": {}, "UnconstrainedPPO": {}}
    for ctrl in ("SafeRL", "UnconstrainedPPO"):
        for i, s in enumerate(seeds):
            t0 = time.time()
            print(f"\n  [{ctrl}] seed {i+1}/{n_seeds}: {s}")
            results[ctrl][f"seed_{s}"] = train_one_seed_named(
                cfg, s, ctrl, n_updates=n_updates)
            print(f"  -> {time.time() - t0:.1f}s")
    return results


def exp_eval_baselines(cfg: SimCfg, train_results: dict,
                       n_seeds: int) -> dict:
    """Evaluate each controller using its own trained policy (for PPO-based
    controllers) or its model-based rule (for the rest)."""
    seeds = canonical_seeds(n_seeds)
    out = {}
    for ctrl in ["SafeRL", "UnconstrainedPPO", "LyapunovOnly", "Threshold"]:
        rows = []
        for s in seeds:
            actor_sd = None
            critic_sd = None
            if ctrl in ("SafeRL", "UnconstrainedPPO"):
                # Each PPO-based controller uses ITS OWN trained policy.
                rec = train_results.get(ctrl, {}).get(f"seed_{s}", {})
                actor_sd = rec.get("actor_state_dict")
                critic_sd = rec.get("critic_state_dict")
            rows.append(eval_one_seed(cfg, s, ctrl,
                                       trained_actor=actor_sd,
                                       trained_critic=critic_sd))
        out[ctrl] = aggregate(rows)
        print(f"  {ctrl:>17}: P={out[ctrl]['avg_power_W']['mean']:.1f}W  "
              f"CVaR={out[ctrl]['cvar_beta']['mean']:.3f}  "
              f"viol={out[ctrl]['viol_rate']['mean']*100:.2f}%  "
              f"p99={out[ctrl]['p99_delay_ms']['mean']:.0f}ms")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", default="all", nargs="?",
                    choices=["train", "eval", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--updates", type=int, default=80)
    args = ap.parse_args()
    cfg = default_cfg()

    os.makedirs(cfg.results_dir, exist_ok=True)
    if args.which in ("train", "all"):
        t0 = time.time()
        train_out = exp_train_all(cfg, args.seeds, args.updates)
        save_json(train_out, os.path.join(cfg.results_dir,
                                           "train_safe_rl.json"))
        print(f"\n[train] done in {time.time() - t0:.1f}s")

    if args.which in ("eval", "all"):
        with open(os.path.join(cfg.results_dir, "train_safe_rl.json")) as f:
            train_out = json.load(f)
        t0 = time.time()
        eval_out = exp_eval_baselines(cfg, train_out, args.seeds)
        save_json(eval_out, os.path.join(cfg.results_dir,
                                          "eval_baselines.json"))
        print(f"\n[eval] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
