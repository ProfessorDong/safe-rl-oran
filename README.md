# Safe-RL for O-RAN

Reference simulator and reproducible numerical results for the paper

> **Lyapunov-Guided Safe Reinforcement Learning for Energy-Aware O-RAN
> Scheduling under Tail-Risk Constraints**
> Liang Dong, Senior Member, IEEE.
> Submitted to *IEEE Transactions on Cognitive Communications and Networking*.

The codebase implements a two-timescale primal-dual actor-critic for a
seven-cell O-RAN cluster: a PyTorch Gaussian policy is regularized by a
risk virtual queue (CVaR-on-loss via the Rockafellar-Uryasev epigraph
form), a Lyapunov-guided safety filter projects exploratory actions onto
a backlog-aware safe set, and the dual variable ascends on the slow
timescale to enforce the long-run risk constraint.

## Headline results

Ten canonical seeds, 100 s evaluation episodes at 10 ms slot granularity,
seven cells, beta=0.95, Gamma=3.0, 95% bootstrap confidence intervals:

| Controller | Power (W) | CVaR | Violation rate | p99 delay | Toggles/min |
|---|---|---|---|---|---|
| **Safe-RL (proposed)** | 1318 [1295, 1341] | **5.94** | **17.5%** | 109 ms | **0.0** |
| Unconstrained PPO | 351 [274, 418] | 10.00 | 99.9% | 79 017 ms | 27.1 |
| Lyapunov drift | 909 [907, 910] | 5.70 | 38.4% | 61 ms | 5674 |
| Threshold heuristic | 454 [453, 456] | 10.00 | 99.9% | 199 ms | 2631 |

The Gamma sweep over {1.5, 2.0, 3.0, 4.5, 7.0} exhibits a U-shaped
violation curve with practical sweet spot at Gamma=4.5
(15.1% violation at 1253 W).

## Repository layout

```
sim/
├── config.py            Calibrated parameters (single source of truth)
├── arrivals.py          Per-cell hourly arrivals: Shanghai Telecom -> C2TM -> synthetic
├── channel_lumos5g.py   Per-slot service-rate jitter from Lumos5G mmWave throughput
├── env.py               Seven-cell gym-style environment with energy + loss accounting
├── networks.py          Two-layer Tanh Gaussian actor + scalar critic (PyTorch)
├── safety_filter.py     Lyapunov-guided action projector (q > q_safety -> force-serve)
├── algorithm.py         Two-timescale primal-dual PPO with risk virtual queue
├── baselines.py         LyapunovOnly + ThresholdHeuristic + factory + rollout helper
├── metrics.py           Per-episode CVaR, p95/p99 delay, bootstrap CI helpers
├── run.py               train | eval | all driver
├── gamma_sweep.py       Pareto sweep over Gamma (figure E3)
├── make_figures.py      Publication-quality PDF figures (E1, E2, E4)
└── make_tables.py       Headline comparison LaTeX table
fig/                     Generated figures (PDF)
tab/                     Generated LaTeX tables
sim/results/             Cached experiment JSONs (intermediate; regenerable)
sim/data/                Datasets (gitignored, see below)
```

## Datasets

Three real datasets drive the simulator. None are committed to this
repository; reacquire from the public sources below and place them under
`sim/data/` before running.

| Dataset | Used for | Source |
|---|---|---|
| **Shanghai Telecom** (Yu et al., IEEE TMC 2018) | Per-cell hourly arrival profile (top-7 cells, June 1-15 2014) | `https://wangshangguang.github.io/telecom_dataset/` |
| **Lumos5G** (Narayanan et al., ACM IMC 2020) | Per-slot mmWave service-rate jitter | `https://github.com/SIGCOMM21-Lumos5G/lumos5g` |
| **City Cellular Traffic Map** (Chen et al., 2015) | Fallback arrival-rate trace | `https://github.com/caesar0301/city-cellular-traffic-map` |

Cached normalized profiles (`arrivals_cache.npz`, `lumos5g_cache.npz`)
are written under `sim/data/` after first load. Delete them to re-derive
from the raw inputs.

## Reproducing the headline numbers

Requires Python 3.10+, PyTorch 2.x, NumPy, pandas, openpyxl, matplotlib.

```bash
# Headline 10-seed sweep (Safe-RL + Unconstrained PPO trained separately,
# all four controllers evaluated)
python3 -m sim.run all --seeds 10 --updates 250

# Gamma-sensitivity Pareto sweep
python3 -m sim.gamma_sweep --gammas 1.5 2.0 3.0 4.5 7.0 --seeds 4 --updates 150

# Figures and tables
python3 -m sim.make_figures
python3 -m sim.make_tables
```

Total wall-clock on a single CPU (no GPU needed): roughly 8 minutes for
the headline sweep plus 6 minutes for the Gamma sweep.

The canonical seed sequence is `numpy.random.SeedSequence(20260601)`,
expanded to ten 32-bit integers used across all experiments.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@article{Dong_SafeRL_TCCN_2026,
  author  = {Liang Dong},
  title   = {Lyapunov-Guided Safe Reinforcement Learning for Energy-Aware
             {O-RAN} Scheduling under Tail-Risk Constraints},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  year    = {2026},
  note    = {Submitted.}
}
```
