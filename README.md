# Safe-RL for O-RAN

Reference simulator and reproducible numerical results for the paper

> **Lyapunov-Guided Safe Reinforcement Learning and Risk-Budget Feasibility
> for Energy-Aware O-RAN Scheduling**
> Liang Dong, Senior Member, IEEE.
> Under review, *IEEE Transactions on Cognitive Communications and Networking*.

The codebase implements a primal-dual actor-critic for a K-cell O-RAN cluster
with inter-cell offloading. A PyTorch policy is constrained by a CVaR budget on
the per-slot loss (Rockafellar-Uryasev form, PID multiplier with anti-windup),
and every action passes through a Lyapunov-guided safety filter: a per-cell
lower-confidence-bound projection with a joint fallback that wakes a
backlogged cell and the neighbors that would offload onto it. Separate critics
are trained on the energy and risk channels.

## Second revision (R2): corrections

The R2 code and results replace those of the first revision. An audit of the
analysis and code found errors that affected several published claims; all
were fixed and every experiment was rerun. In brief:

* **Learned baselines.** PPO-Lagrangian's multiplier used the mean loss while
  its critic and actor used the CVaR surrogate; CRPO fed a routing multiplier
  of 1e6 to the networks as an input (2e4 after scaling), saturating every
  unit; WCSAC fitted one critic to two incompatible targets. All three are
  corrected (`sim/safe_baselines.py`), and their budgets are calibrated on
  validation seeds at the same tightness relative to always-on.
* **Safety filter.** It now predicts arrivals under the sleep pattern that will
  execute, overrides the hysteresis dwell for safety wakes, charges the wake-up
  service loss, and executes a joint fallback when its projection is
  infeasible, logging each case (`sim/safety_filter.py`). The earlier fallback
  raised only the backlogged cell's own share, which an all-sleep adversary
  defeats.
* **Other fixes.** Phase feature aligned with the compressed daily cycle;
  CVaR threshold step ordered after the risk cost; PID anti-windup; exact
  empirical CVaR at ties; correlated channel that preserves the Lumos5G
  marginal exactly; hexagonal topology for K = 19, 37, 61; loaders that refuse
  synthetic substitutes; the rule formerly called drift-plus-penalty renamed
  backlog-proportional, with a textbook drift-plus-penalty controller added.
* **Protocol.** Tuning and budget calibration use ten validation seeds; every
  reported number uses ten separate test seeds. Each learned controller is
  trained once per seed (fresh arrival realization per training episode), its
  checkpoint is saved, and all evaluations load it, so comparisons are paired.

## What the results show

Ten test seeds, 100 s evaluation episodes at 10 ms slots, seven cells,
`beta = 0.95`, `Gamma = 3.5`, 3200 training updates, 95% bootstrap intervals
over seeds:

| Controller | Power (W) | CVaR_0.95 | P{loss > Gamma} | Toggles/min |
|---|---|---|---|---|
| Always-on reference | 1610.0 | 3.050 | 0.7% | 0 |
| **Backlog-proportional, rho = 0.875** (validation-selected) | **1151.0** | **3.408** | **1.2%** | **0** |
| Textbook drift-plus-penalty (closest to budget) | 1314.2 | 3.668 | 2.0% | 3195 |
| PPO-Lagrangian + LCB filter | 1393.7 | 3.320 | 1.1% | 5 |
| CRPO + LCB filter | 1218.5 | 3.717 | 2.3% | 1022 |
| WCSAC-GS (PPO adaptation) + LCB filter | 986.5 | 5.905 | 47.7% | 5581 |
| Proposed | 1343.6 | 3.468 | 1.5% | 53 |

**The risk budget must be screened.** The loss contains an arrival term no
scheduler influences. A certified floor `Gamma_0 = 2.019` (valid for unequally
loaded cells when `eta * max(Abar) <= min(Abar)`, which holds on every trace)
rules out smaller budgets, and the always-on reference `Gamma_AO = 3.050` is a
feasible witness for larger ones. Budgets in between are undetermined.

**A simple rule is not beaten.** Among filtered learned methods whose mean tail
meets the budget, the proposed controller draws the least power, but the
backlog-proportional rule draws 192.7 W less (paired interval [162, 227]) at a
tail that is not statistically distinguishable. The textbook drift-plus-penalty
controller never meets the budget, because its bang-bang decision chatters.

**The filter carries the tail.** Evaluating the trained policy with the filter
bypassed raises CVaR from 3.468 to 7.280; the filter costs 23 W ([2, 50]) and
acts on 4.8% of cell-slots, falling back to full service in 94% of those. With
risk pressure removed, it holds time-average backlog below 13 Mb against an
adversary that asks every cell to sleep, versus up to 14,992 Mb without it.

The paper does not claim an O(1/V) energy-gap guarantee: a per-state safety
filter can carry an energy price that no Lyapunov weight removes, and the
earlier version's claim to the contrary was incorrect.

## Repository layout

```
sim/
├── config.py            Calibrated parameters (single source of truth)
├── arrivals.py          Per-cell arrival profiles from Shanghai Telecom
├── channel_lumos5g.py   Per-slot service-rate multiplier from Lumos5G
├── env.py               K-cell environment: queues, energy, offloading, channel
├── networks.py          Factored actor (sleep x share) and critics
├── safety_filter.py     LCB projection with joint fallback; aggregate comparison
├── algorithm.py         Primal-dual PPO, two critics, PID dual, CVaR threshold
├── baselines.py         Classical controllers and the shared evaluation routine
├── safe_baselines.py    PPO-Lagrangian, CRPO, WCSAC-GS on the same backbone
├── metrics.py           Exact empirical CVaR, bootstrap intervals
├── r2.py                R2 experiment suite (calibrate ... timing)
├── r2_scale_classical.py  Classical references on the larger clusters
├── make_r2_figures.py   TikZ/pgfplots figures from sim/results/r2
├── make_r2_tables.py    LaTeX table and paired statistics
├── make_tikz_figures.py Shared figure preamble (and the R1 figure set)
└── revision.py, ...     R1 drivers, kept for the record of the R1 results
fig/                     Figures (PDF) and their TikZ sources
tab/                     Generated LaTeX tables
sim/results/r2/          R2 results: JSON summaries, checkpoints, per-slot series
sim/results/             R1 results (superseded, kept for the record)
sim/data/                Datasets (gitignored, see below)
```

## Datasets

Two real datasets drive the experiments. Neither is committed; reacquire them
from the public sources below and place them under `sim/data/`.

| Dataset | Used for | Source |
|---|---|---|
| **Shanghai Telecom** (Wang et al., IEEE TMC 2021) | Per-cell hourly load profile (seven busiest cells, June 1-15, 2014) | `https://wangshangguang.github.io/telecom_dataset/` |
| **Lumos5G** (Narayanan et al., ACM IMC 2020) | Per-slot mmWave service-rate multiplier | `https://github.com/SIGCOMM21-Lumos5G/lumos5g` |

`config.require_real_data` defaults to `True`: both loaders raise rather than
substitute a synthetic profile or channel pool, including from a cached file,
and every result file records its data sources and a hash of the simulator
code under `_provenance`.

Arrivals are trace-shaped, not recorded slot by slot. Each episode compresses
one 24-hour profile onto its slots; the nominal load is a scaled Poisson draw
at `0.3 * h_b(t) / max(h_b)` Mb per slot (0.3 Mb is the busy-hour rate) plus
Pareto bursts (probability 0.04, shape 2.5). Realized cell means are 0.18-0.25
Mb per slot. The Lumos5G multiplier is the 68,118 records with a valid
throughput field, normalized by the median and clipped to `[0.1, 2.0]`; the clip
binds on 26.2% of samples above and 10.0% below.

## Reproducing the results

Requires Python 3.10+, PyTorch 2.x, NumPy, SciPy, pandas, openpyxl, and a LaTeX
installation with pgfplots for the figures. The reported runs used Python
3.12, PyTorch 2.11 and NumPy 2.2 on an Intel Core i7-12700KF.

```bash
# Full R2 campaign (about 3.5 hours on 20 CPU workers)
./sim/results/r2/run_all.sh

# or stage by stage
OMP_NUM_THREADS=1 python3 -m sim.r2 calibrate
OMP_NUM_THREADS=1 python3 -m sim.r2 feasible
OMP_NUM_THREADS=1 python3 -m sim.r2 classical
OMP_NUM_THREADS=1 python3 -m sim.r2 train      # 250 training runs
OMP_NUM_THREADS=1 python3 -m sim.r2 evaluate
OMP_NUM_THREADS=1 python3 -m sim.r2 stress
OMP_NUM_THREADS=1 python3 -m sim.r2 corr
OMP_NUM_THREADS=1 python3 -m sim.r2 timing
OMP_NUM_THREADS=1 python3 -m sim.r2_scale_classical

# Figures, table and paired statistics from sim/results/r2
python3 -m sim.make_r2_figures
python3 -m sim.make_r2_tables
```

Set `OMP_NUM_THREADS=1`: the networks are small enough that intra-op
threading costs more than it saves, and single-threaded reductions make a
rerun reproducible. The training stage skips any configuration whose final
checkpoint already exists, so the committed checkpoints let `evaluate` and the
later stages run without retraining. Test seeds are the first ten outputs of
`numpy.random.SeedSequence(20260601)` and validation seeds the next ten;
evaluation traces use `seed + 9000`.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@article{Dong_SafeRL_TCCN_2026,
  author  = {Liang Dong},
  title   = {Lyapunov-Guided Safe Reinforcement Learning and Risk-Budget
             Feasibility for Energy-Aware {O-RAN} Scheduling},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  year    = {2026},
  note    = {Under review.}
}
```
