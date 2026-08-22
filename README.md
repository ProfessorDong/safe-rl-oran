# Safe-RL for O-RAN

Reference simulator and reproducible numerical results for the paper

> **Lyapunov-Guided Safe Reinforcement Learning and Risk-Budget Feasibility
> for Energy-Aware O-RAN Scheduling**
> Liang Dong, Senior Member, IEEE.
> Under review, *IEEE Transactions on Cognitive Communications and Networking*.

The codebase implements a primal-dual actor-critic for a seven-cell O-RAN
cluster: a PyTorch policy is constrained by a risk virtual queue (CVaR on
per-slot loss via the Rockafellar-Uryasev epigraph form), a Lyapunov-guided
safety filter projects exploratory actions onto a backlog-aware safe set, and
a PID dual controller enforces the long-run risk constraint. Separate critics
are trained on the energy and risk channels so the value function a critic
must represent does not move with the multiplier.

## What the results actually show

Two findings organise the evaluation, and the second is not favourable to the
learned controller. Both are reported here as they came out.

**The risk budget is not freely choosable.** The per-slot loss contains an
arrival term no scheduling decision influences, so the attainable tail is
bounded below before any action is taken. On these traces the
policy-independent floor is `Gamma_0 = 2.019` and the floor attainable by the
always-on reference is `Gamma_min = 3.050`. A budget at or below that floor
leaves no Slater margin, and the resulting CVaR gap is easily mistaken for
poor tuning. The experiments therefore use `Gamma = 3.5`.

**A tuned classical controller is not beaten.** Sweeping the
drift-plus-penalty weight traces a frontier the learned controller does not
reach. Ten canonical seeds, 100 s evaluation episodes at 10 ms slots, seven
cells, `beta = 0.95`, `Gamma = 3.5`, 3200 training updates, 95% bootstrap
intervals:

| Controller | Power (W) | CVaR | Violation | Toggles/min |
|---|---|---|---|---|
| Always-on reference | 1610.0 | 3.050 | 0.7% | 0 |
| **Drift-plus-penalty, rho=0.75** | **1126.8** | **3.487** | **1.4%** | **0** |
| Drift-plus-penalty, rho=1 | 1173.8 | 3.350 | 1.1% | 0 |
| Safe-RL (proposed) | 1296.3 | 3.528 | 1.6% | 47 |
| WCSAC-GS + LCB filter | 1260.1 | 3.775 | 3.2% | 427 |
| CRPO + LCB filter | 1211.9 | 4.019 | 4.7% | 90 |

Among learned methods given the same safety filter, the proposed controller
holds the tightest tail and the least switching. Against tuned classical
control it does not: the cheapest drift-plus-penalty setting meeting the
budget draws 169.5 W less at a slightly tighter tail with no switching.

**The safety filter is what makes any of these work.** Evaluating one trained
policy with the projection engaged and bypassed gives CVaR 3.421 versus
10.000 (the loss cap) at essentially identical power -- the filter *saves*
11.9 W while intervening on only 4.9% of cell-slots.

## Repository layout

```
sim/
├── config.py            Calibrated parameters (single source of truth)
├── arrivals.py          Per-cell hourly arrivals from Shanghai Telecom
├── channel_lumos5g.py   Per-slot service-rate jitter from Lumos5G mmWave traces
├── env.py               Seven-cell environment: queues, energy, inter-cell offload
├── networks.py          Factored actor (discrete sleep x continuous throttle) + critics
├── safety_filter.py     LCB projection onto the backlog-aware safe set
├── algorithm.py         Primal-dual PPO, two critics, PID dual, risk virtual queue
├── baselines.py         Classical controllers (always-on, threshold, drift variants)
├── safe_baselines.py    CRPO, WCSAC-GS, PPO-Lagrangian on the same PPO backbone
├── metrics.py           CVaR, delay percentiles, bootstrap CI helpers
├── revision.py          Main experiment suite (see below)
├── frontier.py          Classical achievable frontier (rho and q_s swept)
├── attribution.py       Filter effect on a fixed policy (engaged vs bypassed)
├── dependence.py        Dependence realised by the correlated channel model
├── make_tikz_figures.py TikZ/pgfplots figures drawn at true column width
├── make_rev_figures.py  Matplotlib figure set (superseded by the TikZ set)
└── make_rev_tables.py   LaTeX tables, generated from stored results
fig/                     Generated figures (PDF) and their TikZ sources
tab/                     Generated LaTeX tables
sim/results/             Stored experiment results (JSON, committed)
sim/data/                Datasets (gitignored, see below)
```

Older single-experiment drivers (`run.py`, `gamma_sweep.py`, `ablations.py`,
`critic_sweep.py`, `verification.py`, `make_figures.py`, `make_tables.py`)
predate the revision suite and are retained for reference; the reported
numbers come from `revision.py` and the three standalone experiments.

## Datasets

Two real datasets drive the reported experiments. Neither is committed here;
reacquire from the public sources below and place them under `sim/data/`.

| Dataset | Used for | Source |
|---|---|---|
| **Shanghai Telecom** (Wang et al., IEEE TMC 2021) | Per-cell hourly arrival profile (busiest 7 cells, 1-15 June 2014) | `https://wangshangguang.github.io/telecom_dataset/` |
| **Lumos5G** (Narayanan et al., ACM IMC 2020) | Per-slot mmWave service-rate multiplier | `https://github.com/SIGCOMM21-Lumos5G/lumos5g` |

`config.require_real_data` defaults to `True`, so the loaders raise rather
than silently substitute a synthetic profile, and every stored result records
which dataset fed it under its `_provenance` key. A City Cellular Traffic Map
fallback path remains in `arrivals.py` but was not used for any reported
number.

The Lumos5G multiplier is the 68 118 records with a valid throughput field,
normalised by the trace median and clipped to `[0.1, 2.0]`. The clip is not
cosmetic: it binds on 26.2% of samples above and 10.0% below, so the pool is
a bounded proxy for the measured tail rather than the tail itself.

## Reproducing the results

Requires Python 3.10+, PyTorch 2.x, NumPy, pandas, openpyxl, matplotlib, and
a LaTeX installation with pgfplots for the figures.

```bash
# Full suite at the reported training budget (ten seeds, 3200 updates).
# Sub-commands: feasibility headline filter sensitivity scalability
#               stress vsweep correlation compute curve
OMP_NUM_THREADS=1 python3 -m sim.revision all --seeds 10 --updates 3200 --workers 20

# Standalone experiments
OMP_NUM_THREADS=1 python3 -m sim.frontier    --seeds 10
OMP_NUM_THREADS=1 python3 -m sim.attribution --seeds 5 --updates 3200
OMP_NUM_THREADS=1 python3 -m sim.dependence

# Figures and tables, generated from sim/results/*.json
python3 -m sim.make_tikz_figures
python3 -m sim.make_rev_tables
```

Set `OMP_NUM_THREADS=1`: the networks are small enough that intra-op
threading costs more in synchronisation than it recovers, and pinning it also
fixes the order of floating-point reductions so a rerun is bit-identical.

Wall-clock for the full suite on 20 CPU workers is roughly six hours,
dominated by `sensitivity` (~2.2 h), `headline` (~1 h) and `filter` (~1 h).
No GPU is used or needed; the per-update cost is dominated by launch latency
rather than arithmetic.

Training uses a 2000-slot arrival trace and evaluation a disjoint 10 000-slot
trace drawn from `seed + 9000`, so no evaluation slot is seen during training.
The canonical seed sequence is `numpy.random.SeedSequence(20260601)`, expanded
to ten 32-bit integers used across all experiments.

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
