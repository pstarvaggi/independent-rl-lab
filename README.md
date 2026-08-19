# rl-lab

`rl-lab` is an independent laboratory for reinforcement-learning research: a
small, explicit scientific codebase for asking difficult questions about
stochastic and nonstationary decision processes. It favors inspectable Bellman
operators, reproducible runs, raw diagnostics, and ordinary Python over opaque
training frameworks.

The first laboratory is a finite stochastic maze whose action channel, walls,
rewards, hazards, observations, and regimes can each vary independently. Small
stationary instances expose their exact MDP, so empirical learning can be
compared with ground truth rather than only with another estimator.

## What is included

- A Gymnasium-compatible stochastic maze with dynamic topology, heterogeneous
  action noise, hazards, reward noise, nonstationarity, and optional partial
  observations.
- Dense exact-model construction for tractable stationary mazes, plus augmented
  Markov-wall models when their exponential state space is small enough.
- Transparent value/policy iteration and tabular SARSA, Expected SARSA,
  Q-learning, and Double Q-learning implementations.
- Epsilon-greedy, scheduled epsilon, Boltzmann, and UCB-style exploration.
- Configuration-driven seed and parameter sweeps with provenance-preserving,
  machine-readable results.
- Protocol-v2 run artifacts with versioned manifests, per-trial shards, source
  fingerprints, bounded step retention, and isolated policy evaluation.
- Step-, episode-, and snapshot-level diagnostics, including TD-error
  distributions, visitation, empirical models, and errors against exact
  solutions.
- Scientific plots for maze topology, policies, values, action values,
  visitation, transition noise, TD error, learning curves, and seed-level
  distributions.
- Six executable notebooks that connect the mathematics to the implementation.

## Installation

Python 3.12 or newer is required. The core test matrix covers Python 3.12 and
3.13. Create the environment with a native interpreter for the machine that will
run the experiments.

On Apple silicon, first verify that both the shell and Python report `arm64`:

```bash
uname -m
python3.12 -c "import platform; print(platform.machine())"
```

If either command reports `x86_64`, the interpreter is running under Rosetta.
Install or select a native arm64 Python before creating the environment. When
the repository is inside an iCloud-synced Desktop or Documents folder, keep the
environment outside that tree:

```bash
python3.12 -m venv ~/.virtualenvs/independent-rl-lab
source ~/.virtualenvs/independent-rl-lab/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,parquet]"
python -m ipykernel install --user --name rl-lab --display-name "Python 3 (rl-lab)"
```

For a local, nonsynchronized checkout, `.venv` is also supported and ignored by
Git. Do not copy or synchronize a virtual environment between machines or CPU
architectures. Cloud synchronization can leave package files partially restored
even when the project source is intact.

PyArrow is optional: without it, the runner stores tidy tables as CSV and notes
that choice in metadata. PyTorch, Jupyter, Box2D, and Pygame are confined to the
`notebooks` extra; the tabular laboratory does not require them. Box2D wheels are
available for the notebook-tested Python 3.12/3.13 range; Python 3.14 may require
SWIG and a source build.

The dependency markers also handle Intel Python on macOS, where the newest
published PyTorch wheel is 2.2.x: that combination uses NumPy 1.26 to preserve
binary compatibility. Native Apple-silicon and other supported interpreters use
the newer PyTorch and NumPy 2 lines. Install the declared extras as a unit rather
than upgrading NumPy independently.

## First run

Run the complete repository checks and a bounded installation smoke experiment:

```bash
make check
python experiments/stochastic_maze/run_stochasticity_sweep.py --quick
```

The checked-in stochasticity config is a research-sized sweep, not an
installation test. Run it deliberately:

```bash
rl-lab run configs/stochasticity_sweep.yaml
```

Protocol-v2 configs separate the scientific `experiment` from
`policy_evaluation`, execution policy, and artifact capture.
The example retains a deterministic sample of raw steps while preserving all
online episode and state-action summaries.

Results are written beneath `results/` in a unique run directory. The normalized
configuration, Git and dirty-tree state, source/runtime fingerprints, package
versions, trial metadata, committed table shards, and Q snapshots travel
together. Raw result directories are generated evidence and are intentionally
ignored by Git; version the config and source that reproduce them.

Protocol-v2 runs contain thousands of small shards. If the repository is inside
an iCloud-synced Desktop or Documents folder, write large runs to a local,
nonsynchronized volume so macOS cannot evict individual shards between training
and analysis:

```bash
rl-lab run configs/stochasticity_sweep.yaml \
  --output /path/to/local/rl-lab-results
```

For notebook-created runs, set `RL_LAB_NOTEBOOK_RESULTS` to the same local
results root before starting Jupyter. An existing iCloud-backed run must be fully
downloaded before its manifests and Parquet tables can be validated.

If a run is interrupted or a configured `continue` policy leaves failed trials,
retry only unfinished work against the same plan:

```bash
rl-lab run configs/stochasticity_sweep.yaml --resume results/q_learning_stochasticity-...
```

The manifest reconciles already committed attempts, creates a new numbered
attempt for pending/failed trials, and excludes partial-attempt rows from reads.

Start JupyterLab with:

```bash
jupyter lab
```

Read the notebooks in order:

1. `00_rl_primer.ipynb` derives finite-state methods and implements DQN,
   REINFORCE, advantage actor--critic, DDPG, and SAC directly.
2. `01_stochastic_maze.ipynb` isolates each source of environmental
   stochasticity and compares exact optimal policies.
3. `02_q_learning_experiments.ipynb` studies convergence across movement
   reliability and a spatially heterogeneous risky-versus-safe maze.
4. `03_lunar_lander_sac.ipynb` trains continuous Lunar Lander with a transparent
   SAC implementation, paired-seed benchmarks, diagnostics, and an animated
   deterministic landing replay with flight telemetry.
5. `04_policies_under_risk_drift_and_memory.ipynb` compares tabular backup rules
   under route risk, aligns TD errors around repeated reliability shifts, and
   separates algorithm error from missing Markov-wall state.
6. `05_shortcut_or_shelter.ipynb` turns one route-choice question into a
   publication-shaped study: it locates exact and learned reliability boundaries
   for recoverable and lethal hazards, and distinguishes frozen-greedy deployment
   from continued epsilon-soft exploration.

Notebook 05 is a results notebook. A normal **Run All** loads its checked compact
tables and figures from `reports/shortcut_or_shelter/` in seconds; it neither
retrains the agents nor scans the multi-gigabyte Protocol-v2 run directories.
To rebuild that evidence package from the immutable raw runs, use the explicit
standalone analysis command shown near the end of the notebook (and reproduced
below after the three run commands).

The full stationary risk comparison used by notebook 04 is also a versioned
command-line experiment:

```bash
rl-lab run configs/heterogeneous_routes.yaml --dry-run
rl-lab run configs/heterogeneous_routes.yaml --allow-large-run
```

Notebook 05 has separate full configs because its recoverable and lethal hazard
models require very different reliability grids. Both compare Q-learning,
SARSA, and Expected SARSA with the same 100,000-interaction budget per trial and
the same 20 root seeds. All state-action values start at the route-neutral upper
bound of `8.0`; this declared optimistic initialization prevents an early random
route choice from starving the other route of coverage. Persistent exploration
is epsilon `0.10`, and the matched constant learning rate is alpha `0.05`. Their
frozen-greedy and continuing-behavior evaluations reuse
a paired held-out seed panel at the final checkpoint. Intermediate learning
dynamics come from the much cheaper Q snapshots because episode-indexed policy
evaluation is not comparable across fixed-step trials. Raw step rows are not
retained; episode summaries, exact references, checkpoints, and Q snapshots are.

Run a bounded end-to-end smoke version of all three studies first:

```bash
python experiments/stochastic_maze/run_shortcut_or_shelter.py --quick
```

Inspect the full resource bounds before launching either primary sweep:

```bash
rl-lab run configs/shortcut_or_shelter_recoverable.yaml --dry-run
rl-lab run configs/shortcut_or_shelter_lethal.yaml --dry-run
```

At the checked-in design, the recoverable sweep is 108 million interactions and
the lethal sweep is 84 million. These are deliberate long runs, not sensible
first notebook cells. Each primary sweep intentionally exceeds the default
transition safety limit.
After reviewing the estimates, run it with explicit acknowledgement:

```bash
rl-lab run configs/shortcut_or_shelter_recoverable.yaml --allow-large-run
rl-lab run configs/shortcut_or_shelter_lethal.yaml --allow-large-run
```

The smaller, pre-specified sensitivity study anneals exploration toward zero at
two conditions where the greedy and persistent-epsilon objectives disagree:

```bash
rl-lab run configs/shortcut_or_shelter_annealed.yaml --dry-run
rl-lab run configs/shortcut_or_shelter_annealed.yaml
```

Rebuild the compact Notebook 05 evidence package only when auditing the analysis
or replacing one of those completed runs:

```bash
python experiments/stochastic_maze/analyze_shortcut_or_shelter.py \
  --recoverable-run results/shortcut_or_shelter_recoverable-dea8b3bb98-20260818T165653.284759Z \
  --lethal-run results/shortcut_or_shelter_lethal-c224eb9e19-20260818T184221.884416Z \
  --annealed-run results/shortcut_or_shelter_annealed-7a6ceda8a8-20260818T200356.410162Z
```

### Notebook and generated-artifact policy

`scripts/build_notebooks.py` is the source of truth for notebook cells. The six
canonical `notebooks/*.ipynb` files are checked-in, deterministic, unexecuted
build products: they should have no execution counts or cell outputs. Edit the
generator, run `make notebooks`, and commit the generator and regenerated
notebooks together. `make notebooks-check` fails when either the notebook
inventory or content is stale.

Execute a copy when exploring. Executed copies belong under
`notebooks/executed/`; experiment tables, checkpoints, images, and animations
belong under `notebooks/results/` or the root `results/` directory. All three
locations are ignored. Do not overwrite a canonical notebook with execution
state, and do not commit a result merely because Jupyter embedded it. A
publication snapshot should be exported deliberately to an external release or
archive with its versioned config and Protocol-v2 run metadata.

## Repository map

```text
rl-lab/
├── configs/                    # versioned experiment specifications
├── experiments/stochastic_maze # reproducible research entry points
├── notebooks/                  # mathematical, executable narratives
├── scripts/                    # deterministic artifact builders
├── src/rllab/
│   ├── agents/                 # policies and learning rules
│   ├── environments/           # stochastic processes only
│   ├── evaluation/             # comparison with exact/reference solutions
│   ├── experiments/            # interaction loop, sweeps, provenance, I/O
│   ├── metrics/                # recorders, TD statistics, aggregation
│   ├── theory/                 # finite MDPs and exact Bellman solvers
│   ├── utils/                  # deterministic seeding and small helpers
│   └── visualization/          # scientific plots and animations
├── tests/
└── results/                    # ignored generated artifacts
```

`src/rllab/` and `experiments/` serve different purposes. `src/rllab/` is the
installed, reusable library: environments, algorithms, statistics, artifact
schemas, and plotting APIs live there and must not depend on a particular study.
`experiments/` contains thin, versioned entry points for named research runs. An
entry point may select a config or apply a documented smoke override, but it
should not duplicate an agent, environment, metric, or persistence implementation
from `src/rllab/`. Declarative study definitions live in `configs/`; notebooks
consume those public layers rather than becoming a second library.

The dependency direction is deliberate: an environment defines a stochastic
process; an agent sees observations and transitions; a runner connects them;
metrics observe the interaction; evaluation adds reference quantities; and
visualization consumes recorded results. Environments do not know about agents,
and agents do not write files or draw plots.

## Reproducibility philosophy

A seed is part of a result, not a convenience flag. Each trial derives separate
random streams for the environment, agent, and evaluation. A sweep preserves all
seed-level trajectories and only then computes uncertainty summaries. Exact
solutions are generated only when assumptions justify them; unsupported dynamic
or partially observed cases fail explicitly rather than returning a mislabeled
"ground truth."

Protocol-v2 trial attempts become visible only after their atomic commit record
is written. Committed attempts are immutable; incomplete attempts may be retried
in a new attempt directory, and the parent manifest selects the successful one.
Do not hand-edit tables or manifests. Change a versioned config and start a new
run whenever the scientific condition changes.

## Adding an agent

Implement the small tabular-agent contract in `src/rllab/agents`: initialize the
state/action dimensions, implement action selection, return an update record from
the learning step, and expose current Q-values when meaningful. Put exploration
logic in an exploration strategy or schedule rather than inside the experiment
runner. Register the new kind in the runner factory and add a one-transition
update test before starting a sweep.

Adaptive research ideas can override the update while reusing instrumentation.
For example, a state-action learning rate may depend on visits, rolling TD-error
variance, or a drift statistic without requiring a new environment or logger.

## Adding an environment

Implement Gymnasium's `reset` and `step`, with all stochasticity driven by the
environment's seeded generator. Put scientifically useful latent state—regime,
wall realization, structural events, hazard positions—in the `info` mapping.
If an exact finite model exists, return an explicit `FiniteMDP`; document the
stationarity and observability assumptions under which it is valid.

## Quality checks

The canonical local gate is:

```bash
make check
```

It runs, in order, the non-mutating formatter check, lint, type checking,
deterministic notebook freshness check, and the coverage-enabled test suite. The
individual targets are `make format-check`, `make lint`, `make typecheck`,
`make notebooks-check`, and `make test`. Use `make format` and `make notebooks`
only when you intend to update files.

The tests include exact transition checks, seeded stochastic behavior, wall
dynamics, Gymnasium compliance, Bellman convergence, agent updates, persistence,
metric aggregation, and plotting smoke tests. Stochastic assertions use exact
kernels or statistically justified tolerances rather than brittle single draws.
CI applies the same gates on Python 3.12/3.13 and runs the core suite plus the
Box2D/SAC notebook workflow on a native arm64 macOS worker.

## Scope

This first version intentionally does not add a distributed scheduler, database,
experiment-tracking service, or a deep-agent hierarchy. The stored schema and
component boundaries leave room for those choices later. The immediate goal is
more fundamental: make unusual RL experiments easy to express and hard to
misinterpret.
