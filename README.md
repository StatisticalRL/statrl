# statrl

[![Tests](https://github.com/StatisticalRL/statrl/actions/workflows/tests.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/tests.yml)
[![Lint](https://github.com/StatisticalRL/statrl/actions/workflows/lint.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/lint.yml)
[![Documentation](https://github.com/StatisticalRL/statrl/actions/workflows/docs.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The Statistical Reinforcement Learning Toolkit is a research library organised as a
taxonomy of *settings*, each with a matching environment, agent, and interaction loop.

📖 **[Documentation](https://statisticalrl.github.io/statrl/)**

## Install

Requires Python 3.14+. 

```bash
pip install statrl
```

Installation from sources

```bash
git clone https://github.com/StatisticalRL/statrl.git
cd statrl
pip install -e .          # editable install for development
```

## Quickstart

```python
from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
from statrl.settings.bandits.stochastic.anytime.agents._Oracle import Oracle
from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
from statrl.settings.utils import klBern

env = BernoulliBandit([0.2, 0.9, 0.5])
interaction = BanditInteraction()

scores = interaction.run(env, IMED(env.number_arms, klBern), horizon=2000)
oracle_scores = interaction.run(env, Oracle(env), horizon=2000)

print(f"regret after 2000 rounds: {oracle_scores[-1] - scores[-1]:.1f}")
```

## The protocol

Every setting shares the same protocol:

| Component   | Responsibility                                                                       |
| ----------- | ------------------------------------------------------------------------------------ |
| Environment | Holds the reward distributions; `step(arm)` samples a reward.                          |
| Agent       | `reset()` starts a run; `select_arm()` chooses an arm; `update(arm, reward)` learns.    |
| Interaction | `run(env, learner, horizon)` runs the loop and returns cumulative expected scores.      |

## Implemented settings

Under `statrl.settings`:

BANDITS:
- **`stochastic.anytime`**
- **`stochastic.knownhorizon`** : horizon-aware wrapper over the anytime setting.
- **`stochastic.batch`** : when considering batch schedule.
- **`stochastic.kernel`** : RKHS structure on arms.
- **`adversarial.lipschitz`** : an adversarial Lipschitz forecaster.

MARKOV DECISION PROCESSES:
- **`discrete_nostructure`**: Abstract discrete MDPs.
- **`gridworld`**: Gridworld MDPs.

## Running experiments

`statrl.experiments` benchmarks agents: many replicates in parallel, regret against an
oracle, and plots.

```python
from statrl.experiments.massiveruns import runLargeMulticoreExperiment

runLargeMulticoreExperiment(
    env, agents, oracle, interact,
    timeHorizon=1000, nbReplicates=100, root_folder="results/",
)
```

Results (per-replicate dumps, a logfile, regret plots) are written under `root_folder`.
See [`examples/`](examples/) for complete runnable scripts.

## Development

```bash
pip install -e ".[test,lint]"
pytest                                    # tests
pytest --doctest-modules src/statrl \
       --ignore-glob='*_test.py'          # docstring examples
ruff check src tests                      # lint
mypy                                      # type-check
```

## Documentation

Full docs (quickstart, user guide, API reference) live under `docs/`:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs/source docs/_build/html
sphinx-build -b doctest docs/source docs/_build/doctest
xdg-open docs/_build/html/index.html
```
