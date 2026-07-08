# statrl

[![Tests](https://github.com/StatisticalRL/statrl/actions/workflows/tests.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/tests.yml)
[![Lint](https://github.com/StatisticalRL/statrl/actions/workflows/lint.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/lint.yml)
[![Documentation](https://github.com/StatisticalRL/statrl/actions/workflows/docs.yml/badge.svg)](https://github.com/StatisticalRL/statrl/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The Statistical Reinforcement Learning Toolkit is a research library organised as a taxonomy of *settings*, each with a matching
environment, agent, and interaction loop.

## Install

Requires Python 3.9+. Not yet on PyPI (install from source):

```bash
git clone https://github.com/StatisticalRL/statrl.git
cd statrl
pip install -e .          # editable install for development
```

## The protocol

Every bandit setting shares the same protocol:

| Component   | Responsibility                                                              |
| ----------- | ---------------------------------------------------------------------------- |
| Environment | Holds the reward distributions; `pull(arm)` samples a reward.         |
| Agent       | `reset()` starts a run; `select_arm()` chooses an arm; `update(arm, reward)` learns. |
| Interaction | `interact(env, learner, horizon)` runs the loop and returns cumulative scores. |

## Implemented settings

Under `statrl.settings.bandits`:

- **`stochastic.anytime`** 
- **`stochastic.knownhorizon`** : horizon-aware wrapper over the anytime setting.
- **`adversarial.lipschitz`** : an adversarial Lipschitz forecaster.

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

## Development

```bash
pip install -e .[test,lint]
pytest                  # run tests
ruff check src tests    # lint
mypy                    # type-check
```

## Documentation

Full docs (quickstart, user guide, API reference) live under `docs/`:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```
