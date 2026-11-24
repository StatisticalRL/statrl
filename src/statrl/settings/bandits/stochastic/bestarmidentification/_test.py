
"""Smoke tests and small experiments for stochastic BAI agents.

The BAI environment currently reuses the stochastic environments registered
in the ``anytime`` setting. The interaction protocol, however, is specific to
best-arm identification and is therefore exercised through ``BAIInteraction``.
"""

from pathlib import Path

from statrl.settings.bandits.stochastic.anytime.envs.parametric import (
    BernoulliBandit,
)

from statrl.settings.bandits.stochastic.bestarmidentification.interaction import (
    BAIInteraction,
)
from statrl.experiments.massiveruns import runLargeMulticoreExperiment


from statrl.settings.bandits.stochastic.bestarmidentification.agents._Oracle import _BAIOracle

from statrl.settings.bandits.stochastic.bestarmidentification.agents.TopTwoTS import TopTwoTS


def _small_environment():
    """Return a small Bernoulli instance with a unique best arm."""
    env = BernoulliBandit([0.2, 0.9, 0.7, 0.5])
    env.displayname = "Bernoulli BAI (4 arms)"
    return env


def test_render() -> None:
    """Render a short TTTS run to inspect the interaction loop."""
    env = _small_environment()
    agent = TopTwoTS(env.number_arms, beta=0.5, name="TTTS-render")
    interaction = BAIInteraction()

    interaction.renderrun(env, agent, horizon=10)


def test_run() -> None:
    """Run TTTS for a short fixed budget and print its identification scores."""
    env = _small_environment()
    agent = TopTwoTS(env.number_arms, beta=0.5, name="TTTS-run")
    interaction = BAIInteraction()

    scores = interaction.run(env, agent, horizon=100)
    print(f"{env.name}:{agent.name}: {scores}")
    print(f"Final recommendation: {agent.recommend_arm()}")
    print(f"True best arm: {env.optimal_arm}")


def test_load() -> None:
    """Load a registered stochastic environment and run the BAI interaction.

    BAI currently reuses the environment registry from the ``anytime``
    setting: its environment class is explicitly a re-export of the same
    ``StochasticBanditEnv`` implementation.
    """
    from statrl.experiments.utils import load, make

    settings_root = Path(__file__).resolve().parents[1]
    env_registry = (
            settings_root / "anytime" / "envs" / "environments.yaml"
    )
    envs = load(str(env_registry))
    env = make(envs["bernoulli_hard"])

    agent = TopTwoTS(env.number_arms, beta=0.5, name="TTTS-load")
    interaction = BAIInteraction()
    scores = interaction.run(env, agent, horizon=200)

    assert scores.shape == (200,)
    assert set(scores).issubset({0.0, 1.0})


def test_algos() -> None:
    """Instantiate TTTS through StatRL's algorithm factory and run it."""
    from statrl.experiments.utils import load,makealgo

    env = _small_environment()

    interaction = BAIInteraction()

    algos = load("agents/agents.yaml")
    for alg in algos:
        agent = makealgo(algos[alg], env)
        interaction.renderrun(env, agent, horizon=200)


def test_massive() -> None:
    """Run a small parallel BAI benchmark against the known-best-arm oracle.

    Increase ``timeHorizon`` and ``nbReplicates`` for a more reliable
    empirical comparison. This test writes its usual experiment outputs to
    ``results/`` through ``runLargeMulticoreExperiment``.
    """
    env = _small_environment()
    interaction = BAIInteraction()

    agents = [
        TopTwoTS(env.number_arms, beta=0.5, name="TTTS-beta-0.5"),
        TopTwoTS(env.number_arms, beta=0.7, name="TTTS-beta-0.7"),
    ]
    oracle = _BAIOracle(env)

    runLargeMulticoreExperiment(
        env,
        agents,
        oracle,
        interaction,
        timeHorizon=100,
        nbReplicates=50,
    )

if __name__ == "__main__":
    #test_render()
    #test_run()
    #test_load()
    #test_algos()
    test_massive()