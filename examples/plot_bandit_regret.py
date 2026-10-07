"""
Comparing bandit algorithms on a Bernoulli instance
===================================================

Benchmarks IMED against uniform exploration on a three-armed Bernoulli bandit
and plots the regret of each, averaged over replicates.

This is the shape of every experiment in ``statrl``: build an environment, list
the agents to compare, add an oracle to define zero regret, and hand the four to
:func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`.

Run with::

    python examples/plot_bandit_regret.py

Writes a logfile and regret figures under ``results/``.
"""

from statrl.experiments.massiveruns import runLargeMulticoreExperiment
from statrl.settings.bandits.stochastic.anytime.agents.Oracle import Oracle
from statrl.settings.bandits.stochastic.anytime.agents._Random import Random
from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
from statrl.settings.utils import klBern, klGauss


def main():
    # Arm 1 is best. The gap to arm 2 is 0.4, wide enough that a good algorithm
    # resolves it quickly and the regret curve flattens within the horizon.
    env = BernoulliBandit([0.2, 0.9, 0.5])
    nA = env.number_arms

    agents = [
        # Two divergences on the same rewards: klBern is the matched choice for
        # Bernoulli arms, klGauss a valid but looser one. Distinct names, or the
        # two collide in the dump filenames and the legend.
        IMED(nA, kullback=klBern, name="IMED-Bernoulli"),
        IMED(nA, kullback=klGauss, name="IMED-Gaussian"),
        Random(env),
    ]

    runLargeMulticoreExperiment(
        env,
        agents=agents,
        oracle=Oracle(env),
        interact=BanditInteraction(),
        timeHorizon=2000,
        nbReplicates=50,
        root_folder="results/",
    )


if __name__ == "__main__":
    main()
