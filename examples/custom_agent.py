"""
Writing your own bandit agent
=============================

Implements a greedy agent and an epsilon-greedy one, then benchmarks both
against IMED. The point of the comparison is what greedy gets wrong: with no
exploration it can lock onto an arm that looked good early and never revisit
that decision, so its regret grows linearly on some runs and stays near zero on
others. Averaged over replicates that shows up as a wide quantile band — which
is exactly what the shaded regions in the output figure display.

Run with::

    python examples/custom_agent.py

Writes a logfile and regret figures under ``results/``.
"""

import numpy as np

from statrl.experiments.massiveruns import runLargeMulticoreExperiment
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.bandits.stochastic.anytime.agents.Oracle import Oracle
from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
from statrl.settings.utils import klBern, randmax


class Greedy(BanditAgent):
    """Always play the arm with the best empirical mean.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    name : str, default='Greedy'
        Label used in logfiles and plot legends.
    """

    def __init__(self, nbArms, name="Greedy"):
        self.nbArms = nbArms
        super().__init__(name=name)

    def reset(self):
        """Clear every statistic before a new run."""
        super().reset()
        self.counts = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)

    def select_arm(self):
        """Play the arm of highest empirical mean, breaking ties at random.

        Returns
        -------
        int
            Index of the selected arm.
        """
        # randmax, not np.argmax: at t=0 every mean is 0, and argmax would
        # always return arm 0, so the agent would never try anything else.
        return randmax(self.means)

    def update(self, arm, reward):
        """Fold one reward into the running mean of its arm.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for it.
        """
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]


class EpsilonGreedy(Greedy):
    """Greedy, but explore uniformly with probability ``epsilon``.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    epsilon : float, default=0.1
        Probability of ignoring the empirical means and drawing at random. A
        constant rate keeps exploring forever, so the regret stays linear —
        with a much smaller slope than greedy's worst case.
    name : str, default='eps-Greedy'
        Label used in logfiles and plot legends.
    """

    def __init__(self, nbArms, epsilon=0.1, name="eps-Greedy"):
        self.epsilon = epsilon
        super().__init__(nbArms, name=name)

    def select_arm(self):
        """Explore with probability ``epsilon``, otherwise play greedily.

        Returns
        -------
        int
            Index of the selected arm.
        """
        if self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(self.nbArms))
        return randmax(self.means)


def main():
    env = BernoulliBandit([0.2, 0.9, 0.5])
    nA = env.number_arms

    runLargeMulticoreExperiment(
        env,
        agents=[
            IMED(nA, kullback=klBern),
            EpsilonGreedy(nA, epsilon=0.1),
            Greedy(nA),
        ],
        oracle=Oracle(env),
        interact=BanditInteraction(),
        timeHorizon=2000,
        nbReplicates=50,
        root_folder="results/",
    )


if __name__ == "__main__":
    main()
