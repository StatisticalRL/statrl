
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv

import numpy as np
class Random(BanditAgent):
    """Uniform exploration: pull an arm uniformly at random every round.

    Parameters
    ----------
    env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        The environment, read only for its number of arms.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.agents.Oracle.Oracle :
        The opposite baseline, which always plays the best arm.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> agent = Random(BernoulliBandit([0.2, 0.9, 0.5]))
    >>> agent.reset()
    >>> agent.select_arm() in (0, 1, 2)
    True
    """

    def __init__(self, env: StochasticBanditEnv) -> None:
        self.env= env
        BanditAgent.__init__(self, name="Random")

    def reset(self) -> None:
        """Start a new run. No statistics are kept, so this does nothing."""

    def select_arm(self) -> int:
        """Draw an arm uniformly at random.

        Returns
        -------
        int
            An index drawn uniformly from ``range(env.number_arms)``.
        """
        return np.random.randint(self.env.number_arms)

    def update(self, arm: int, reward: float) -> None:
        """Ignore the observed reward (this agent does not learn).

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward.
        """