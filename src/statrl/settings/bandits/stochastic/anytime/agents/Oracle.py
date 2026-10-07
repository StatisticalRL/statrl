
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv


class Oracle(BanditAgent):
    """Baseline that always plays the best arm.

    The oracle knows the arm means and so incurs no regret. 

    Parameters
    ----------
    env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        The environment.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.agents._Random.Random :
        The opposite baseline, which never exploits.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> oracle = Oracle(BernoulliBandit([0.2, 0.9, 0.5]))
    >>> oracle.select_arm()
    1
    """

    def __init__(self, env: StochasticBanditEnv) -> None:
        self.env=env
        BanditAgent.__init__(self, name="Oracle")

    @property
    def policy(self) -> list[int]:
        """list of int: The optimal arm, as a one-element list.

        Written to the experiment logfile by
        :func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`.
        """
        return [self.env.optimal_arm]

    def reset(self) -> None:
        """Start a new run. The oracle keeps no statistics."""
        pass

    def select_arm(self) -> int:
        """Play the arm with the highest mean.

        Returns
        -------
        int
            ``env.optimal_arm``.
        """
        return self.env.optimal_arm

    def update(self, arm: int, reward: float) -> None:
        """Ignore the observed reward (the oracle has nothing to learn).

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward.
        """
        pass