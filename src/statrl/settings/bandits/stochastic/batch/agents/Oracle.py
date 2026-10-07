


from statrl.settings.bandits.stochastic.batch.agent import BatchBanditAgent

class Oracle(BatchBanditAgent):
    """Baseline filling every batch with the best arm. 

    Parameters
    ----------
    env : BatchMAB
        The environment, read for its ``optimal_arm``.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.batch.environment import BatchMAB
    >>> Oracle(BatchMAB(BernoulliBandit([0.2, 0.9]), [3])).batchplay(3)
    [1, 1, 1]
    """

    def __init__(self,env):
        self.env=env
        BatchBanditAgent.__init__(self, name="Oracle")


    @property
    def policy(self) -> list[int]:
        """list of int: The optimal arm, as a one-element list.

        Written to the experiment logfile by
        :func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`.
        """
        return [self.env.optimal_arm]


    def reset(self) -> None:
        """Start a new run. The oracle keeps no statistics, so this is a no-op."""
        pass

    def play(self) -> int:
        """Return the arm with the highest mean.

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

    def batchplay(self,batchsize):
        """Fill the whole batch with the optimal arm.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch.

        Returns
        -------
        list of int
            ``batchsize`` copies of ``env.optimal_arm``.
        """
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        """Ignore the batch's rewards (the oracle has nothing to learn).

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled.
        batchreward : list of float
            The rewards observed for them.
        """
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)
