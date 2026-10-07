




from statrl.settings.bandits.stochastic.batch.agent import BatchBanditAgent
import numpy as np
class Random(BatchBanditAgent):
    """Uniform exploration: fill each batch with independent random arms. 

    Parameters
    ----------
    env : BatchMAB
        The environment, read only for its number of arms.

    See Also
    --------
    statrl.settings.bandits.stochastic.batch.agents.Oracle.Oracle :
        The opposite baseline.
    """

    def __init__(self, env) -> None:
        self.env= env
        BatchBanditAgent.__init__(self, name="Random")

    def reset(self) -> None:
        """Start a new run. No statistics are kept, so this does nothing."""
        pass

    def play(self) -> int:
        """Draw an arm uniformly at random.

        Returns
        -------
        int
            An index drawn uniformly from the arms of the wrapped bandit.
        """
        return np.random.randint(self.env.mab.number_arms)

    def update(self, arm: int, reward: float) -> None:
        """Ignore the observed reward (this agent does not learn).

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward.
        """
        pass

    def batchplay(self,batchsize):
        """Fill the batch with independent uniform draws.

        Parameters
        ----------
        batchsize : int
            Number of pulls in this batch.

        Returns
        -------
        list of int
            ``batchsize`` independently drawn arm indices.
        """
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        """Ignore the batch's rewards (this agent does not learn).

        Parameters
        ----------
        batcharm : list of int
            The arms that were pulled.
        batchreward : list of float
            The rewards observed for them.
        """
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)


