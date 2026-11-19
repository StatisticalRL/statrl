import numpy as np


from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.utils import randmax
from math import sqrt, log

def polynomial_delta(exponent: float = 3):
    return lambda t: 1.0 / (t + 1) ** exponent

class UCB(BanditAgent):
    """
    Upper Confidence Bound (UCB)

    The UCB algorithm is a classical optimism-in-the-face-of-uncertainty
    strategy for stochastic multi-armed bandits.

    Each arm is assigned an index composed of:
        - empirical mean reward
        - an exploration bonus derived from concentration inequalities

    The arm with the highest index is selected. Here we assume sigma-sub-Gaussian distribution. ([0,1]-bounded distributions are 1/2_sub-Gaussian)

    Attributes
    ----------
    delta : callable
        Function mapping time step to confidence parameter δ(t).
    time : int
        Current time step.
    nbDraws : np.ndarray
        Number of pulls per arm.
    cumRewards : np.ndarray
        Cumulative reward per arm.
    means : np.ndarray
        Empirical mean reward per arm.
    indexes : np.ndarray
        UCB index per arm.
    """

    def __init__(self, nbArms, delta=None, sigma=0.5, name="UCB"):
        """
        Parameters
        ----------
        nbArms : int
            Number of arms.
        delta : callable
            Confidence schedule function δ(t).
        """
        if delta is None:
            self.delta = lambda t: 1./(t+1)**3
        else:
            self.delta = delta
        self.sigma=sigma
        self.nA=nbArms
        if name is None:
            BanditAgent.__init__(self, name="UCB")
        else:
            BanditAgent.__init__(self, name=name)

    def reset(self):
        """
        Reset internal statistics before a new run.
        """
        self.time = 0
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)
        self.means = np.zeros(self.nA)
        self.indexes = np.zeros(self.nA)

    def select_arm(self):
        """
        Select the arm with the highest UCB index.

        Returns
        -------
        int
            Index of selected arm.
        """
        return randmax(self.indexes)

    def update(self, arm, reward):
        """
        Update internal statistics after receiving a reward.

        Parameters
        ----------
        arm : int
            Selected arm index.
        reward : float
            Observed reward.

        Procedure
        ---------
        1. Increment global time step
        2. Update cumulative reward and pull count
        3. Update empirical mean
        4. Recompute UCB index for sigma-sub-Gaussian distribution.
        """
        self.time += 1

        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1

        # Empirical mean update
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        # UCB index computation
        self.indexes = np.array([
            (
                self.means[a]
                + self.sigma*sqrt(2*log(1 / self.delta(self.time)) /  self.nbDraws[a])
            ) if self.nbDraws[a] > 0 else np.inf
            for a in range(self.nA)
        ])