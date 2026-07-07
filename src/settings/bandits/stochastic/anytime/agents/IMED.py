import numpy as np
from src.settings.bandits.stochastic.anytime.agent import BanditAgent
from src.settings.utils import randmin, klGauss
from math import log


class IMED(BanditAgent):
    """
    Indexed Minimum Empirical Divergence (IMED)

    IMED is an index-based algorithm for stochastic multi-armed bandits,
    designed to balance exploration and exploitation using a divergence-based index.

    The key idea is to assign to each arm an index of the form:

        I_a(t) = N_a(t) * KL(μ_a(t), μ*(t)) + log(N_a(t))

    where:
        - N_a(t) is the number of pulls of arm a
        - μ_a(t) is the empirical mean reward of arm a
        - μ*(t) is the best empirical mean across arms
        - KL(·,·) is a divergence function (typically Bernoulli KL)

    The algorithm selects the arm with the minimal index.

    Attributes
    ----------
    kl : callable
        Kullback-Leibler divergence function (problem-dependent).
    nbDraws : np.ndarray
        Number of times each arm has been selected.
    cumRewards : np.ndarray
        Cumulative reward per arm.
    means : np.ndarray
        Empirical mean reward per arm.
    maxMeans : float
        Maximum empirical mean across all arms.
    indexes : np.ndarray
        IMED index value per arm.
    """

    def __init__(self, nbArms, kullback=klGauss):
        """
        Parameters
        ----------
        nbArms : int
            Number of available arms.
        kullback : callable
            Function computing KL divergence between two scalar means.
        """
        self.kl = kullback
        self.nA = nbArms
        BanditAgent.__init__(self, name="IMED")

    def reset(self):
        """
        Reset internal statistics before a new run.
        """
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)
        self.means = np.zeros(self.nA)
        self.maxMeans = 0.0
        self.indexes = np.zeros(self.nA)

    def play(self, state=0):
        """
        Select the next arm to pull.

        Returns
        -------
        int
            Index of selected arm.

        Decision rule
        -------------
        Select arm with minimum IMED index:
            argmin_a I_a(t)
        """
        return randmin(self.indexes)

    def update(self, arm, reward):
        """
        Update internal statistics after observing a reward.

        Parameters
        ----------
        arm : int
            Selected arm index.
        reward : float
            Observed reward.

        Procedure
        ---------
        1. Update cumulative reward and pull count
        2. Recompute empirical means
        3. Update best empirical mean
        4. Recompute IMED indices
        """
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1

        # Empirical mean update
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        # Best empirical mean across arms
        self.maxMeans = np.max(self.means)

        # IMED index computation
        self.indexes = np.array([
            (self.nbDraws[a] * self.kl(self.means[a], self.maxMeans)
             + log(self.nbDraws[a])) if self.nbDraws[a] > 0 else 0
            for a in range(self.nA)
        ])