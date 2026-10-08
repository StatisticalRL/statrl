import numpy as np


from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.utils import randmax

class NPTS_old(BanditAgent):
    """
    Non-Parametric Thompson Sampling (NPTS)
    a.k.a. Bounded Dirichlet Sampling (BDS)

    NPTS is a non-parametric Bayesian-inspired bandit algorithm that replaces
    classical parametric posterior sampling (e.g., Beta-Bernoulli) with a
    resampling scheme over empirical reward histories.

    Instead of assuming a parametric reward distribution, it constructs
    random convex combinations of past observations using Dirichlet weights.

    This produces a stochastic estimate of the arm mean without requiring
    explicit distributional assumptions.

    Attributes
    ----------
    bound : float
        Initial artificial reward used to stabilize early sampling.
    nbDraws : np.ndarray
        Number of pulls per arm.
    cumRewards : np.ndarray
        Cumulative reward per arm.
    meanRewards : np.ndarray
        Empirical mean reward per arm.
    rewardHistory : list[list]
        Full reward history per arm (including initialization with `bound`).
    playbuffer : list
        Temporary scheduling buffer used to enforce structured exploration.
    """

    def __init__(self, nbArms, bound=1.0,name="NPTS_old"):
        """
        Parameters
        ----------
        nbArms : int
            Number of arms in the bandit problem.
        bound : float, optional
            Initial reward seed for each arm (default is 1.0).
        """
        self.nA=nbArms
        self.bound = bound
        if name is None:
            BanditAgent.__init__(self, name="NPTS_old")
        else:
            BanditAgent.__init__(self, name=name)

    def reset(self):
        """
        Reset the internal state before a new experiment.
        """
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)
        self.meanRewards = np.zeros(self.nA, dtype=float)

        # Each arm starts with a bounded pseudo-observation
        self.rewardHistory = [[self.bound] for _ in range(self.nA)]

        # Buffer controlling exploration order
        self.playbuffer = list(range(self.nA))

    def select_arm(self):
        """
        Select the next arm to pull.

        Returns
        -------
        int
            Index of selected arm.

        Mechanism
        ---------
        - If the internal buffer is non-empty, pop the next arm.
        - Otherwise, recompute a candidate set of under-explored arms:
            * Identify leader arm (most sampled)
            * For less-sampled arms, compute Dirichlet resampled mean
            * Add arms that may compete with leader into buffer
        """
        if len(self.playbuffer) == 0:
            leader = randmax(self.nbDraws)
            muleader = self.meanRewards[leader]

            for a in range(self.nA):
                if (
                    self.nbDraws[a] < self.nbDraws[leader]
                    and self.nbDraws[a] > 0
                ):
                    tmua = self._dirichletmean(self.rewardHistory[a])

                    if max(self.meanRewards[a], tmua) >= muleader:
                        self.playbuffer.append(a)

        return self.playbuffer.pop()

    def _dirichletmean(self, rewards):
        """
        Compute a stochastic non-parametric estimate of the mean reward.

        This is done by sampling Dirichlet weights over past observations
        and computing a convex combination.

        Parameters
        ----------
        rewards : list[float]
            Historical rewards of an arm.

        Returns
        -------
        float
            Randomized estimate of the arm's mean.
        """
        w = np.random.dirichlet(np.ones(len(rewards)))
        return np.dot(w, rewards)

    def update(self, arm, reward):
        """
        Update internal statistics after observing a reward.

        Parameters
        ----------
        arm : int
            Index of selected arm.
        reward : float
            Observed reward.
        """
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1

        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)