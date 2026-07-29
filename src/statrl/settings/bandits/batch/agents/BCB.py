from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.utils import randmax

import numpy as np

"""
BCB — Bounded CVaR Bandit (Gautron et al., 2024)
=================================================
We implement the parameter choice CVaR = Expectation (alpha -> 1), which
corresponds to Non-Parametric Thompson Sampling with a Dirichlet prior
anchored at the upper bound B of the reward support.

References
----------
Gautron et al. (2024) "Bandits with Bounded CVaR Constraints".
"""


class BCB(BatchBanditAgent):
    """BCB with CVaR = Expectation (adapted batch version).

    In each batch, the arm counts are updated *sequentially* inside the batch
    (optimistic within-batch exploration), but reward histories (used for the
    Dirichlet draw) are only updated at the end of the batch via batchupdate.

    Parameters
    ----------
    nbArms : int
    bound : float
        Upper bound B of the reward support.  The Dirichlet prior is
        initialised with a single pseudo-observation at B.
    """

    def __init__(self, nbArms, bound=1.0):
        self.nbArms = nbArms
        self.bound = bound
        BatchBanditAgent.__init__(self, name="BCB-adapted")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        # Each arm starts with one pseudo-observation at the upper bound B
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]

    # ------------------------------------------------------------------
    # Core Dirichlet sampling
    # ------------------------------------------------------------------
    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return float(np.dot(w, rewards))

    def play(self):
        return randmax([self._dirichletmean(self.rewardHistory[a])
                        for a in range(self.nbArms)])

    # ------------------------------------------------------------------
    # Online (non-batch) interface
    # ------------------------------------------------------------------
    def update(self, arm, reward):
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)

    # ------------------------------------------------------------------
    # Batch interface (adapted: counts updated during play)
    # ------------------------------------------------------------------
    def batchplay(self, batchsize):
        # Scores are computed from rewardHistory which does not change during
        # batchplay (only updated in batchupdate), so the same arm is always
        # selected. Compute once and fill — no loop needed.
        scores = np.array([self._dirichletmean(self.rewardHistory[a])
                  for a in range(self.nbArms)])
        a = randmax(scores)
        self.nbDraws[a] += batchsize   # optimistic count increment
        return [a] * batchsize

    def batchupdate(self, batcharm, batchreward):
        """Receive rewards and update histories at end of batch."""
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.cumRewards[a] += rewards_a.sum()
                self.rewardHistory[a].extend(rewards_a.tolist())
        # Recompute means from cumRewards and actual nbDraws
        # (nbDraws was pre-incremented in batchplay, stays consistent)
        for a in range(self.nbArms):
            if self.nbDraws[a] > 0:
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]


class BCBnaif(BatchBanditAgent):
    """BCB with CVaR = Expectation (naive batch version).

    All batchsize draws are made from the *same* Dirichlet distribution
    (no within-batch count updates).  Equivalent to drawing batchsize i.i.d.
    samples from the current policy and then updating at the end.
    """

    def __init__(self, nbArms, bound=1.0):
        self.nbArms = nbArms
        self.bound = bound
        BatchBanditAgent.__init__(self, name="BCB")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return float(np.dot(w, rewards))

    def play(self):
        return randmax([self._dirichletmean(self.rewardHistory[a])
                        for a in range(self.nbArms)])

    def update(self, arm, reward):
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)

    def batchplay(self, batchsize):
        scores = [self._dirichletmean(self.rewardHistory[a])
                  for a in range(self.nbArms)]
        a = randmax(scores)
        return [a] * batchsize

    def batchupdate(self, batcharm, batchreward):
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.cumRewards[a] += rewards_a.sum()
                self.nbDraws[a] += mask.sum()
                self.rewardHistory[a].extend(rewards_a.tolist())
        for a in range(self.nbArms):
            if self.nbDraws[a] > 0:
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]