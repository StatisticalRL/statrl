from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.utils import randmin,randmax, KLinf_threshold

import numpy as np
from math import log

"""
IMED variants — non-parametric version using KLinf_threshold
============================================================
All three classes use the empirical KL-divergence  Kinf(hat_F_a, mu*)
(computed via KLinf_threshold) instead of a parametric kl(mu_a, mu*).

This makes the algorithms distribution-free for any bounded reward
distribution with known upper bound `bound`.

Index formula (all variants):
    I_a = N_a * Kinf(hat_F_a, mu*) + log(N_a)

with the difference lying in *when* N_a and hat_F_a are updated:

  IMED        (3b adapted)  — N_a incremented & index refreshed at each
                              step inside the batch; hat_F_a updated at
                              end of batch.  The index only changes when
                              the *selected arm* changes (only N_a of the
                              pulled arm changes), so we skip redundant
                              recomputation of other arms.
  IMEDnaive   (3a naive)    — N_a *not* incremented during the batch; all
                              arms are drawn from the same index computed
                              before the batch starts → always returns the
                              same arm, so batchplay is O(1).
"""


class BatchIMED(BatchBanditAgent):
    """B-IMED adapté (3b): within-batch sequential index updates.

    During batchplay, N_a is incremented after each draw and only the index
    of the pulled arm is recomputed (other arms are unchanged).  This reduces
    batchplay from O(B × K × KLinf) to O(B × 1 × KLinf).
    Reward histories are updated only at the end via batchupdate.
    """

    def __init__(self, nbArms, bound=1.0, batchagnostic=False, **kwargs):
        self.nbArms = nbArms
        self.bound = bound
        self.batchagnostic = batchagnostic
        name= "ABatchIMED" if self.batchagnostic else "BatchIMED"
        BatchBanditAgent.__init__(self, name=name)

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms) # total number of draws of each arm
        self.nbDraws_start = np.zeros(self.nbArms) # number of draws of each arm at the start of an episode
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.maxMeans = 0.0
        self.indexes = np.zeros(self.nbArms)
        self.kinfs = np.zeros(self.nbArms)
        self.rewardHistory = [[] for _ in range(self.nbArms)]

    def play(self):
        a1 = randmax(self.meanRewards)
        a0 = randmin(np.array([self.indexes[a] for a in range(self.nbArms) if (self.meanRewards[a]==self.meanRewards[a1]) or (self.nbDraws[a]<=np.floor(self.x_threshold*self.nbDraws_start[a])+1)]))
        return a0

    def _update_index(self):
        for a in range(self.nbArms):
            self._update_index_arm(a)

    def _update_index_arm(self, a):
        """Recompute the index for a single arm (used in batchplay)."""
        n = self.nbDraws[a]
        nn = self.nbDraws_start[a]
        self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))

    # ------------------------------------------------------------------
    # Online interface
    # ------------------------------------------------------------------
    def update(self, arm, reward):
        self.rewardHistory[arm].append(reward)
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.maxMeans = float(np.max(self.meanRewards))
        self._update_index()

    # ------------------------------------------------------------------
    # Batch interface (adapted)
    # ------------------------------------------------------------------
    def batchplay(self, batchsize):
        """Increment N_a and refresh only the pulled arm's index each step.

        Since rewardHistory (and therefore hat_F_a) does not change during
        batchplay, only N_a of the selected arm changes → only that arm's
        index needs recomputing.  This reduces cost from O(B×K×KLinf) to
        O(B×KLinf).
        """
        batcharms = []
        t = sum(self.nbDraws)
        if self.batchagnostic:
            self.x_threshold = np.log(t + 1) / np.log(t) if (t > 1) else 1.
        else:
            self.x_threshold = np.log(t + batchsize) / np.log(t) if (t > 1) else 1.
        self._update_index()
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a] += 1
            if self.batchagnostic:
                self.x_threshold = np.log(t + b+2) / np.log(t) if (t > 1) else 1.
            self._update_index_arm(a)
        return batcharms

    def batchupdate(self, batcharm, batchreward):
        """Receive rewards; update histories and means at end of batch."""
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.rewardHistory[a].extend(rewards_a.tolist())
                self.cumRewards[a] += rewards_a.sum()
                # nbDraws already incremented during batchplay
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]
                self.nbDraws_start[a] = self.nbDraws[a]
        self.maxMeans = float(np.max(self.meanRewards))
        for a in range(self.nbArms):
            if (self.meanRewards[a] < self.maxMeans) and (len(self.rewardHistory[a]) > 0):
                self.kinfs[a] = KLinf_threshold(self.rewardHistory[a], self.maxMeans,
                                upper_bound=self.bound)
            else:
                self.kinfs[a]=0


class BatchIMED2(BatchBanditAgent):
    """B-IMED adapté (3b): within-batch sequential index updates.

    During batchplay, N_a is incremented after each draw and only the index
    of the pulled arm is recomputed (other arms are unchanged).  This reduces
    batchplay from O(B × K × KLinf) to O(B × 1 × KLinf).
    Reward histories are updated only at the end via batchupdate.
    """

    def __init__(self, nbArms, bound=1.0, batchagnostic=False, **kwargs):
        self.nbArms = nbArms
        self.bound = bound
        self.batchagnostic = batchagnostic
        name= "B-IMED (2)" if self.batchagnostic else "B-IMED (1)"
        BatchBanditAgent.__init__(self, self.nbArms, name=name)

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms) # total number of draws of each arm
        self.nbDraws_start = np.zeros(self.nbArms) # number of draws of each arm at the start of an episode
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.maxMeans = 0.0
        self.indexes = np.zeros(self.nbArms)
        self.kinfs = np.zeros(self.nbArms)
        self.rewardHistory = [[] for _ in range(self.nbArms)]

    def play(self):
        a1 = randmax(self.meanRewards)
        # THE FOLLOWING RULE fails:
        #a = randmin([self.indexes[a] for a in range(self.nbArms) if (self.meanRewards[a]==self.meanRewards[a1]) or (self.nbDraws[a]<=np.floor(self.x_threshold*self.nbDraws_start[a])+1)])
        #return a

        a0 =  randmin(self.indexes)
        if (self.nbDraws[a0]* self.kinfs[a0] <= self.x_threshold*self.nbDraws_start[a0]* self.kinfs[a0]):
        #if (self.meanRewards[a0]==self.meanRewards[a1]) or (self.nbDraws[a0]<=np.floor(self.x_threshold*self.nbDraws_start[a0])+1):
            return a0
        #a2 = randmin([self.indexes[a] for a in range(self.nbArms) if self.indexes[a]<=self.indexes[a1]])
        #if (self.meanRewards[a2] == self.meanRewards[a1]) or (self.nbDraws[a2] <= np.floor(self.x_threshold * self.nbDraws_start[a2]) + 1):
        #    return a2
        return a1

    def play_alter(self):
       # Alternative version:
       a1 = randmax(self.meanRewards)

       a0 = randmin([self.nbDraws[a] for a in range(self.nbArms) if self.indexes[a]<=self.indexes[a1]])
       if (self.nbDraws[a0] * self.kinfs[a0] <= self.x_threshold * self.nbDraws_start[a0] * self.kinfs[a0]):
           return a0
       return a1


    def _update_index(self):
        for a in range(self.nbArms):
            self._update_index_arm(a)

    def _update_index_arm(self, a):
        """Recompute the index for a single arm (used in batchplay)."""
        n = self.nbDraws[a]
        nn = self.nbDraws_start[a]
        self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))
       # Alternative version:
       # if self.meanRewards[a]==max(self.meanRewards):
       #     self.indexes[a] = log(sum(self.nbDraws_start)+ self.batchsize)
       # else:
       #     self.indexes[a] = n * self.kinfs[a] + self.x_threshold * log(max(nn, 1))

    # ------------------------------------------------------------------
    # Online interface
    # ------------------------------------------------------------------
    def update(self, arm, reward):
        self.rewardHistory[arm].append(reward)
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.maxMeans = float(np.max(self.meanRewards))
        self._update_index()

    # ------------------------------------------------------------------
    # Batch interface (adapted)
    # ------------------------------------------------------------------
    def batchplay(self, batchsize):
        """Increment N_a and refresh only the pulled arm's index each step.

        Since rewardHistory (and therefore hat_F_a) does not change during
        batchplay, only N_a of the selected arm changes → only that arm's
        index needs recomputing.  This reduces cost from O(B×K×KLinf) to
        O(B×KLinf).
        """
        batcharms = []
        self.batchsize=batchsize
        t = sum(self.nbDraws)
        if self.batchagnostic:
            self.x_threshold =np.log(t + 1) / np.log(t) if (t > 1) else 1.
        else:
            self.x_threshold =  np.log(t + batchsize) / np.log(t) if (t > 1) else 1.
        self._update_index()
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a] += 1
            if  self.batchagnostic:
                self.x_threshold = np.log(t + b+2) / np.log(t) if (t > 1) else 1.
            self._update_index_arm(a)
        return batcharms

    def batchupdate(self, batcharm, batchreward):
        """Receive rewards; update histories and means at end of batch."""
        arm_arr = np.asarray(batcharm)
        rew_arr = np.asarray(batchreward)
        for a in range(self.nbArms):
            mask = arm_arr == a
            if mask.any():
                rewards_a = rew_arr[mask]
                self.rewardHistory[a].extend(rewards_a.tolist())
                self.cumRewards[a] += rewards_a.sum()
                # nbDraws already incremented during batchplay
                self.meanRewards[a] = self.cumRewards[a] / self.nbDraws[a]
                self.nbDraws_start[a] = self.nbDraws[a]
        self.maxMeans = float(np.max(self.meanRewards))
        for a in range(self.nbArms):
            if (self.meanRewards[a] < self.maxMeans) and (len(self.rewardHistory[a]) > 0):
                self.kinfs[a] = KLinf_threshold(self.rewardHistory[a], self.maxMeans,
                                upper_bound=self.bound)
            else:
                self.kinfs[a]=0
        #self._update_index()