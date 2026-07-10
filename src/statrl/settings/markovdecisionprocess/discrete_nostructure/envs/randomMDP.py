

from statrl.settings.markovdecisionprocess.discrete_nostructure.environment import DiscreteMDP
from statrl.settings.utils import  Dirac
import scipy.stats as stat
import numpy as np


class RandomMDP(DiscreteMDP):
    def __init__(self, nbStates, nbActions, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1,
                 maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5,ergodic=0.00,
                 seed=None,name="RandomMDP"):
        self.nS = nbStates
        self.nA = nbActions
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)

        self.seed(seed)

        self.startdistribution = np.zeros((self.nS))
        self.rewards = {}
        self.transitions = {}
        self.P = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}
            self.rewards[s] = {}
            for a in self.actions:
                self.P[s][a] = []
                self.transitions[s][a] = {}
                my_mean = self.sparserand(p=maxProportionSupportReward, min=minNonZeroReward,max=0.99)
                if False and (rewardStd > 0 and my_mean > 0 and my_mean < 1):
                    ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                    self.rewards[s][a] = stat.truncnorm(ma, mb, loc=my_mean, scale=rewardStd)
                else:
                    self.rewards[s][a] = Dirac(my_mean)
                transitionssa = np.zeros((self.nS))
                for s2 in self.states:
                    transitionssa[s2] =  max(self.sparserand(p=maxProportionSupportTransition, min=minNonZeroProbability), ergodic)
                mass = sum(transitionssa)
                if (mass > 0):
                    transitionssa = transitionssa / sum(transitionssa)
                    transitionssa = self.reshapeDistribution(transitionssa, minNonZeroProbability)
                else:
                    transitionssa[self.np_random.integers(self.nS)] = 1.
                li = self.P[s][a]
                [li.append((transitionssa[s], s, False)) for s in self.states if transitionssa[s] > 0]
                self.transitions[s][a] = {ss: transitionssa[ss] for ss in self.states}

            self.startdistribution[s] = self.sparserand(p=maxProportionSupportStart, min=minNonZeroProbability)
        mass = sum(self.startdistribution)
        if (mass > 0):
            self.startdistribution = self.startdistribution / sum(self.startdistribution)
            self.startdistribution = self.reshapeDistribution(self.startdistribution, minNonZeroProbability)
        else:
            self.startdistribution[self.np_random.integers(self.nS)] = 1.

        checkRewards = sum([sum([self.rewards[s][a].mean() for a in self.actions]) for s in self.states])
        if (checkRewards == 0):
            s = self.np_random.integers(0, self.nS)
            a = self.np_random.integers(0, self.nA)
            my_mean = min(minNonZeroReward + self.np_random.random() * (1. - minNonZeroReward), 0.99)
            if (rewardStd > 0 and my_mean > 0 and my_mean < 1):
                ma, mb = (0 - my_mean) / rewardStd, (1 - my_mean) / rewardStd
                self.rewards[s][a] = stat.truncnorm(ma, mb, loc=my_mean, scale=rewardStd)
            else:
                self.rewards[s][a] = Dirac(my_mean)
        # print("Random MDP is generated")
        # print("initial:",self.startdistribution)
        # print("rewards:",self.rewards)
        # print("transitions:",self.P)

        # Now that the Random MDP is generated with a given seed, we finalize its generation with an empty seed (seed=None) so that transitions/rewards are indeed stochastic:
        super(RandomMDP, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution, seed=None,name=name)

    def sparserand(self, p=0.5, min=0., max=1.):
        u = self.np_random.random()
        if (u <= p):
            return min + self.np_random.random() * (max - min)
        return 0.

    def reshapeDistribution(self, distribution, p):
        mdistribution = [0 if x < p else x for x in distribution]
        mass = sum(mdistribution)
        while (mass < 0.99999999):
            i = self.np_random.integers(0, len(distribution))
            if (mdistribution[i] < p):
                newp = min(p, 1. - mass)
                if (newp == p):
                    mass = mass - mdistribution[i] + p
                    mdistribution[i] = p
            if (mdistribution[i] >= p):
                newp = min(1., mdistribution[i] + 1. - mass)
                mass = mass - mdistribution[i] + newp
                mdistribution[i] = newp
        mass = sum(mdistribution)
        mdistribution = [x / mass for x in mdistribution]
        return mdistribution

