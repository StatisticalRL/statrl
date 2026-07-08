
from abc import ABC
import numpy as np

from gymnasium import Env, spaces
from gymnasium.utils import seeding

class StochasticBanditEnv(Env):
    """
    Abstract stochastic multi-armed bandit environment.

    Each arm corresponds to an independent reward distribution.
    Pulling an arm produces an independent sample from its
    associated distribution.

    The environment is stateless: interactions do not modify
    the underlying distributions.
    """

    def __init__(self,rewarddistributions,name):
        self.rewarddistributions = rewarddistributions
        self.name = name
        self.renderers = []

    @property
    def number_arms(self) -> int:
        """Number of available arms."""
        return len(self.rewarddistributions)

    @property
    def means(self):
        """
        Mean reward of every arm.

        Intended only for evaluation and oracle construction.
        """
        return [arm.mean for arm in self.rewarddistributions]

    @property
    def optimal_mean(self):
        return max(self.means)

    @property
    def optimal_arm(self):
        return int(np.argmax(self.means))

    def step(self, arm: int)-> float:
        """
        Sample one reward from the specified arm.
        """

        """
        :param a: action
        :return:  (state, reward, IsDone?, IsTruncated?, meanreward)
        The meanreward is returned for information, it should not be given to the learner.
        """
        r = self.rewarddistributions[arm].sample()
        self.last=(arm,r)
        return r

    def expected_reward(self, arm):
        return self.means[arm]

    def reset(self, seed=None, options=None):
        """
              Reset the random generator.
              """
        #super().reset(seed=seed, options=options)
        self.np_random, self.seed = seeding.np_random(seed)
        self.last = (None,0)
        return 0

    def render(self, mode='human'):
        for re in self.renderers:
            re.render(self.last)
