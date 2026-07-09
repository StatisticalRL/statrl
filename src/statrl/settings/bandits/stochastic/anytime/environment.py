
from typing import Optional

import numpy as np

from gymnasium import Env
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

    def __init__(self, rewarddistributions: list, name: str, last: tuple[Optional[int], float] = (None, 0.0)) -> None:
        self.rewarddistributions = rewarddistributions
        self.name = name
        self.renderers: list = []
        self.last = last

    @property
    def number_arms(self) -> int:
        """Number of available arms."""
        return len(self.rewarddistributions)

    @property
    def means(self) -> list[float]:
        """
        Mean reward of every arm.

        Intended only for evaluation and oracle construction.
        """
        return [arm.mean for arm in self.rewarddistributions]

    @property
    def optimal_mean(self) -> float:
        return max(self.means)

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    def step(self, arm: int) -> float:  # type: ignore[override]  # bandit API: reward only, not gym's 5-tuple
        """
        Sample one reward from the specified arm.

        Returns the sampled reward. (The arm's mean is available via
        `expected_reward`; it must not be given to the learner.)
        """
        r = self.rewarddistributions[arm].sample()
        self.last=(arm,r)
        return r

    def expected_reward(self, arm: int) -> float:
        return self.means[arm]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> int:  # type: ignore[override]  # bandit API: no observation tuple
        """
              Reset the random generator.
              """
        #super().reset(seed=seed, options=options)
        self.np_random, self.seed = seeding.np_random(seed)
        self.last = (None,0)
        return 0

    def render(self, mode: str = 'human') -> None:
        for re in self.renderers:
            re.render(self,self.last)

    def close(self) -> None:
        for re in self.renderers:
            re.stop(self)
