
from settings.bandits.stochastic.anytime.agent import BanditAgent
from settings.bandits.stochastic.anytime.environment import StochasticBanditEnv

import numpy as np
class Random(BanditAgent):
    """Uniform Exploration"""

    def __init__(self, env: StochasticBanditEnv) -> None:
        self.env= env
        BanditAgent.__init__(self, name="Random")

    def reset(self) -> None:
        """Initialize a new independent run."""

    def select_arm(self) -> int:
        """Return the arm to pull."""
        return np.random.randint(self.env.number_arms)

    def update(self, arm: int, reward: float) -> None:
        """Update the learner after observing the reward."""