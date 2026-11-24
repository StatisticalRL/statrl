from statrl.settings.bandits.stochastic.bestarmidentification.agent import (
    BAIBanditAgent,
)
import numpy as np

class Random(BAIBanditAgent):
    """Reference agent that always recommends the environment's best arm.

    This oracle is used only as the zero-error reference in ``test_massive``.
    It knows the environment's optimal arm and is not a learning algorithm.
    """

    def __init__(self, env):
        super().__init__(name="BAI-Oracle")
        self.env = env

    def select_arm(self):
        return np.random.randint(self.env.number_arms)

    def update(self, arm, reward):
        pass

    def recommend_arm(self):
        return np.random.randint(self.env.number_arms)

    def is_done(self):
        return False