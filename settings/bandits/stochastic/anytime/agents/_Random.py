
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent

class Random(BanditAgent):
    """Uniform Exploration"""

    def __init__(self,env):
        self.env= env
        BanditAgent.__init__(self, name="Random")

    def reset(self):
        """Initialize a new independent run."""

    def select_arm(self) -> int:
        """Return the arm to pull."""
        return self.env.action_space.sample()

    def update(self, arm: int, reward: float):
        """Update the learner after observing the reward."""