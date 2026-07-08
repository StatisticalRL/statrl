from abc import ABC

from gymnasium.utils import seeding

class BanditAgent(ABC):

    def __init__(self,name: str,seed=1):
        self.name = name
        self.seed =seed


    def reset(self):
        """Initialize a new independent run."""
        self.np_random, self.seed = seeding.np_random(self.seed)

    def select_arm(self) -> int:
        """Return the arm to pull."""

    def update(self, arm: int, reward: float):
        """Update the learner after observing the reward."""
