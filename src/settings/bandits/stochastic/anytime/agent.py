from abc import ABC, abstractmethod

from gymnasium.utils import seeding

class BanditAgent(ABC):

    def __init__(self,name: str):
        self.name = name


    def reset(self,seed):
        """Initialize a new independent run."""
        self.np_random, self.seed = seeding.np_random(seed)

    def play(self) -> int:
        """Return the arm to pull."""

    def update(self, arm: int, reward: float):
        """Update the learner after observing the reward."""
