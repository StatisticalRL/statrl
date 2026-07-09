from abc import ABC, abstractmethod

from gymnasium.utils import seeding

class BanditAgent(ABC):

    def __init__(self, name: str, seed: int = 1) -> None:
        self.name = name
        self.seed =seed


    def reset(self) -> None:
        """Initialize a new independent run."""
        self.np_random, self.seed = seeding.np_random(self.seed)

    @abstractmethod
    def select_arm(self) -> int:
        """Return the arm to pull."""
        ...

    def update(self, arm: int, reward: float) -> None:
        """Update the learner after observing the reward."""
