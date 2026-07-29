from abc import ABC, abstractmethod
from gymnasium.utils import seeding


class BatchBanditAgent(ABC):
    def __init__(self, name="BanditAgent", seed=1):
        self.name = name
        self.seed =seed

    def reset(self) -> None:
        """Initialize a new independent run."""
        self.np_random, self.seed = seeding.np_random(self.seed)

    @abstractmethod
    def play(self):
        """Return one arm to pull."""

    @abstractmethod
    def update(self, action, reward)-> None:
        """Update the learner after observing the reward."""


    def batchplay(self,batchsize):
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)