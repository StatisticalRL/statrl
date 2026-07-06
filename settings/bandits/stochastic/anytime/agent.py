from abc import ABC, abstractmethod


class BanditAgent(ABC):

    def reset(self):
        """Initialize a new independent run."""

    def select_arm(self) -> int:
        """Return the arm to pull."""

    def update(self, arm: int, reward: float):
        """Update the learner after observing the reward."""
