




from statrl.settings.bandits.batch.agent import BatchBanditAgent
import numpy as np
class Random(BatchBanditAgent):
    """Uniform Exploration"""

    def __init__(self, env) -> None:
        self.env= env
        BatchBanditAgent.__init__(self, name="Random")

    def reset(self) -> None:
        """Initialize a new independent run."""
        pass

    def play(self) -> int:
        """Return the arm to pull."""
        return np.random.randint(self.env.mab.number_arms)

    def update(self, arm: int, reward: float) -> None:
        """Update"""
        pass


