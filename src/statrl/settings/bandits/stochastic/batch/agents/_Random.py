




from statrl.settings.bandits.stochastic.batch.agent import BatchBanditAgent
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

    def batchplay(self,batchsize):
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)


