
from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv

class Random(KernelBanditAgent):
    """Uniform exploration."""

    def __init__(self, env: KernelBanditEnv) -> None:
        self.env = env
        KernelBanditAgent.__init__(
            self,
            arms=env.arm_features,
            kernel=env.kernel,
            noise_std=env.noise_std,
            name="Random",
        )

    def reset(self) -> None:
        """Initialize a new independent run."""
        super().reset()

    def select_arm(self) -> int:
        """Return an arm uniformly at random."""
        return int(self.np_random.integers(self.env.number_arms))

    def update(self, arm: int, reward: float) -> None:
        """Update the learner after observing the reward."""
        pass