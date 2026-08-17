

from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class Oracle(KernelBanditAgent):
    """Oracle knowing the true optimal arm of the kernel bandit."""

    def __init__(self, env: KernelBanditEnv) -> None:
        self.env = env
        KernelBanditAgent.__init__(
            self,
            name="Oracle",
            arms=env.arm_features,
            kernel=env.kernel,
            noise_std=env.noise_std,
        )

    @property
    def policy(self) -> list[int]:
        return [self.env.optimal_arm]

    def reset(self) -> None:
        pass

    def select_arm(self) -> int:
        return self.env.optimal_arm

    def update(self, action: int, reward: float) -> None:
        pass