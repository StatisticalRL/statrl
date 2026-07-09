
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv


class Oracle(BanditAgent):
    """Oracle"""
    def __init__(self, env: StochasticBanditEnv) -> None:
        self.env=env
        BanditAgent.__init__(self, name="Oracle")

    @property
    def policy(self) -> list[int]:
        return [self.env.optimal_arm]

    def reset(self) -> None:
        pass

    def select_arm(self) -> int:
        return self.env.optimal_arm

    def update(self, action: int, reward: float) -> None:
        pass