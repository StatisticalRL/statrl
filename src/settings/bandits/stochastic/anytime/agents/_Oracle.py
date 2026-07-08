
from settings.bandits.stochastic.anytime.agent import BanditAgent


class Oracle(BanditAgent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        BanditAgent.__init__(self, name="Oracle")

    @property
    def policy(self):
        return [self.env.optimal_arm]

    def reset(self):
        ()

    def select_arm(self):
        return self.env.optimal_arm

    def update(self, action, reward):
       ()