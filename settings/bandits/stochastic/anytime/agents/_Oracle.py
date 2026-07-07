
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent


class Oracle(BanditAgent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        BanditAgent.__init__(self, name="Oracle")

    def reset(self):
        ()

    def select_arm(self):
        return self.env.optimal_arm

    def update(self, action, reward):
       ()