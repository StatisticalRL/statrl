


from statrl.settings.bandits.stochastic.batch.agent import BatchBanditAgent

class Oracle(BatchBanditAgent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        BatchBanditAgent.__init__(self, name="Oracle")


    @property
    def policy(self) -> list[int]:
        return [self.env.optimal_arm]


    def reset(self) -> None:
        pass

    def play(self) -> int:
        return self.env.optimal_arm

    def update(self, action: int, reward: float) -> None:
        pass
    # def reset(self):
    #     ()

    #
    # def update(self, action, reward):
    #    ()
    #
    def batchplay(self,batchsize):
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)
