
from typing import Any, Optional

from settings.bandits.stochastic.anytime.environment import StochasticBanditEnv as MAB

class BatchMAB(MAB):
    def __init__(self, mab: MAB, batchsize: Any) -> None:
        self.mab = mab
        # Accept a plain list (picklable for multiprocessing) or a callable.
        if callable(batchsize):
            self.batchsize = batchsize
        else:
            _sizes = list(batchsize)
            self.batchsize = lambda ell: _sizes[ell] if ell < len(_sizes) else 1
        self.name = "Batch"+self.mab.name+"-batch"+str(self.batchsize(0))
        self.round = 0
        super(BatchMAB, self).__init__(self.mab.rewarddistributions, name=self.name)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:  # type: ignore[override]
        observation = super().reset(seed=seed, options=options)  # MAB.reset returns the dummy observation
        self.round = 0
        info = {"nextbatchsize": self.batchsize(self.round)}
        return observation, info

    def step(self, action: list) -> tuple:  # type: ignore[override]
        """action = [4,3,2]"""
        B= self.batchsize(self.round)
        assert len(action)==B
        batchreward = []
        batchobservation=[]
        batchmean=[]
        for aa in action:
            reward = self.mab.step(aa)                       # MAB.step returns the reward only
            batchobservation.append(0)                       # bandit is stateless: constant dummy observation
            batchreward.append(reward)
            batchmean.append(self.mab.expected_reward(aa))   # arm mean, for regret accounting
        self.round=self.round+1
        info = {"nextbatchsize": self.batchsize(self.round), "mean": sum(batchmean)}
        return (batchobservation,batchreward,info)