
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
        observation, info = super().reset(seed=seed, options=options)
        self.round = 0
        info["nextbatchsize"]=self.batchsize(self.round)
        return observation, info

    def step(self, action: list) -> tuple:  # type: ignore[override]
        """action = [4,3,2]"""
        B= self.batchsize(self.round)
        assert len(action)==B
        batchreward = []
        batchobservation=[]
        batchmean=[]
        info={}
        for aa in action:
            observation, reward, done, truncated, info = self.mab.step(aa)
            batchobservation.append(observation)
            batchreward.append(reward)
            batchmean.append(info["mean"])
        self.round=self.round+1
        info["nextbatchsize"]=self.batchsize(self.round)
        info["mean"]=sum(batchmean)
        return (batchobservation,batchreward,info)