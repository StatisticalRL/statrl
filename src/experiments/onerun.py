
from experiments.utils import dump

import time

from abc import ABC, abstractmethod
import numpy as np

class Interaction(ABC):

    @abstractmethod
    def run(self, env, learner, horizon)-> np.ndarray:
        return np.array([])


    @abstractmethod
    def renderrun(self, env, learner, horizon):
        ()




def oneRunWithDump(env,learner,interact, timeHorizon,root_folder):
    scoretimeseries=interact.run(env,learner,timeHorizon)
    assert(len(scoretimeseries)==timeHorizon)

    tag = env.name + "_scores_" + learner.name + "_" + str(timeHorizon) +"_" + str(time.time())
    filename = dump(scoretimeseries,"aux",tag,root_folder)
    return filename