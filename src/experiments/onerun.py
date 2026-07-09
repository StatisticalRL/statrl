
from experiments.utils import dump

import time

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Interaction(ABC):

    @abstractmethod
    def run(self, env: Any, learner: Any, horizon: int)-> np.ndarray:
        return np.array([])


    @abstractmethod
    def renderrun(self, env: Any, learner: Any, horizon: int) -> None:
        ...




def oneRunWithDump(env: Any, learner: Any, interact: Any, timeHorizon: int, root_folder: str) -> str:
    scoretimeseries=interact.run(env,learner,timeHorizon)
    assert len(scoretimeseries) == timeHorizon

    tag = f"{env.name}_scores_{learner.name}_{timeHorizon}_{time.time()}"
    filename = dump(scoretimeseries,"aux",tag,root_folder)
    return filename