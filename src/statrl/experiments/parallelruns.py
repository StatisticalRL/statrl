import time
import copy
from joblib import Parallel, delayed
from typing import Any, Callable


## Parallelization
def multicoreRuns(env: Any, learner: Any, interact: Any, nbReplicates: int, timeHorizon: int, oneRunFunction: Callable[..., Any], root_folder: str) -> tuple[Any, float]:
    #FIXME  Should be made more general? indep of gymnasium?
        #envs.append(gymnasium.make(envRegisterName).unwrapped)
    jobs = [
        (copy.deepcopy(env), copy.deepcopy(learner), copy.deepcopy(interact), timeHorizon, root_folder)
        for _ in range(nbReplicates)
    ]

    t0 = time.time()
    scores = Parallel(n_jobs=-1)(delayed(oneRunFunction)(*job) for job in jobs)
    elapsed = time.time() - t0

    return scores, elapsed / nbReplicates