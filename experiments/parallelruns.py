import gymnasium
import time
import copy
from joblib import Parallel, delayed

import multiprocessing


## Parallelization
def multicoreRuns(envRegisterName, learner, interact, nbReplicates, timeHorizon, oneRunFunction, root_folder):
    num_cores = multiprocessing.cpu_count()
    envs = []
    learners = []
    interacts= []
    timeHorizons = []
    rootFolders= []

    for i in range(nbReplicates):
        envs.append(gymnasium.make(envRegisterName).unwrapped)
        learners.append(copy.deepcopy(learner))
        interacts.append(copy.deepcopy(interact))
        timeHorizons.append(copy.deepcopy(timeHorizon))
        rootFolders.append(root_folder)

    t0 = time.time()

    scores = Parallel(n_jobs=num_cores)(delayed(oneRunFunction)(*i) for i in zip(envs,learners,interacts, timeHorizons, rootFolders))

    elapsed = time.time()-t0
    return scores, elapsed / nbReplicates