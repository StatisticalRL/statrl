
from statrl.experiments.utils import dump

import time

def oneRunWithDump(env,learner,interact, timeHorizon,root_folder):
    scoretimeseries=interact(env,learner,timeHorizon)

    tag = env.name + "_scores_" + learner.name() + "_" + str(timeHorizon) +"_" + str(time.time())
    filename = dump(scoretimeseries,"aux",tag,root_folder)
    return filename