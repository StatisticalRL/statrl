

import statrl.experiments.onerun as oR
import statrl.experiments.parallelruns as pR
import statrl.experiments.analyzeruns as aR
import statrl.experiments.plotruns as plR
from statrl.experiments.utils import clear_auxiliaryfiles

import time
import os
from typing import Any
ROOT="results/"


def runLargeMulticoreExperiment(env: Any, agents: list[Any], oracle: Any, interact: Any, timeHorizon: int=1000, nbReplicates: int=100, root_folder: str=ROOT) -> None:
    '''  Note: Runs single interaction of oracle with envs to compute oracle score ref.
    :param env:
    :param agents:
    :param oracle:
    :param timeHorizon:
    :param opttimeHorizon:
    :param nbReplicates:
    :param root_folder:
    :return:
    '''
    os.makedirs(root_folder, exist_ok=True)

    envName = env.name
    learners = agents

    print("-"*30+"Massive Multicore Experiment"+"-"*30)
    print(f'Environment: {envName}')
    print(f'Learners: {[learner.name for learner in learners]}')
    print(f'[INFO] Run {nbReplicates} many interactions of length {timeHorizon} for each learner:')
    dump_scores = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name)
        dump_scores_learner, meanelapsedtime_learner = pR.multicoreRuns(env, learner, interact, nbReplicates, timeHorizon, oR.oneRunWithDump, root_folder=root_folder)
        dump_scores.append(dump_scores_learner)
        meanelapsedtimes.append(meanelapsedtime_learner)

    dump_scoresopt, meanelapsedtime = pR.multicoreRuns(env, oracle, interact, nbReplicates, timeHorizon,
                                                    oR.oneRunWithDump, root_folder=root_folder)
    dump_scores.append(dump_scoresopt)

    ## Report statistics and compute regret:
    timestamp = str(time.time())
    logfilename = f"{root_folder}logfile_{envName}_{timestamp}.txt"
    with open(logfilename, 'w') as logfile:
        logfile.write("Environment " + envName + "\n")
        logfile.write("Optimal policy is: " + str(oracle.policy) + "\n")
        logfile.write("Learners " + str([learner.name for learner in learners]) + "\n")
        logfile.write("Time horizon is " + str(timeHorizon) + ", nb of replicates is " + str(nbReplicates) + "\n")
        for name, meanelapsedtime in zip(names, meanelapsedtimes):
            logfile.write(f"{name} average runtime is {meanelapsedtime}\n")
        print("[INFO] A log-file has been generated in ", logfilename)
        print("[INFO]  Compute Statistics...")
        mean, median, quantile1, quantile2, times = aR.computeScoreDiffs(names, dump_scores, timeHorizon, envName, root_folder=root_folder)
        print("[INFO]  Plot results...")
        plR.plotScoreDiffs(names, envName, envName, mean, median, quantile1, quantile2, times, timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder)
        print("[INFO]  Clean Auxiliary files...")
    clear_auxiliaryfiles(env, root_folder)
    print("[INFO]  Massive multicore experiment successfully completed.")
