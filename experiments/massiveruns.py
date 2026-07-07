

import statrl.experiments.onerun as oR
import statrl.experiments.parallelruns as pR
import statrl.experiments.analyzeruns as aR
import statrl.experiments.plotruns as plR
from statrl.experiments.utils import clear_auxiliaryfiles

import time
import os
ROOT="results/"


def runLargeMulticoreExperiment(env, agents, oracle, interact, timeHorizon=1000,  nbReplicates=100, root_folder=ROOT):
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
    os.mkdir(root_folder)

    envFullName= env.name

    #opti_learner=opt.build_opti(envFullName, env.env, env.observation_space.n, env.action_space.n)
    learners = [x[0](**x[1]) for x in agents]

    print("*********************************************")
    dump_scores = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name)
        learner_scores, meanelapsedtime = pR.multicoreRuns(envFullName, learner, interact, nbReplicates, timeHorizon,oR.oneRunWithDump, root_folder=root_folder)
        dump_scores.append(learner_scores)
        meanelapsedtimes.append(meanelapsedtime)

    #dump_scoresopt = oR.oneRunWithDump(env, oracle, interact, timeHorizon, root_folder=root_folder)
    dump_scoresopt, meanelapsedtime = pR.multicoreRuns(envFullName, oracle, interact, nbReplicates, timeHorizon,
                                                    oR.oneRunWithDump, root_folder=root_folder)

    dump_scores.append(dump_scoresopt)

    ## Report statistics and compute regret:
    #print('************** ANALYSIS **************')
    timestamp = str(time.time())
    logfilename=root_folder+"logfile_"+env.name+"_"+timestamp+".txt"
    logfile = open(logfilename,'w')
    logfile.write("Environment "+env.name +"\n")
    logfile.write("Optimal policy is: " + str(oracle.policy)+"\n")
    logfile.write("Learners "+str([learner.name for learner in learners]) +"\n")
    logfile.write("Time horizon is "+ str(timeHorizon) + ", nb of replicates is "+ str(nbReplicates) +"\n")
    [logfile.write(str(names[i])+ " average runtime is "+ str(meanelapsedtimes[i])  +"\n") for i in range(len(names))]
    mean,median, quantile1,quantile2,times = aR.computeScoreDiffs(names, dump_scores, timeHorizon, envFullName, root_folder=root_folder)
    title = f"{env.name}"
    plR.plotScoreDiffs(names, envFullName, title, mean, median, quantile1, quantile2, times, timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder)
    #print("*********************************************")
    clear_auxiliaryfiles(env, root_folder)
    print("\n[INFO] A log-file has been generated in ",logfilename)