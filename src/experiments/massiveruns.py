

import experiments.onerun as oR
import experiments.parallelruns as pR
import experiments.analyzeruns as aR
import experiments.plotruns as plR
from experiments.utils import clear_auxiliaryfiles

import time
import os
ROOT="results/"
from time import sleep


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
    # The following harness is used so that in case the root_folder already exists, the os does not through an error.
    # So we try to make a new repo, but if it already exists, we simply do nothing.
    try:
        os.mkdir(root_folder)
    except:
        ()

    environment=env[0](**env[1])
    envName= environment.name

    #opti_learner=opt.build_opti(envFullName, env.env, env.observation_space.n, env.action_space.n)
    learners = [x[0](**x[1]) for x in agents]

    print("*********************************************")
    dump_scores = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name)
        dump_scores_learner, meanelapsedtime_learner = pR.multicoreRuns(environment, learner, interact, nbReplicates, timeHorizon,oR.oneRunWithDump, root_folder=root_folder)
        dump_scores.append(dump_scores_learner)
        meanelapsedtimes.append(meanelapsedtime_learner)

    #dump_scoresopt = oR.oneRunWithDump(env, oracle, interact, timeHorizon, root_folder=root_folder)
    dump_scoresopt, meanelapsedtime = pR.multicoreRuns(environment, oracle, interact, nbReplicates, timeHorizon,
                                                    oR.oneRunWithDump, root_folder=root_folder)
    dump_scores.append(dump_scoresopt)

    #sleep(5)

    ## Report statistics and compute regret:
    #print('************** ANALYSIS **************')
    timestamp = str(time.time())
    logfilename=root_folder+"logfile_"+environment.name+"_"+timestamp+".txt"
    logfile = open(logfilename,'w')
    logfile.write("Environment "+environment.name +"\n")
    logfile.write("Optimal policy is: " + str(oracle.policy)+"\n")
    logfile.write("Learners "+str([learner.name for learner in learners]) +"\n")
    logfile.write("Time horizon is "+ str(timeHorizon) + ", nb of replicates is "+ str(nbReplicates) +"\n")
    [logfile.write(str(names[i])+ " average runtime is "+ str(meanelapsedtimes[i])  +"\n") for i in range(len(names))]
    mean,median, quantile1,quantile2,times = aR.computeScoreDiffs(names, dump_scores, timeHorizon, envName, root_folder=root_folder)
    title = f"{envName}"
    plR.plotScoreDiffs(names, envName, title, mean, median, quantile1, quantile2, times, timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder)
    #print("*********************************************")
    clear_auxiliaryfiles(environment, root_folder)
    print("\n[INFO] A log-file has been generated in ",logfilename)