

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
    """Benchmark several agents on one environment and plot their regret.

    For each agent it runs ``nbReplicates`` independent interactions in parallel, runs the oracle for
    the same number, computes regret as the oracle's cumulative score minus each agent's, and writes 
    a logfile and regret figures under ``root_folder``. 

    Parameters
    ----------
    env : object
        Environment to benchmark on. Must expose ``name``; an optional
        ``displayname`` is used as the figure title when present.
    agents : list of object
        Agents to compare. Their ``name`` attributes must be distinct — dump
        filenames and plot legends are keyed on them, so duplicates silently
        merge two agents' results.
    oracle : object
        Reference agent defining zero regret, and the only one required to
        expose a ``policy`` (it is written to the logfile). Must belong to the
        same setting as ``agents``.
    interact : statrl.experiments.onerun.Interaction
        Interaction loop of the setting, shared by every agent in the run.
    timeHorizon : int, default=1000
        Number of rounds per interaction.
    nbReplicates : int, default=100
        Number of independent runs per agent. Regret quantiles are taken
        across these, so a handful of replicates gives a very rough band.
    root_folder : str, default='results/'
        Output directory, created if absent. Must end with a separator.

    Returns
    -------
    None
        Everything is written to disk. ``root_folder`` receives a
        ``logfile_*.txt``, one ``regret_*`` pickle per agent, and the figures
        ``Regrets_*.png`` / ``.pdf`` in linear and log-y scale. The
        intermediate ``aux_*`` dumps are deleted on the way out.

    See Also
    --------
    statrl.experiments.parallelruns.multicoreRuns : The parallel layer underneath.
    statrl.experiments.analyzeruns.computeScoreDiffs : Turns the dumps into regret statistics.
    statrl.experiments.plotruns.plotScoreDiffs : Draws the figures.

    Notes
    -----
    Cost grows as ``(len(agents) + 1) * nbReplicates * timeHorizon``. Start
    small — the defaults already amount to 100 000 rounds per agent.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
    >>> from statrl.settings.bandits.stochastic.anytime.agents.Oracle import Oracle
    >>> from statrl.settings.bandits.stochastic.anytime.agents._Random import Random
    >>> from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
    >>> from statrl.settings.utils import klBern
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])              # doctest: +SKIP
    >>> runLargeMulticoreExperiment(                        # doctest: +SKIP
    ...     env,
    ...     agents=[IMED(env.number_arms, klBern), Random(env)],
    ...     oracle=Oracle(env),
    ...     interact=BanditInteraction(),
    ...     timeHorizon=1000, nbReplicates=50,
    ... )
    """
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
        mean, median, quantile1, quantile2,quantile3,quantile4, times = aR.computeScoreDiffs(names, dump_scores, timeHorizon, envName, root_folder=root_folder)
        print("[INFO]  Plot results...")

        title = env.displayname if hasattr(env, "displayname") else envName
        labelx,labely = interact.plotlabels
        plR.plotScoreDiffs(names, envName, (title,labelx,labely), mean, median, quantile1, quantile2,quantile3,quantile4, times, timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder)
        print("[INFO]  Clean Auxiliary files...")
    clear_auxiliaryfiles(env, root_folder)
    print("[INFO]  Massive multicore experiment successfully completed.")
