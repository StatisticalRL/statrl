import io
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless; must precede any pylab import (plotruns/massiveruns)

import numpy as np
from joblib import parallel_backend

from experiments.onerun import oneRunWithDump
from experiments.parallelruns import multicoreRuns
from experiments.analyzeruns import computeScoreDiffs
from experiments.plotruns import plotScoreDiffs
from experiments.massiveruns import runLargeMulticoreExperiment
from settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
from settings.bandits.stochastic.anytime.agents.IMED import IMED
from settings.bandits.stochastic.anytime.agents._Oracle import Oracle
from settings.bandits.stochastic.anytime.interaction import BanditInteraction

MEANS = [0.2, 0.9, 0.5]
H = 20  # tiny horizon keeps runs fast


def _setup():
    return BernoulliBandit(MEANS), IMED(len(MEANS)), BanditInteraction()


def test_onerun_dumps_score_series(tmp_path):
    env, learner, interact = _setup()
    fname = oneRunWithDump(env, learner, interact, H, str(tmp_path) + "/")
    assert os.path.exists(fname)
    with open(fname, "rb") as f:
        scores = pickle.load(f)
    assert len(scores) == H


def test_parallelruns_runs_replicates(tmp_path):
    env, learner, interact = _setup()
    with parallel_backend("sequential"):
        scores, elapsed = multicoreRuns(
            env, learner, interact, 3, H, oneRunWithDump, str(tmp_path) + "/"
        )
    assert len(scores) == 3
    assert all(os.path.exists(fn) for fn in scores)
    assert elapsed >= 0


def test_analyzeruns_computes_regret(tmp_path):
    root = str(tmp_path) + "/"

    def _dump(arr, tag):
        p = root + tag
        with open(p, "wb") as f:
            pickle.dump(np.asarray(arr, dtype=float), f)
        return p

    alg = [_dump(np.arange(H) * 1.0, f"alg_{i}") for i in range(2)]
    oracle = [_dump(np.arange(H) * 2.0, f"orc_{i}") for i in range(2)]
    mean, median, q1, q2, times = computeScoreDiffs(
        ["alg0"], [alg, oracle], H, "envX", root
    )
    assert len(times) == H
    assert len(mean) == 1 and len(mean[0]) == H
    # regret = oracle(2t) - alg(t) = t
    np.testing.assert_allclose(mean[0], np.arange(H) * 1.0)


def test_plotruns_writes_figures(tmp_path):
    curve = np.linspace(0.0, 10.0, H)
    logfile = io.StringIO()
    plotScoreDiffs(
        ["alg0"], "envX", "title",
        [curve], [curve], [curve * 0.9], [curve * 1.1],
        list(range(H)), H,
        logfile=logfile, timestamp="t", root_folder=str(tmp_path) + "/",
    )
    assert len(list(tmp_path.glob("Regrets_*.png"))) == 2   # linear + ylog
    assert len(list(tmp_path.glob("Regrets_*.pdf"))) == 2
    assert "regret" in logfile.getvalue()


def test_massiveruns_end_to_end(tmp_path):
    env = BernoulliBandit(MEANS)
    agents = [IMED(len(MEANS))]
    oracle = Oracle(env)
    with parallel_backend("sequential"):
        runLargeMulticoreExperiment(
            env, agents, oracle, BanditInteraction(),
            timeHorizon=H, nbReplicates=2, root_folder=str(tmp_path) + "/",
        )
    assert list(tmp_path.glob("logfile_*.txt"))
    assert list(tmp_path.glob("Regrets_*.png"))
