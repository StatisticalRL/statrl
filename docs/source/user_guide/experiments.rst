Running experiments
===================

*Module:* ``statrl.experiments``

The ``experiments`` package is the harness for **benchmarking** agents: it runs an
algorithm for many independent replicates (in parallel), measures regret against an oracle,
and plots the result. Everything is glued together by one entry point.

The orchestrator
----------------

:func:`~statrl.experiments.massiveruns.runLargeMulticoreExperiment`:

.. code-block:: python

   runLargeMulticoreExperiment(
       env, agents, oracle, interact,
       timeHorizon=1000, nbReplicates=100, root_folder="results/",
   )

For each agent it launches ``nbReplicates`` runs, runs the ``oracle`` for the same
horizon, computes the regret (oracle score minus agent score), writes a logfile, and saves
regret plots under ``root_folder``.

The pipeline underneath
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Function
     - Role
   * - :func:`~statrl.experiments.parallelruns.multicoreRuns`
     - Runs ``nbReplicates`` replicates of one learner across CPU cores with
       :mod:`joblib`, building environments via ``gymnasium.make``. Returns the scores and
       the mean elapsed time.
   * - :func:`~statrl.experiments.onerun.oneRunWithDump`
     - Runs a single interaction and pickles the score series to disk (one replicate).
   * - :func:`~statrl.experiments.analyzeruns.computeScoreDiffs`
     - Loads the pickled scores and computes regret statistics vs. the oracle: mean,
       median, and the 0.25 / 0.75 quantiles over time.
   * - :func:`~statrl.experiments.plotruns.plotScoreDiffs`
     - Produces the regret plots (linear and log-scaled ``y``) as PNG/PDF with
       :mod:`matplotlib`.
   * - :func:`~statrl.experiments.utils.dump`
     - Pickle helper used to persist intermediate score series (with
       :func:`~statrl.experiments.utils.clear_auxiliaryfiles` to clean them up).

Results are written under ``root_folder`` (default ``"results/"``): the intermediate
per-replicate dumps, a logfile, and the regret figures.
