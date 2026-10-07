Batched bandits
===============

*Module:* ``statrl.settings.bandits.stochastic.batch``

In the **batched** setting the agent must commit to a whole block of pulls
before observing any of their rewards. This is the realistic regime whenever
feedback is delayed — a clinical trial recruits a cohort before reading any
outcome, an A/B test runs for a week before it is analysed, a physical
experiment is run in plates. The agent cannot react pull by pull, so it must
decide *in advance* how to spread a batch across arms.

The cost of batching is not the number of pulls but the staleness of the
statistics used to allocate them. A large batch is committed on statistics
frozen at its start, so the harm grows with the batch size — which is why
schedules matter as much as algorithms here.

The environment
---------------

:class:`~statrl.settings.bandits.stochastic.batch.environment.BatchMAB` wraps any
:class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`,
so every stochastic instance is available here for free:

.. doctest::

   >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
   >>> from statrl.settings.bandits.stochastic.batch.environment import BatchMAB
   >>>
   >>> env = BatchMAB(BernoulliBandit([0.2, 0.9, 0.5]), batchsize=[2, 4, 8])
   >>> info = env.reset()
   >>> info["nextbatchsize"]
   2
   >>> rewards, info = env.step([0, 1])       # one arm per slot of the batch
   >>> len(rewards), info["nextbatchsize"]
   (2, 4)

``step`` takes a *list* of arms and returns a list of rewards. Each ``info``
announces the size of the next batch, which the agent needs before it can act;
``info["mean"]`` carries the batch's expected score for regret accounting.

.. warning::

   A round is a **batch**, not a pull. Under a growing schedule a horizon of 100
   can amount to millions of pulls — ``"doubleexp"`` reaches
   :math:`e^{2^\ell}`. Check what your schedule sums to before setting a
   horizon.

Batch schedules
---------------

The schedule maps a round index to that round's batch size. Pass a callable, or
a plain list — which is picklable and therefore survives the process pool used
by :func:`~statrl.experiments.parallelruns.multicoreRuns`, where a lambda would
not.

The environment factories accept a schedule by name:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Name
     - Size at round :math:`\ell`
     - Use
   * - ``"constant"``
     - 10
     - Baseline; batching cost stays flat.
   * - ``"linear"``
     - :math:`\ell + 1`
     - Mild growth.
   * - ``"quadratic"``, ``"cubic"``
     - :math:`(\ell+1)^2`, :math:`(\ell+1)^3`
     - Fewer, larger batches.
   * - ``"exp"``, ``"doubleexp"``
     - :math:`2^\ell`, :math:`e^{2^\ell}`
     - Very few rounds; keep the horizon small.
   * - ``"abrupt"``
     - 100, then :math:`(\ell+1)^3`
     - Tests adaptation to a sudden change.
   * - ``"exotic"``
     - Cycles four growth rates
     - Stresses agents assuming a regular schedule.
   * - ``"baba,<horizon>"``
     - The BABA epoch grid
     - Required by :class:`~statrl.settings.bandits.stochastic.batch.agents.BABA.BABA`.

The agent
---------

:class:`~statrl.settings.bandits.stochastic.batch.agent.BatchBanditAgent` exposes two
levels. ``play()`` and ``update(arm, reward)`` are the per-pull rules;
``batchplay(batchsize)`` and ``batchupdate(batcharm, batchreward)`` are what the
interaction loop calls, and are the ones a subclass must implement.

The split is what allows the interesting behaviour: an agent may update its
index *between* the pulls of a batch — using only information available when
the batch began — even though no reward has arrived. Incrementing :math:`N_a`
after each committed pull is enough to keep a batch from pouring all of itself
into one arm.

Shipped agents
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Agent
     - Description
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents.BIMED.BIMED`
     - IMED with the non-parametric :math:`K_{\inf}`, recomputed inside the
       batch. Set ``batchagnostic=True`` for a control that ignores the batch
       structure.
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents.BCB.BCB`
     - Non-parametric Thompson sampling with a Dirichlet prior anchored at the
       reward bound. See [Gautron2024]_.
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents.BCB.BCBnaif`
     - BCB without the optimistic within-batch count increment; the control
       isolating what that increment buys.
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents.BABA.BABA`
     - Five-phase schedule-driven learner. See [Jin2021]_. Requires the
       ``"baba,<horizon>"`` schedule.
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents.Oracle.Oracle`
     - Fills every batch with the best arm; the regret reference.
   * - :class:`~statrl.settings.bandits.stochastic.batch.agents._Random.Random`
     - Uniform exploration; the control whose regret batching does not affect.

Every agent except BABA takes a ``bound`` — a known upper bound on the reward
support. It is what makes the non-parametric divergence well defined, so
understating it invalidates the index. Pair them with
:class:`~statrl.settings.bandits.stochastic.batch.envs.parametric.BatchTruncatedGaussianBandit`,
whose support is bounded by construction.

A full run
----------

.. doctest::

   >>> from statrl.settings.bandits.stochastic.batch.agents.BIMED import BIMED
   >>> from statrl.settings.bandits.stochastic.batch.agents.Oracle import Oracle
   >>> from statrl.settings.bandits.stochastic.batch.interaction import BatchBanditInteraction
   >>>
   >>> env = BatchMAB(BernoulliBandit([0.2, 0.9, 0.5]), batchsize=[8] * 100)
   >>> interaction = BatchBanditInteraction()
   >>>
   >>> scores = interaction.run(env, BIMED(3, bound=1.0), horizon=100)
   >>> oracle_scores = interaction.run(env, Oracle(env), horizon=100)
   >>> bool((oracle_scores - scores)[-1] < 60)     # 800 pulls, 8 at a time
   True

Note the oracle's regret is zero by construction: knowing the best arm, it loses
nothing to batching. All of the regret above is the price of *learning* under
delayed feedback.

See :doc:`experiments` to benchmark these over many replicates, and
:doc:`../api/batch` for the reference.
