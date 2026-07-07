Anytime
=======

*Module:* ``statrl.settings.bandits.stochastic.anytime``

In the **stochastic** setting each arm has a fixed, unknown reward distribution, and
pulling an arm draws an independent sample from it. The **anytime** variant makes no
assumption about the horizon: the agent should perform well whenever it is stopped, so it
cannot tune its behaviour to a known number of rounds.

The environment
---------------

:class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv` is an
abstract, *stateless* multi-armed bandit environment — interactions never modify the
underlying distributions. A subclass supplies the arms' reward distributions and
implements:

- ``n_arms`` — number of arms,
- ``means`` — the true mean of each arm (used for evaluation and to build an oracle), and
- ``pull(arm)`` — draw one reward from the given arm.

It also derives ``optimal_mean`` and ``optimal_arm`` from ``means``, and supports seeding
and rendering via ``gymnasium`` utilities.

The agent
---------

:class:`~statrl.settings.bandits.stochastic.anytime.agent.BanditAgent` is the abstract base
class. Every agent carries a ``name`` and implements ``reset()``, ``select_arm()``, and
``update(arm, reward)``.

Shipped agents
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Agent
     - Description
   * - :class:`~statrl.settings.bandits.stochastic.anytime.agents.IMED.IMED`
     - Indexed Minimum Empirical Divergence — an asymptotically optimal, index-based
       algorithm. See the deep dive below.
   * - :class:`~statrl.settings.bandits.stochastic.anytime.agents._Oracle.Oracle`
     - Baseline that always plays the best arm; used as the regret reference.
   * - :class:`~statrl.settings.bandits.stochastic.anytime.agents._Random.Random`
     - Uniform exploration — samples an arm uniformly at random each round.

The interaction loop
--------------------

:func:`~statrl.settings.bandits.stochastic.anytime.interaction.interact` resets the
environment and learner, then repeats ``select_arm`` → ``pull`` → ``update`` for
``horizon`` rounds, accumulating the reward into a running score it returns as an array.

IMED in depth
-------------

.. toctree::
   :maxdepth: 1

   imed
