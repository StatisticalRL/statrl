Stochastic bandits
==================

*Module:* ``statrl.settings.bandits.stochastic``

In the **stochastic** setting each arm has a fixed, unknown reward distribution, and
pulling an arm draws an independent sample from it. ``statrl`` ships two variants,
distinguished by what the agent knows about the time horizon:

Every stochastic-bandit agent implements the same three methods:

``reset()``
   Start a new, independent run (clear all statistics).

``select_arm() -> int``
   Return the index of the arm to pull next.

``update(arm, reward)``
   Incorporate the observed reward for the chosen arm.

An environment exposes its arms through ``n_arms`` and ``means`` (the latter for
evaluation and oracle construction only) and samples rewards through ``pull(arm)``. The
loop ``interact(env, learner, horizon)`` ties them together and returns a cumulative-score
:class:`numpy.ndarray`.


.. toctree::
   :maxdepth: 1

   stochastic_anytime
   stochastic_knownhorizon
