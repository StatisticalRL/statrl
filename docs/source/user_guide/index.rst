User guide
==========

``statrl`` organises reinforcement-learning problems as a **taxonomy of settings**. Each
setting is a self-contained package under ``statrl.settings`` that defines three things:

- an **environment** — the problem (arms and their reward distributions),
- an **agent** — the learning algorithm, and
- an **interaction loop** — the function that runs an agent against an environment for a
  fixed horizon and records performance.

The shared protocol
-------------------

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

Implemented settings
--------------------

.. toctree::
   :maxdepth: 1

   stochastic_anytime
   stochastic_knownhorizon
   adversarial_lipschitz
   experiments

.. admonition:: Scope

   The stochastic (anytime and known-horizon) and adversarial Lipschitz settings are
   implemented. The ``envs/`` directories and most ``wrappers/`` directories are empty
   scaffolding that mark the intended taxonomy rather than shipped code.
