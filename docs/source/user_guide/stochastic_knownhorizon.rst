Known horizon
=============

*Module:* ``statrl.settings.bandits.stochastic.knownhorizon``

This setting is the stochastic bandit problem again, but with the **time horizon known in
advance**. That extra information lets algorithms budget exploration against a fixed number
of rounds, so the agent's ``reset`` receives the horizon:

.. code-block:: python

   class BanditAgent(ABC):
       def reset(self, timehorizon): ...
       def select_arm(self) -> int: ...
       def update(self, arm, reward): ...

The environment
---------------

:class:`~statrl.settings.bandits.stochastic.knownhorizon.environment.StochasticBanditEnv`
mirrors the anytime environment — arms with reward distributions, ``n_arms``, ``means``,
``pull(arm)`` — the difference is entirely on the agent side.

Bridging to the anytime setting
-------------------------------

Because the two stochastic settings share the same environment/agent shape, the
``wrappers`` package adapts an object built for one setting to the other. Four adapters are
provided in
:mod:`~statrl.settings.bandits.stochastic.knownhorizon.wrappers.wrapper_anytime_knownhorizon`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Wrapper
     - Adapts …
   * - ``AnytimeToKnownHorizonEnvironmentWrapper``
     - an anytime environment for use in the known-horizon setting.
   * - ``KnownHorizonToAnytimeEnvironmentWrapper``
     - a known-horizon environment for use in the anytime setting.
   * - ``AnytimeToKnownHorizonAgentWrapper``
     - an anytime agent so it accepts a horizon at ``reset``.
   * - ``KnownHorizonToAnytimeAgentWrapper``
     - a known-horizon agent (given a fixed horizon) for anytime use.

This lets you benchmark an algorithm designed for one setting against the environments and
loops of the other without reimplementing it.

The interaction loop
--------------------

:func:`~statrl.settings.bandits.stochastic.knownhorizon.interaction.interact` runs the same
``select_arm`` → ``pull`` → ``update`` cycle for ``horizon`` rounds and returns the
cumulative-score array.
