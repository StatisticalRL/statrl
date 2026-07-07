Adversarial Lipschitz online optimization
=========================================

*Module:* ``statrl.settings.bandits.adversarial.lipschitz``

This setting leaves the finite-arm world behind. The action space is a **continuous metric
space**, and at each round an adversary chooses a reward function :math:`f_t(x)` (assumed
Lipschitz). The learner picks an action :math:`x_t`, then observes the reward
:math:`f_t(x_t)`. Because the reward functions are adversarial rather than drawn from a
fixed distribution, the algorithms differ from the stochastic settings.

The environment
---------------

:class:`~statrl.settings.bandits.adversarial.lipschitz.environment.LipschitzAdversarialEnv`
is a ``gymnasium``-like environment constructed from:

- ``action_space`` — the continuous domain (e.g. a ``Box``),
- ``reward_function_sequence`` — a callable mapping a round ``t`` to its reward function
  ``f_t(x)``,
- ``horizon`` — the number of rounds ``T``, and
- ``observation_fn`` — optional; produces the observation at each round (trivial in the
  pure bandit case).

``step(action)`` returns the ``gymnasium`` 5-tuple ``(observation, reward, terminated,
truncated, info)``.

The agent
---------

The base :class:`~statrl.settings.bandits.adversarial.lipschitz.agent.Agent` defines
``play(observation)`` (return an action) and an optional ``update(action, reward,
observation=None)``.

ALF — Adversarial Lipschitz Forecaster
--------------------------------------

:class:`~statrl.settings.bandits.adversarial.lipschitz.agents.ALF.ALFLearner` reduces the
continuous problem to a finite one: it builds an **ε-net** (a discrete cover) of the action
space, then runs an exponential-weights / Hedge update over those points. Constructor
parameters:

- ``epsilon`` — discretization resolution of the cover,
- ``eta`` — learning rate for the exponential weights,
- ``horizon`` — used to tune ``epsilon``/``eta``,
- ``metric`` — optional metric for building the cover, and
- ``sampling`` — ``"argmax"`` (play the best expert) or ``"sample"`` (draw from the weight
  distribution).

The interaction loop
--------------------

:func:`~statrl.settings.bandits.adversarial.lipschitz.interaction.run_lipschitz_online_learning`
drives ``play`` → ``step`` → ``update`` over the horizon and returns the reward series as
an array.
