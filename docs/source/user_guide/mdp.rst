Markov decision processes
=========================

*Module:* ``statrl.settings.markovdecisionprocess``

An MDP adds the one thing a bandit lacks: **state**. An action no longer just
returns a reward, it also moves the process, so the value of an action depends
on where you are and on where it takes you. An agent must therefore learn a
transition kernel as well as a reward function, and must be willing to accept a
poor reward now to reach a rewarding region later — a trade-off that has no
bandit counterpart.

``discrete_nostructure`` is the setting where nothing links one state-action
pair to another: all :math:`S \times A` pairs must be learned separately. That
is the hardest finite case, and the one classical regret bounds are stated for.

The environment
---------------

:class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.environment.DiscreteMDP`
is built from a transition kernel ``P``, reward distributions ``R``, and an
initial distribution. Unlike a bandit environment it is stateful, and follows
the standard :class:`gymnasium.Env` five-tuple:

.. doctest::

   >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim import RiverSwim
   >>>
   >>> env = RiverSwim(5)
   >>> env.nS, env.nA
   (5, 2)
   >>> state, info = env.reset()
   >>> state, reward, done, truncated, info = env.step(0)     # 0 = right, 1 = left

``info["mean"]`` carries the pair's mean reward for regret accounting and must
not be given to the learner.

Shipped environments
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Environment
     - Description
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim.RiverSwim`
     - The standard hard-exploration benchmark [Strehl2008]_. Going left is safe
       and pays a little; going right may pay a lot but often fails. An agent
       must forgo a certain small reward for many steps to reach an uncertain
       large one — which is exactly what a naive learner will not do.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim.ErgodicRiverSwim`
     - RiverSwim with a small leak on the left action, so every state stays
       reachable under every policy. Required by algorithms assuming
       ergodicity.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.envs.randomMDP.RandomMDP`
     - Randomly generated instances with sparse transitions and rewards.
       Averaging over many draws measures typical behaviour rather than fit to
       one hand-built instance.

Use ``RiverSwim`` to separate algorithms by their exploration, and ``RandomMDP``
to check that a result is not an artefact of a single instance.

The agent
---------

:class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agent.MDPAgent`
is state-aware throughout: ``play(state)`` chooses an action for a state, and
``update(state, action, reward, observation)`` receives the state reached as
well as the reward. That extra argument is the entire difference from a bandit
agent — learning the kernel is half the problem.

Unlike the bandit base class this one is concrete, its defaults implementing a
uniform random policy, so a subclass can override only what it needs.

Shipped agents
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Agent
     - Description
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL.IMEDRL`
     - Carries the IMED index to ergodic MDPs, taking the divergence over the
       joint law of *reward plus next-state bias* rather than the reward alone.
       Assumes an ergodic MDP. See [Pesquerel2022]_.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.PSRL.PSRL`
     - Posterior sampling: draw a whole MDP from the posterior at the start of
       each episode and follow its optimal policy. Sampling a coherent world
       rather than perturbing each pair independently is what makes its
       exploration deep enough for RiverSwim. Assumes Bernoulli rewards. See
       [Osband2013]_.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Oracle.Opti_controller`
     - Solves the true MDP by value iteration; the regret reference. Build it
       with :func:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Oracle.build_opti`.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Random.Random`
     - Uniform exploration. On RiverSwim it essentially never reaches the far
       state, which is the point of the benchmark.
   * - :class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.agents.Human.Human`
     - Prompts on stdin for each action. For exploring an instance by hand; it
       cannot be used in an experiment, which runs replicates in parallel with
       no attached terminal.

PSRL assumes rewards in :math:`[0, 1]` — it updates a Beta posterior with
``reward`` and ``1 - reward``, so rewards outside that range corrupt it
silently. IMED-RL assumes ergodicity: pair it with ``ErgodicRiverSwim``, not
``RiverSwim``.

The interaction loop
--------------------

:class:`~statrl.settings.markovdecisionprocess.discrete_nostructure.interaction.MDPInteraction`
threads the state through the loop and, on a terminal transition, resets the
environment and continues. ``horizon`` therefore counts **steps**, not episodes.

.. doctest::

   >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.agents.PSRL import PSRL
   >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Oracle import Opti_swimmer
   >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.interaction import MDPInteraction
   >>>
   >>> env = RiverSwim(5)
   >>> interaction = MDPInteraction()
   >>> scores = interaction.run(env, PSRL(env.nS, env.nA), horizon=2000)
   >>> scores.shape
   (2000,)

   >>> oracle = Opti_swimmer(env)
   >>> oracle_scores = interaction.run(env, oracle, horizon=2000)
   >>> bool((oracle_scores - scores)[-1] >= 0)
   True

Gridworlds
----------

.. note::

   ``statrl.settings.markovdecisionprocess.gridworld`` is a **work in
   progress** and is deliberately absent from the :doc:`../api/index`.
   :class:`~statrl.settings.markovdecisionprocess.gridworld.envs.gridworlds.GridWorld`
   and its walled variant build, and reuse the ``discrete_nostructure``
   environment, agent, and interaction unchanged — but the setting ships no
   agents of its own, and the hand-coded gridworld oracles
   (``Opti_77_4room``, ``Opti_911_2room``) are not constructible as written.
   Treat it as unsupported until it appears in the API reference.

Two variants of the four-room layout [Sutton1999]_ exist. ``GridWorldWithWall``
keeps wall cells as states, so state
indexing stays a plain ``x * sizeY + y``; ``GridWorld`` drops them, exposing
``mapping`` and ``revmapping`` to translate between the two indexings. Prefer
``GridWorld`` — a state space padded with unreachable cells makes an instance
look harder than it is, and inflates every bound that depends on :math:`S`.

See :doc:`../api/mdp` for the reference.
