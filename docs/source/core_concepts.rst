Core concepts
=============

Everything in ``statrl`` is organised around one idea: a **setting** is an environment, an agent, and an interaction loop. 

The three roles
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Role
     - Responsibility
   * - **Environment**
     - The arms and their reward distributions, or the
       states, actions, and transition kernel. Samples feedback, and knows the
       ground truth an agent must not see.
   * - **Agent**
     - The learning algorithm. Chooses what to do next and updates itself from
       whatever feedback it observes.
   * - **Interaction**
     - Runs the two against each other for a fixed horizon and returns the
       cumulative score used to compute regret.

The interaction loop
--------------------

Each setting's loop is a subclass of
:class:`~statrl.experiments.onerun.Interaction`. For the anytime stochastic
bandit it is three lines repeated ``horizon`` times:

.. code-block:: python

   for t in range(horizon):
       arm = learner.select_arm()          # agent decides
       reward = env.step(arm)              # environment responds
       learner.update(arm, reward)         # agent learns
       steps_scores[t] = env.expected_reward(arm)

   return np.cumsum(steps_scores)

Note that the loop records means, not rewards.** ``steps_scores[t]`` is the *expected*
reward of the arm played, not the reward that was actually observed. The agent
still only ever sees the realized ``reward``. Accumulating means strips the
reward noise out of the score, so the regret curve reflects the quality of the
agent's choices rather than the luck of its draws.
 

Regret
------

A score is never meaningful on its own, only against what was achievable. So
every experiment also runs an **oracle**, an agent holding the ground truth:
:class:`~statrl.settings.bandits.stochastic.anytime.agents.Oracle.Oracle`
plays the best arm every round.
 