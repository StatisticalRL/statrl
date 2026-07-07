Quickstart
==========

``statrl`` is built around one small protocol, shared by every bandit setting:

============  ===========================================================================
Component     Responsibility
============  ===========================================================================
Environment   Holds the arms (reward distributions); ``pull(arm)`` samples a reward.
Agent         ``reset()`` starts a run; ``select_arm()`` chooses an arm; ``update(arm, reward)`` learns.
Interaction   ``interact(env, learner, horizon)`` runs the loop and returns cumulative scores.
============  ===========================================================================

The interaction loop
--------------------

For stochastic, horizon-agnostic bandits the loop lives in
:func:`statrl.settings.bandits.stochastic.anytime.interaction.interact`:

.. code-block:: python

   def interact(env, learner, horizon):
       env.reset()
       learner.reset()
       score = []
       for t in range(horizon):
           arm = learner.select_arm()
           reward = env.pull(arm)
           learner.update(arm, reward)
           score.append(reward + (score[t - 1] if t > 0 else 0))
       return np.array(score)

Bringing your own environment
-----------------------------

.. note::

   The ``envs/`` folders are scaffolding — ``statrl`` does not yet ship a concrete
   environment. You subclass the abstract
   :class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`
   and provide the arms' reward distributions (any object with a ``sample()`` method and
   a ``mean`` attribute).

A minimal sketch of what you provide:

.. code-block:: python

   from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv

   class MyBanditEnv(StochasticBanditEnv):
       name = "my-bandit"

       @property
       def n_arms(self):
           return len(self.rewarddistributions)

       @property
       def means(self):
           return [arm.mean for arm in self.rewarddistributions]

       def pull(self, arm):
           return self.rewarddistributions[arm].sample()

Then pair it with an agent (for example :class:`~statrl.settings.bandits.stochastic.anytime.agents._Random.Random`
for uniform exploration) and call ``interact(env, learner, horizon)``.

Where to go next
----------------

- :doc:`user_guide/index` — each setting and its algorithms explained.
- :doc:`user_guide/experiments` — run many replicates in parallel and plot regret.
- :doc:`api/index` — the full reference.
