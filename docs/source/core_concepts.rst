Core concepts
=============

The interaction loop
--------------------

To be completed

Bringing your own environment
-----------------------------


A minimal sketch:

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
