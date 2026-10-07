Anytime
=======

Read more in the :doc:`user guide <../user_guide/stochastic_anytime>`.

Protocol
--------

.. currentmodule:: statrl.settings.bandits.stochastic.anytime

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agent.BanditAgent
   environment.StochasticBanditEnv
   interaction.BanditInteraction

Agents
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agents.IMED.IMED
   agents.Oracle.Oracle
   agents._Random.Random

Environments
------------

Factories building a
:class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`
from a vector of arm means.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   envs.parametric.BernoulliBandit
   envs.parametric.BinomialBandit
   envs.parametric.GaussianBandit
   envs.parametric.TruncatedGaussianBandit
   envs.parametric.RandomBernoulliBandit

Reward distributions
--------------------

The individual arms the factories above are built from.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   envs.distributions.Arm
   envs.distributions.Bernoulli
   envs.distributions.Binomial
   envs.distributions.Gaussian
   envs.distributions.Exponential
   envs.distributions.TruncatedGaussian
   envs.distributions.TruncatedExponential

Renderers
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   renderers.textrenderer.Textrenderer
