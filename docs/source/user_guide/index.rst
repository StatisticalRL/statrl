User guide
==========

``statrl`` organises reinforcement-learning problems as a **taxonomy of settings**. Each
setting is a self-contained package under ``statrl.settings`` that defines three things:

- an **environment** : the problem (arms and their reward distributions),
- an **agent** : the learning algorithm, and
- an **interaction loop** : the function that runs an agent against an environment for a
  fixed horizon and records performance.

.. toctree::
   :maxdepth: 1

   ../quickstart

 

Implemented settings
--------------------

.. toctree::
   :maxdepth: 1

   stochastic
   adversarial_lipschitz
   experiments 