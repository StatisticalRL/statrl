Markov decision processes
=========================

Read more in the :doc:`user guide <../user_guide/mdp>`.

Protocol
--------

.. currentmodule:: statrl.settings.markovdecisionprocess.discrete_nostructure

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agent.MDPAgent
   environment.DiscreteMDP
   interaction.MDPInteraction

Agents
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   agents.IMED_RL.IMEDRL
   agents.PSRL.PSRL
   agents._Oracle.build_opti
   agents._Oracle.Opti_controller
   agents._Oracle.Opti_swimmer
   agents._Random.Random
   agents.Human.Human
   agents.Human.keyboard_waitfor

Environments
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   envs.riverswim.RiverSwim
   envs.riverswim.ErgodicRiverSwim
   envs.randomMDP.RandomMDP

Renderers
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   renderers.textRenderer.TextRenderer
