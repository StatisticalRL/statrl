Quickstart
==========

.. toctree::
   :maxdepth: 1

   install
   core_concepts

``statrl`` is built around one small protocol, shared by every bandit setting:

============  ===========================================================================
Component     Responsibility
============  ===========================================================================
Environment   Holds the arms (reward distributions); ``pull(arm)`` samples a reward.
Agent         ``reset()`` starts a run; ``select_arm()`` chooses an arm; ``update(arm, reward)`` learns.
Interaction   ``interact(env, learner, horizon)`` runs the loop and returns cumulative scores.
============  ===========================================================================

See :doc:`core_concepts` for the interaction loop and how to bring your own environment.

Where to go next
----------------

- :doc:`user_guide/index` — each setting and its algorithms explained.
- :doc:`user_guide/experiments` — run many replicates in parallel and plot regret.
- :doc:`api/index` — the full reference.
