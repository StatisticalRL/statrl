Quickstart
==========

.. toctree::
   :hidden:

   install
   core_concepts

Install ``statrl`` (see :doc:`install`), then run an agent against a bandit:

.. doctest::

   >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
   >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
   >>> from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
   >>> from statrl.settings.utils import klBern
   >>>
   >>> env = BernoulliBandit([0.2, 0.9, 0.5])
   >>> agent = IMED(env.number_arms, kullback=klBern)
   >>> scores = BanditInteraction().run(env, agent, horizon=2000)

``scores`` is the cumulative expected reward, one entry per round:

.. doctest::

   >>> scores.shape
   (2000,)

Did it learn? Arm 1 is the best, and it should have taken almost every pull:

.. doctest::

   >>> env.optimal_arm
   1
   >>> bool(agent.nbDraws[1] > 0.95 * agent.nbDraws.sum())
   True

Measuring regret
----------------

Regret compares that score against an oracle that always plays the best arm.
Because the loop accumulates *expected* rewards, the two scores are directly
comparable:

.. doctest::

   >>> from statrl.settings.bandits.stochastic.anytime.agents.Oracle import Oracle
   >>>
   >>> oracle_scores = BanditInteraction().run(env, Oracle(env), horizon=2000)
   >>> regret = oracle_scores - scores
   >>> bool(regret[-1] < 40)          # IMED's regret grows logarithmically
   True

Compare with uniform exploration, whose regret grows linearly:

.. doctest::

   >>> from statrl.settings.bandits.stochastic.anytime.agents._Random import Random
   >>>
   >>> random_scores = BanditInteraction().run(env, Random(env), horizon=2000)
   >>> bool((oracle_scores - random_scores)[-1] > 500)
   True

That gap — tens versus hundreds — is the whole point of the library, and
:doc:`user_guide/experiments` turns it into a plot averaged over replicates.

Watching a run
--------------

On a short run, ``renderrun`` prints each pull instead of returning a score.
Arms are labelled ``A``, ``B``, ``C``...:

.. code-block:: python

   BanditInteraction().renderrun(env, IMED(env.number_arms, klBern), horizon=5)

.. code-block:: text

   Environment: MAB-Bernoulli-means-0.2-0.9-0.5
   Actions: ABC
   ------------------------------
   (A)     r=0.00
   (B)     r=1.00
   (C)     r=1.00
   (B)     r=1.00
   (B)     r=1.00
   ------------------------------

The protocol
------------

Every bandit setting shares the same three-part protocol:

============  ===========================================================================
Component     Responsibility
============  ===========================================================================
Environment   Holds the arms; ``step(arm)`` samples a reward.
Agent         ``reset()`` starts a run; ``select_arm()`` chooses an arm; ``update(arm, reward)`` learns.
Interaction   ``run(env, learner, horizon)`` runs the loop and returns cumulative scores.
============  ===========================================================================

See :doc:`core_concepts` for how the loop works and how to plug in your own
environment or agent.

Where to go next
----------------

- :doc:`core_concepts` — the environment / agent / interaction protocol in detail.
- :doc:`user_guide/index` — each setting and the algorithms it ships.
- :doc:`user_guide/experiments` — many replicates in parallel, with regret plots.
- :doc:`api/index` — the full reference.
