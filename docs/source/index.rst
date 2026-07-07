statrl
======

.. rst-class:: lead

   The **Statistical Reinforcement Learning Toolkit** — a research library for
   multi-armed bandit algorithms, organised as a taxonomy of *settings*, each with a
   matching environment, agent, and interaction loop.

``statrl`` gives you a small, explicit protocol shared across bandit settings — an
**environment** exposing a set of arms, an **agent** that selects arms and learns from
rewards, and an **interaction loop** that runs the two against each other and returns a
cumulative-score time series. On top of that sit reference algorithms (IMED, an oracle,
uniform exploration, an adversarial Lipschitz forecaster) and an ``experiments`` harness
for running many replicates in parallel and plotting regret.

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: quickstart
      :link-type: doc

      Install ``statrl`` and see the environment / agent / interaction protocol.

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Narrative walkthrough of each bandit setting and the algorithms it ships.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Auto-generated reference for every public module, class, and function.

   .. grid-item-card:: :octicon:`beaker` Experiments
      :link: user_guide/experiments
      :link-type: doc

      Run benchmarks in parallel.

.. toctree::
   :hidden:
   :caption: User guide

   user_guide/index

.. toctree::
   :hidden:
   :caption: Reference

   api/index
