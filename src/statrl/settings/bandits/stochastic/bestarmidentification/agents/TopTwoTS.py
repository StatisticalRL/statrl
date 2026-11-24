"""Top-Two Thompson Sampling for Bernoulli best-arm identification.

Implementation of the TTTS algorithm described by Russo (2016, 2020).

The agent maintains independent Beta(1, 1) priors for the Bernoulli
success probabilities. At each round it samples a posterior leader and,
with probability ``1 - beta``, samples again until a different arm is
identified as the challenger.

This implementation is intended for the StatRL BAI setting:
``select_arm``, ``update``, ``recommend_arm``, ``is_done``.
"""

import numpy as np

from statrl.settings.bandits.stochastic.bestarmidentification.agent import (
    BAIBanditAgent,
)


class TopTwoTS(BAIBanditAgent):
    """Top-Two Thompson Sampling (TTTS) for Bernoulli bandits.

    Parameters
    ----------
    nbArms : int
        Number of arms.
    beta : float, default=0.5
        Probability of selecting the posterior-sampling leader. With
        probability ``1 - beta``, the challenger is selected instead.
        Must lie strictly between zero and one.
    name : str, default="TopTwoTS"
        Agent name used by StatRL for experiment outputs.
    seed : int, default=1
        Seed for the agent-local random number generator.

    Notes
    -----
    The prior for every arm is Beta(1, 1). After observing ``s`` successes
    and ``f`` failures, its posterior is Beta(s + 1, f + 1).

    The implementation uses the rejection-sampling version of TTTS:
    posterior samples are drawn until their maximizing arm differs from
    the initially sampled leader. This avoids explicitly computing the
    posterior probability that each arm is optimal.
    """

    def __init__(self, nbArms, beta=0.5, name="TopTwoTS", seed=1):
        if not isinstance(nbArms, (int, np.integer)) or nbArms < 1:
            raise ValueError("nbArms must be a positive integer.")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie strictly between 0 and 1.")

        super().__init__(name=name, seed=seed)

        self.nA = int(nbArms)
        self.beta = float(beta)

    def reset(self):
        """Reset the agent and initialize independent Beta(1, 1) priors."""
        super().reset()

        self.nbDraws = np.zeros(self.nA, dtype=int)
        self.cumRewards = np.zeros(self.nA, dtype=float)

        # Beta posterior parameters: alpha = successes + 1,
        #                         b = failures + 1.
        self.alpha = np.ones(self.nA, dtype=float)
        self.b = np.ones(self.nA, dtype=float)

        self.leader = None
        self.challenger = None

    def _sample_posterior_winner(self):
        """Draw one independent posterior sample per arm and return its argmax."""
        samples = self.np_random.beta(self.alpha, self.b)
        return int(np.argmax(samples))

    def select_arm(self):
        """Select the next arm according to Top-Two Thompson Sampling.

        First, a posterior sample determines the leader. The leader is
        selected with probability ``beta``. Otherwise, posterior samples
        are drawn repeatedly until a different arm wins; that arm is the
        challenger and is selected.

        Returns
        -------
        int
            Index of the selected arm.
        """
        leader = self._sample_posterior_winner()
        self.leader = leader

        if self.np_random.random() < self.beta:
            self.challenger = None
            return leader

        challenger = self._sample_posterior_winner()
        while challenger == leader:
            challenger = self._sample_posterior_winner()

        self.challenger = challenger
        return challenger

    def update(self, arm, reward):
        """Update the Beta posterior after observing a Bernoulli reward.

        Parameters
        ----------
        arm : int
            Index of the selected arm.
        reward : float
            Observed reward, which must be either zero or one.
        """
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.nA:
            raise ValueError(f"arm must be an integer in [0, {self.nA - 1}].")

        if reward not in (0, 1):
            raise ValueError(
                "TopTwoTS uses a Bernoulli likelihood: reward must be 0 or 1."
            )

        self.nbDraws[arm] += 1
        self.cumRewards[arm] += reward

        if reward == 1:
            self.alpha[arm] += 1.0
        else:
            self.b[arm] += 1.0

    def recommend_arm(self):
        """Recommend the arm with the largest posterior mean.

        Under the Beta posterior, the posterior mean of arm ``a`` is
        ``alpha[a] / (alpha[a] + b[a])``.
        """
        posterior_means = self.alpha / (self.alpha + self.b)
        return int(np.argmax(posterior_means))

    def is_done(self):
        """Return whether the agent has reached its stopping condition.

        TTTS is implemented here as a fixed-budget allocation rule.
        Consequently, stopping is controlled by the horizon in
        ``BAIInteraction`` rather than by an internal stopping criterion.
        """
        return False
