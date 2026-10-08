from typing import Callable

import numpy as np
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent
from statrl.settings.utils import klBern, klGauss, randmin


class IMED(BanditAgent):
    """Indexed Minimum Empirical Divergence, an asymptotically optimal bandit algorithm.

    IMED assigns every arm the index

    .. math::

       I_a(t) = N_a(t)\\,\\mathrm{kl}\\!\\left(\\hat{\\mu}_a(t), \\hat{\\mu}^\\star(t)\\right)
                + \\log N_a(t)

    and pulls the arm minimizing it. The first term is large for an arm whose
    empirical mean is confidently below the best one, the second penalizes
    arms already pulled often; minimizing their sum balances exploration
    against exploitation with **no tuning parameter** — no confidence level,
    no exploration bonus, no schedule.

    The regret matches the Lai-Robbins lower bound asymptotically [1]_, provided
    ``kullback`` is the divergence of the true reward family.

    Parameters
    ----------
    nbArms : int
        Number of arms, which must equal ``env.number_arms``.
    kullback : callable, default=:func:`~statrl.settings.utils.klGauss`
        Divergence ``kl(x, y)`` between two means. Choose it to match the
        rewards: :func:`~statrl.settings.utils.klBern` for Bernoulli,
        :func:`~statrl.settings.utils.klGauss` for Gaussian or any
        sub-Gaussian reward. A mismatched choice costs the optimality
        guarantee but stays well defined.
    name : str, default='IMED'
        Label used in logfiles and plot legends. Give two IMED instances
        distinct names when comparing divergences in one experiment.

    Attributes
    ----------
    kl : callable
        The divergence passed as ``kullback``.
    nbDraws : ndarray of shape (nbArms,)
        Number of pulls of each arm, :math:`N_a(t)`.
    cumRewards : ndarray of shape (nbArms,)
        Cumulative reward collected from each arm.
    means : ndarray of shape (nbArms,)
        Empirical mean of each arm, :math:`\\hat{\\mu}_a(t)`.
    maxMeans : float
        Best empirical mean, :math:`\\hat{\\mu}^\\star(t)`.
    indexes : ndarray of shape (nbArms,)
        Current index of each arm; :meth:`select_arm` minimizes over it.

    See Also
    --------
    statrl.settings.bandits.stochastic.batch.agents.BIMED.BIMED :
        The batched, distribution-free variant built on
        :func:`~statrl.settings.utils.KLinf_threshold`.
    statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL.IMEDRL :
        The extension of the same index to ergodic MDPs.

    Notes
    -----
    Arms never pulled have index ``0``, the smallest value the index can take,
    so every arm is played once before any is repeated. Both a pull and an
    update cost :math:`O(K)` time and :math:`O(K)` memory.

    References
    ----------
    .. [1] Honda, J. and Takemura, A. "Non-asymptotic analysis of a new bandit
           algorithm for semi-bounded rewards." *Journal of Machine Learning
           Research*, 16(113):3721-3756, 2015.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.anytime.interaction import BanditInteraction
    >>> from statrl.settings.utils import klBern
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])
    >>> agent = IMED(env.number_arms, kullback=klBern)
    >>> scores = BanditInteraction().run(env, agent, horizon=500)
    >>> int(agent.nbDraws.argmax())          # the best arm is pulled most often
    1
    """

    def __init__(self, nbArms: int, kullback: Callable[[float, float], float] = klGauss, name="IMED") -> None:
        self.kl = kullback
        self.nA = nbArms
        if name is None:
            BanditAgent.__init__(self, name="IMED")
        else:
            BanditAgent.__init__(self, name=name)

    def reset(self) -> None:
        """Clear every statistic before a new independent run.

        Counts, cumulative rewards, means, and indexes are all zeroed, so all
        arms start tied and each is pulled once before any repeat.
        """
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)
        self.means = np.zeros(self.nA)
        self.maxMeans = 0.0
        self.indexes = np.zeros(self.nA)

    def select_arm(self, state: int = 0) -> int:
        """Pull the arm of minimal IMED index, :math:`\\arg\\min_a I_a(t)`.

        Parameters
        ----------
        state : int, default=0
            Ignored; a bandit has no state. Accepted so the agent also fits
            the state-passing signature used in the MDP settings.

        Returns
        -------
        int
            Index of the selected arm. Ties are broken uniformly at random by
            :func:`~statrl.settings.utils.randmin`, which matters at the start
            of a run when every index is still ``0``.
        """
        return randmin(self.indexes)

    def update(self, arm: int, reward: float) -> None:
        """Refresh the empirical means and all affected indices.

        Increments the pull count and cumulative reward of ``arm``, updates
        its empirical mean and the running best mean. If the best mean changes,
        every index is recomputed, otherwise only the pulled arm's index
        can have changed.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward observed for that arm.
        """
        previous_max = self.maxMeans
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1

        # Empirical mean update
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        # Best empirical mean across arms
        self.maxMeans = float(np.max(self.means))

        if self.maxMeans == previous_max:
            draws = self.nbDraws[arm]
            self.indexes[arm] = (
                draws * self.kl(self.means[arm], self.maxMeans)
                + np.log(draws)
            )
            return

        selected = self.nbDraws > 0 # Identify arms that have been pulled
        means = self.means[selected]
        draws = self.nbDraws[selected]

        if self.kl is klBern:
            divergences = klBern(means, self.maxMeans)
        elif self.kl is klGauss:
            divergences = (means - self.maxMeans) ** 2 / 2
        else:
            divergences = np.array([
                self.kl(mean, self.maxMeans)
                for mean in means
            ])

        indexes = np.zeros(self.nA)
        indexes[selected] = draws * divergences + np.log(draws)
        self.indexes = indexes
