
from typing import Optional

import numpy as np

from gymnasium import Env as Environment
from gymnasium.utils import seeding

class StochasticBanditEnv(Environment):
    """Stochastic multi-armed bandit environment.

    Each arm carries an independent reward distribution; :meth:`step` draws one
    sample from the chosen arm. The environment is *stateless*: interactions
    never modify the underlying distributions. 

    Parameters
    ----------
    rewarddistributions : list
        One distribution per arm. Each must expose a ``mean`` attribute and a
        ``sample()`` method — see
        :class:`~statrl.settings.bandits.stochastic.anytime.envs.distributions.Arm`.
    name : str
        Label used in logfiles, plot titles, and dump filenames.
    last : tuple of (int or None, float), default=(None, 0.0)
        Most recent ``(arm, reward)`` pair, consumed by the renderers.

    Attributes
    ----------
    renderers : list
        Renderers notified on every :meth:`render` call. Empty by default;
        :meth:`~statrl.settings.bandits.stochastic.anytime.interaction.BanditInteraction.renderrun`
        installs a
        :class:`~statrl.settings.bandits.stochastic.anytime.renderers.textrenderer.Textrenderer`.
    np_random : numpy.random.Generator
        Environment-local generator, available after :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.envs.parametric.BernoulliBandit :
        Factory for a Bernoulli instance.
    statrl.settings.bandits.stochastic.batch.environment.BatchMAB :
        Wrapper turning any instance into a batched bandit.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])
    >>> env.number_arms
    3
    >>> env.optimal_arm
    1
    >>> _ = env.reset(seed=0)
    >>> reward = env.step(1)
    >>> reward in (0.0, 1.0)
    True
    """

    def __init__(self, rewarddistributions: list, name: str, last: tuple[Optional[int], float] = (None, 0.0)) -> None:
        self.rewarddistributions = rewarddistributions
        self._means = np.asarray(
            [arm.mean for arm in rewarddistributions], dtype=float
        )
        self.name = name
        self.displayname: str = name
        self.renderers: list = []
        self.last = last

    @property
    def number_arms(self) -> int:
        """Number of available arms."""
        return len(self.rewarddistributions)

    @property
    def means(self) -> list[float]:
        """
        Mean reward of every arm. 
        """
        return self._means.tolist()

    @property
    def optimal_mean(self) -> float:
        """float: Mean reward of the best arm, :math:`\\mu^\\star`.

        Used to define regret.
        """
        return float(np.max(self._means))

    @property
    def optimal_arm(self) -> int:
        """int: Index of the best arm.

        Ties are broken by :func:`numpy.argmax`, i.e. the lowest index wins. 
        """
        return int(np.argmax(self._means))

    def step(self, arm: int) -> float:  # type: ignore[override]  # bandit API: reward only, not gym's 5-tuple
        """Sample one reward from the given arm.

        Parameters
        ----------
        arm : int
            Index of the arm to pull, in ``range(number_arms)``.

        Returns
        -------
        float
            An independent draw from that arm's reward distribution.
 
        """
        r = self.rewarddistributions[arm].sample()
        self.last=(arm,r)
        return r

    def expected_reward(self, arm: int) -> float:
        """Mean reward of an arm, for regret accounting only.
 
        Parameters
        ----------
        arm : int
            Index of the arm.

        Returns
        -------
        float
            That arm's true mean. Never pass this to a learner.
        """
        return float(self._means[arm])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> int:  # type: ignore[override]  # bandit API: no observation tuple
        """Start a new run by reseeding the environment.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's generator. ``None`` draws a fresh one.
        options : dict, optional
            Unused; accepted for :class:`gymnasium.Env` compatibility.

        Returns
        -------
        int
            The constant dummy observation ``0`` — a bandit is stateless.
        """
        #super().reset(seed=seed, options=options)
        self.np_random, self.seed = seeding.np_random(seed)
        self.last = (None,0)
        return 0

    def render(self, mode: str = 'human') -> None:
        """Forward the last ``(arm, reward)`` pair to every attached renderer.

        Parameters
        ----------
        mode : str, default='human'
            Unused; accepted for :class:`gymnasium.Env` compatibility. Output
            is whatever the objects in :attr:`renderers` produce.
        """
        for re in self.renderers:
            re.render(self,self.last)

    def close(self) -> None:
        """Release every attached renderer at the end of a rendered run."""
        for re in self.renderers:
            re.stop(self)
