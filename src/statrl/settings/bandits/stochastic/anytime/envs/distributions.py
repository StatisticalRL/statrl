"""arms: each exposes .mean and a sample() method producing rewards"""

from functools import partial
from math import exp, log, sqrt
from random import random

from typing import Any, Callable, Optional

import numpy as np

from scipy.stats import bernoulli, binom, norm, expon, truncnorm


class Arm:
    """One bandit arm, adapting a frozen :mod:`scipy.stats` distribution.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Any object exposing ``mean()`` and ``rvs()``.

    Attributes
    ----------
    mean : float
        True mean of the arm, used for regret accounting only.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> arm = Arm(bernoulli(0.3))
    >>> float(arm.mean)
    0.3
    """

    def __init__(
        self, dist: Any, sampler: Optional[Callable[..., Any]] = None
    ) -> None:
        self._dist = dist
        self._sampler = sampler
        self.mean = dist.mean()

    def sample(self) -> float:
        """Draw one independent reward.

        Returns
        -------
        float
            A sample from the underlying distribution.
        """
        if self._sampler is not None:
            return self._sampler()
        return self._dist.rvs()

    def sample_many(self, size: int) -> np.ndarray:
        """Draw several independent rewards in one vectorized call."""
        if self._sampler is not None:
            return np.asarray(self._sampler(size=size))
        return np.asarray(self._dist.rvs(size=size))


def Bernoulli(p: float) -> Arm:
    """Bernoulli arm with success probability ``p``, so rewards are 0 or 1."""
    return Arm(bernoulli(p), sampler=partial(np.random.binomial, 1, p))


def Binomial(n: int, p: float) -> Arm:
    """Binomial arm: the number of successes in ``n`` trials of probability ``p``."""
    return Arm(binom(n, p), sampler=partial(np.random.binomial, n, p))


def Gaussian(mu: float, var: float = 1) -> Arm:
    """Gaussian arm of mean ``mu`` and variance ``var`` (unbounded rewards)."""
    return Arm(norm(mu, sqrt(var)), sampler=partial(np.random.normal, mu, sqrt(var)))


def Exponential(p: float) -> Arm:
    """Exponential arm of rate ``p``, hence mean ``1 / p``."""
    return Arm(expon(scale=1 / p), sampler=partial(np.random.exponential, 1 / p))


def TruncatedGaussian(mean: float, sigma: float, low: float, high: float) -> Arm:
    """Gaussian truncated to ``[low, high]``.

    Parameters
    ----------
    mean : float
        Mean of the *untruncated* Gaussian.
    sigma : float
        Standard deviation of the untruncated Gaussian.
    low, high : float
        Bounds of the support.

    Returns
    -------
    Arm
        An arm whose ``mean`` is the true *truncated* mean, which differs from
        ``mean`` whenever the bounds are not symmetric around it.
    """
    a = (low - mean) / sigma
    b = (high - mean) / sigma
    return Arm(truncnorm(a, b, loc=mean, scale=sigma))


class TruncatedExponential:
    """Exponential arm clipped at ``trunc``, with mass piling on the boundary.

    Kept custom rather than delegating to :class:`scipy.stats.truncexpon`,
    which *renormalizes* the tail instead of clipping it and therefore
    describes a different distribution.

    Parameters
    ----------
    p : float
        Rate of the underlying exponential.
    trunc : float
        Value at which samples are clipped.

    Attributes
    ----------
    mean : float
        True mean of the clipped variable, :math:`(1 - e^{-p\\,\\mathrm{trunc}})/p`.
    variance : float
        Placeholder, left at ``0``; not used for regret accounting.
    """

    def __init__(self, p: float, trunc: float) -> None:
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p
        self.variance = 0

    def sample(self) -> float:
        """Draw one reward, clipped at ``trunc``.

        Returns
        -------
        float
            An exponential draw, capped above by ``trunc``.
        """
        return min(-(1 / self.p) * log(random()), self.trunc)
