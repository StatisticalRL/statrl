"""arms: each exposes .mean and a sample() method producing rewards"""

from math import sqrt, log, exp
from random import random

from scipy.stats import bernoulli, binom, norm, expon, truncnorm


class Arm:
    """Adapter over a frozen scipy.stats distribution."""

    def __init__(self, dist) -> None:
        self._dist = dist
        self.mean = dist.mean()

    def sample(self) -> float:
        return self._dist.rvs()


def Bernoulli(p: float) -> Arm:          return Arm(bernoulli(p))
def Binomial(n: int, p: float) -> Arm:   return Arm(binom(n, p))
def Gaussian(mu: float, var: float = 1) -> Arm:   return Arm(norm(mu, sqrt(var)))
def Exponential(p: float) -> Arm:        return Arm(expon(scale=1 / p))


def TruncatedGaussian(mean: float, sigma: float, low: float, high: float) -> Arm:
    """Gaussian truncated to [low, high]; .mean is the true truncated mean."""
    a = (low - mean) / sigma,
    b = (high - mean) / sigma
    return Arm(truncnorm(a, b, loc=mean, scale=sigma))


class TruncatedExponential:
    """Exponential clipped at `trunc` (mass piles on the boundary).

    Kept custom because scipy's truncexpon renormalizes instead of clipping.
    """

    def __init__(self, p: float, trunc: float) -> None:
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p
        self.variance = 0

    def sample(self) -> float:
        return min(-(1 / self.p) * log(random()), self.trunc)
