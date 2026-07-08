"""different classes of arms, all of them have a sample() method which produce rewards"""

import numpy as np
from random import random
from math import sqrt, log, exp
from scipy.stats import truncnorm


class Bernoulli:

    def __init__(self, p):
        # create a Bernoulli arm with mean p
        self.mean = p
        self.variance = p * (1 - p)

    def sample(self):
        # generate a reward from a Bernoulli arm
        return float(random() < self.mean)


class Binomial:

    def __init__(self, n, p):
        # create a Bernoulli arm with mean p
        self.mean = p*n
        self.n = n
        self.p=p

    def sample(self):
        # generate a reward from a Bernoulli arm
        return np.random.binomial(self.n,self.p)

class Gaussian:

    def __init__(self, mu, var=1):
        # create a Gaussian arm with specified mean and variance
        self.mean = mu
        self.variance = var

    def sample(self):
        # generate a reward from a Gaussian arm
        return self.mean + sqrt(self.variance) * np.random.normal()


class Exponential:

    def __init__(self, p):
        # create an Exponential arm with parameter p
        self.mean = 1 / p
        self.variance = 1 / (p * p)

    def sample(self):
        # generate a reward from an Exponential arm
        return -(self.mean) * log(random())


class TruncatedExponential:

    def __init__(self, p, trunc):
        # create a truncated Exponential arm with parameter p
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p
        self.variance = 0

    def sample(self):
        # generate a reward from an Exponential arm
        return min(-(1 / self.p) * log(random()), self.trunc)





class TruncatedGaussian:
    """
    Gaussian distribution truncated to [low, high].

    Parameters
    ----------
    mean : float
        Mean of the *untruncated* Gaussian (location parameter).
        The true mean of the truncated distribution is computed and stored
        in self.mean.
    sigma : float
        Standard deviation of the underlying Gaussian.
    low, high : float
        Truncation bounds (default [-1, 1]).

    Notes
    -----
    Use this distribution with KLinf_threshold (upper_bound = high) since the
    support is bounded.  For the standard [-1, 1] setup, set sigma ~ 0.5 so
    that the effective mass of the Gaussian stays well within the bounds.
    """

    def __init__(self, mean, sigma=0.5, low=-1.0, high=1.0):
        self.low = low
        self.high = high
        self.sigma = sigma
        a = (low - mean) / sigma
        b = (high - mean) / sigma
        self._dist = truncnorm(a, b, loc=mean, scale=sigma)
        # true mean of the truncated distribution
        self.mean = float(self._dist.mean())

    def sample(self):
        return float(self._dist.rvs())

#
# class TruncatedGaussian:
#     """
#     Truncated Gaussian distribution.
#
#     Samples from a Gaussian distribution N(mu, sigma²) restricted to the
#     interval [low, high].
#
#     Parameters
#     ----------
#     mu : float
#         Mean of the underlying Gaussian distribution.
#     sigma : float, default=1.0
#         Standard deviation of the underlying Gaussian.
#     low : float, default=-np.inf
#         Lower truncation bound.
#     high : float, default=np.inf
#         Upper truncation bound.
#     """
#
#     def __init__(self, mu, sigma=1.0, low=-np.inf, high=np.inf):
#         if sigma <= 0:
#             raise ValueError("sigma must be strictly positive.")
#         if low >= high:
#             raise ValueError("low must be smaller than high.")
#
#         self.mu = mu
#         self.sigma = sigma
#         self.low = low
#         self.high = high
#
#         # Standardized truncation limits
#         alpha = (low - mu) / sigma
#         beta = (high - mu) / sigma
#
#         # Standard normal PDF
#         phi = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
#
#         # Standard normal CDF
#         Phi = lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
#
#         Z = Phi(beta) - Phi(alpha)
#
#         # Numerical safeguard
#         if Z <= 0:
#             raise ValueError("The truncation interval has negligible probability.")
#
#         # Mean of the truncated Gaussian
#         self.mean = mu + sigma * (phi(alpha) - phi(beta)) / Z
#
#         # Variance of the truncated Gaussian
#         self.variance = sigma ** 2 * (
#             1
#             + (alpha * phi(alpha) - beta * phi(beta)) / Z
#             - ((phi(alpha) - phi(beta)) / Z) ** 2
#         )
#
#     def sample(self):
#         """
#         Generate one sample using rejection sampling.
#         """
#         while True:
#             x = np.random.normal(self.mu, self.sigma)
#             if self.low <= x <= self.high:
#                 return x