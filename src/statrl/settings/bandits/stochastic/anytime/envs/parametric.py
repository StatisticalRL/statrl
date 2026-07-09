


import numpy as np
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
import statrl.settings.bandits.stochastic.anytime.envs.distributions as distributions

## some functions that create specific MABs

def BernoulliBandit(means: np.ndarray, name: str = "MAB-Bernoulli") -> StochasticBanditEnv:
    """define a Bernoulli MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Bernoulli(p) for p in means], name=name)

def BinomialBandit(means: np.ndarray, repetitions: int = 200, name: str = "MAB-Binomial") -> StochasticBanditEnv:
    """define a Binomial MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}{repetitions}-means-{s}'
    return StochasticBanditEnv([distributions.Binomial(repetitions, p) for p in means], name=name)

def GaussianBandit(means: np.ndarray, vars: np.ndarray, name: str = "MAB-Gaussian") -> StochasticBanditEnv:
    """define a Gaussian MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Gaussian(m, v) for m,v in zip(means, vars)], name=name)

def TruncatedGaussianBandit(means: np.ndarray, sigma: float = 0.5, low: float = -1.0, high: float = 1.0, name: str = "MAB-TruncGaussian") -> StochasticBanditEnv:
    s = "-".join(str(m) for m in means)
    name = f'{name}-means-{s}-sigma{sigma}'
    return StochasticBanditEnv([distributions.TruncatedGaussian(p, sigma=sigma, low=low, high=high) for p in means], name=name)


def RandomBernoulliBandit(Delta: float, K: int, name: str = "MAB-RandomBernoulli") -> StochasticBanditEnv:
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand() * (1. - Delta)
    secondmaxMean = maxMean - Delta
    means = secondmaxMean * np.random.random(K)
    bestarm, secondbestarm = np.random.choice(K, 2, replace=False)
    means[bestarm] = maxMean
    means[secondbestarm] = secondmaxMean
    return BernoulliBandit(means, name=name)
