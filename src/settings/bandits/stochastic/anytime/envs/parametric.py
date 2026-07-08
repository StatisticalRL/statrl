


from settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
import settings.bandits.stochastic.anytime.envs.distributions as distributions

## some functions that create specific MABs

def BernoulliBandit(means, name="MAB-Bernoulli"):
    """define a Bernoulli MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Bernoulli(p) for p in means], name=name)

def BinomialBandit(means, repetitions=200, name="MAB-Binomial"):
    """define a Bernoulli MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}{repetitions}-means-{s}'
    return StochasticBanditEnv([distributions.Binomial(repetitions, p) for p in means], name=name)

def GaussianBandit(means, vars, name="MAB-Gaussian"):
    """define a Gaussian MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}-means-{s}'
    return StochasticBanditEnv([distributions.Gaussian(m, v) for m,v in zip(means, vars)], name=name)

def TruncatedGaussianBandit(means, sigma=0.5, low=-1.0, high=1.0, name="MAB-TruncGaussian"):
    s = "-".join(str(m) for m in means)
    name = f'{name}-sigma{sigma}-means-{s}'
    return StochasticBanditEnv([distributions.TruncatedGaussian(p, sigma=sigma, low=low, high=high) for p in means], name=name)


import numpy as np

def RandomBernoulliBandit(Delta, K, name="MAB-RandomBernoulli"):
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand() * (1. - Delta)
    secondmaxMean = maxMean - Delta
    means = secondmaxMean * np.random.random(K)
    bestarm = np.random.randint(0, K)
    secondbestarm = np.random.randint(0, K)
    while (secondbestarm == bestarm):
        secondbestarm = np.random.randint(0, K)
    means[bestarm] = maxMean
    means[secondbestarm] = secondmaxMean
    return BernoulliBandit(means, name=name)
