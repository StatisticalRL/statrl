import numpy as np
from math import log


## A function that returns an argmax at random in case of multiple maximizers

def randmax(A):
    maxValue = max(A)
    index = [i for i in range(len(A)) if A[i] == maxValue]
    return np.random.choice(index)


## A function that returns an argmin at random in case of multiple maximizers

def randmin(A):
    minValue = min(A)
    index = [i for i in range(len(A)) if A[i] == minValue]
    return np.random.choice(index)


## Kullback-Leibler divergence in exponential families

eps = 1e-15

def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klGauss(x, y, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x - y) * (x - y) / (2 * sig2)


def klPoisson(x, y):
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klExp(x, y):
    """Kullback-Leibler divergence for Exponential distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return (x / y - 1 - log(x / y))