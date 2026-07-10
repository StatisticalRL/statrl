import numpy as np
from math import log


## A function that returns an argmax at random in case of multiple maximizers

def randmax(A: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(A == A.max())))


## A function that returns an argmin at random in case of multiple minimizers

def randmin(A: np.ndarray) -> int:
    return int(np.random.choice(np.flatnonzero(A == A.min())))


def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return (max_, all_)


## Kullback-Leibler divergence in exponential families

eps = 1e-15

def klBern(x: float, y: float) -> float:
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klGauss(x: float, y: float, sig2: float = 1.) -> float:
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x - y) * (x - y) / (2 * sig2)


def klPoisson(x: float, y: float) -> float:
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klExp(x: float, y: float) -> float:
    """Kullback-Leibler divergence for Exponential distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return (x / y - 1 - log(x / y))


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()




class Dirac:
    def __init__(self, value):
        self.v = value

    def rvs(self):
        return self.v

    def mean(self):
        return self.v