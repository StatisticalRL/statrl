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




import numpy as np


from scipy.optimize import minimize_scalar, root_scalar
def KLinf_threshold(reward_history, mean_threshold,upper_bound=1.0, custom_optim=True):
    # Kinf is caculated via its concave dual problem
    #         max_{0<=lambda<=1/(B-mu^*)} E[log(1-(X-mu^*)*lambda)],
    #         where E is taken w.r.t the empirical measure hat{F}_k(t).
    X = np.array(reward_history)
    # Faster optimization: many times, the maximum of the concave
    # dual objective is attained on the boundary 0 or 1/(B-mu).
    # ~x2 speedup on some bandit instances.
    # If problem, fall back to standard minimize_scalar.
    fallback = False
    if custom_optim:
        def f(l):
            return np.mean(np.log(1 - (X - mean_threshold) * l))

        def jac(l):
            return -np.mean((X - mean_threshold) / (1 - (X - mean_threshold) * l))

        l_plus = 1e12 if mean_threshold == upper_bound else 1 / (upper_bound - mean_threshold)

        if jac(0) * jac(l_plus) >= 0:
            kinf = np.maximum(f(0), f(l_plus))
        else:
            ret = root_scalar(
                jac, method='brentq', bracket=[0, l_plus]
            )
            if ret.converged:
                kinf = np.max([f(ret.root), f(0), f(l_plus)])
            else:
                fallback = True
    if not custom_optim or fallback:
        # minimize -E[log(1-(X-mu^*)*lambda)]
        def f(l):
            return -np.mean(np.log(1 - (X - mean_threshold) * l))

        ret = minimize_scalar(
            f, method='bounded', bounds=(0, 1 / (upper_bound - mean_threshold))
        )
        if ret.success:
            kinf = -ret.fun
        else:
            # if error, just make this arm not eligible this turn
            kinf = np.inf
    return kinf