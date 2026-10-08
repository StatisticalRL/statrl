"""Numerical helpers shared by every settings"""

from typing import Union, overload

import numpy as np
from math import log
from scipy.optimize import minimize_scalar, root_scalar


def randmax(A: np.ndarray) -> int:
    """Index of a maximum of ``A``, breaking ties uniformly at random.

    Parameters
    ----------
    A : ndarray
        Array of values to maximize over.

    Returns
    -------
    int
        Index of a maximal entry. When several entries attain the maximum,
        one of them is drawn uniformly at random.

    See Also
    --------
    randmin : The minimizing counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> randmax(np.array([0.1, 0.7, 0.3]))
    1
    """
    return int(np.random.choice(np.flatnonzero(A == A.max())))


def randmin(A: np.ndarray) -> int:
    """Index of a minimum of ``A``, breaking ties uniformly at random.

    Parameters
    ----------
    A : ndarray
        Array of values to minimize over.

    Returns
    -------
    int
        Index of a minimal entry. When several entries attain the minimum,
        one of them is drawn uniformly at random.

    See Also
    --------
    randmax : The maximizing counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> randmin(np.array([0.4, 0.2, 0.9]))
    1
    """
    return int(np.random.choice(np.flatnonzero(A == A.min())))


def allmax(a):
    """Maximum of ``a`` together with *every* index attaining it.

    Unlike :func:`randmax`, no tie is broken: all maximizers are returned,
    which lets a caller build a uniform policy over optimal actions.

    Parameters
    ----------
    a : sequence of float
        Values to maximize over.

    Returns
    -------
    tuple of (float, list of int) or list
        ``(max_value, indices)`` where ``indices`` lists every position
        attaining ``max_value``. An empty input returns an empty list.

    Examples
    --------
    >>> allmax([0.2, 0.5, 0.5, 0.1])
    (0.5, [1, 2])
    """
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

@overload
def klBern(x: float, y: float) -> float: ...


@overload
def klBern(x: np.ndarray, y: Union[float, np.ndarray]) -> np.ndarray: ...


@overload
def klBern(x: float, y: np.ndarray) -> np.ndarray: ...


def klBern(
    x: Union[float, np.ndarray], y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Kullback-Leibler divergence between two Bernoulli distributions.

    .. math::

       \\mathrm{kl}(x, y) = x \\log\\frac{x}{y} + (1-x)\\log\\frac{1-x}{1-y}

    Parameters
    ----------
    x, y : float
        Means in :math:`[0, 1]`. Both are clipped into
        :math:`[\\varepsilon, 1-\\varepsilon]` with ``eps = 1e-15`` so the
        divergence stays finite at the boundary.

    Returns
    -------
    float
        The divergence.

    See Also
    --------
    klGauss, klPoisson, klExp

    Examples
    --------
    >>> round(klBern(0.5, 0.5), 12)
    0.0
    >>> klBern(0.5, 0.9) > 0
    True
    """
    x = np.clip(x, eps, 1 - eps)
    y = np.clip(y, eps, 1 - eps)
    divergence = x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
    return float(divergence) if np.ndim(divergence) == 0 else divergence


def klGauss(x: float, y: float, sig2: float = 1.) -> float:
    """Kullback-Leibler divergence between two Gaussians of equal variance.

    .. math:: \\mathrm{kl}(x, y) = \\frac{(x - y)^2}{2\\sigma^2} 

    Parameters
    ----------
    x, y : float
        Means of the two distributions.
    sig2 : float, default=1.0
        Common variance :math:`\\sigma^2`.

    Returns
    -------
    float
        The divergence, in nats.

    See Also
    --------
    klBern, klPoisson, klExp

    Examples
    --------
    >>> klGauss(0.0, 1.0)
    0.5
    """
    return (x - y) * (x - y) / (2 * sig2)


def klPoisson(x: float, y: float) -> float:
    """Kullback-Leibler divergence between two Poisson distributions.

    .. math:: \\mathrm{kl}(x, y) = y - x + x\\log\\frac{x}{y}

    Parameters
    ----------
    x, y : float
        Rates of the two distributions, clipped below at ``eps = 1e-15``.

    Returns
    -------
    float
        The divergence, in nats.

    See Also
    --------
    klBern, klGauss, klExp

    Examples
    --------
    >>> round(klPoisson(2.0, 2.0), 12)
    0.0
    """
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klExp(x: float, y: float) -> float:
    """Kullback-Leibler divergence between two exponential distributions.

    .. math:: \\mathrm{kl}(x, y) = \\frac{x}{y} - 1 - \\log\\frac{x}{y}

    Parameters
    ----------
    x, y : float
        Means of the two distributions, clipped below at ``eps = 1e-15``.

    Returns
    -------
    float
        The divergence, in nats.

    See Also
    --------
    klBern, klGauss, klPoisson

    Examples
    --------
    >>> round(klExp(1.0, 1.0), 12)
    0.0
    """
    x = max(x, eps)
    y = max(y, eps)
    return (x / y - 1 - log(x / y))


def categorical_sample(prob_n, np_random):
    """Draw one index from a categorical distribution.

    Parameters
    ----------
    prob_n : array-like of float
        Probability vector; must sum to one.
    np_random : numpy.random.Generator
        Generator supplying the uniform draw, so that a seeded environment
        stays reproducible.

    Returns
    -------
    int
        The sampled index.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()




class Dirac:
    """Deterministic distribution concentrated on a single value.
 
    Parameters
    ----------
    value : float
        The value carrying all the mass.

    Examples
    --------
    >>> d = Dirac(0.7)
    >>> d.rvs(), d.mean()
    (0.7, 0.7)
    """

    def __init__(self, value):
        self.v = value

    def rvs(self):
        """Draw a sample — always ``value``.

        Returns
        -------
        float
            The constant this distribution is concentrated on.
        """
        return self.v

    def mean(self):
        """Mean of the distribution — always ``value``.

        Returns
        -------
        float
            The constant this distribution is concentrated on.
        """
        return self.v




def KLinf_threshold(reward_history, mean_threshold, upper_bound=1.0, custom_optim=True):
    """Non-parametric divergence :math:`K_{\\inf}` between an empirical
    distribution and the set of distributions with mean above a threshold, evaluated using the concave dual.

    Parameters
    ----------
    reward_history : array-like of float
        Rewards observed so far for one arm; they define the empirical
        measure :math:`\\hat{F}`.
    mean_threshold : float
        Threshold mean :math:`\\mu^*`, in practice the best empirical mean
        across arms.
    upper_bound : float, default=1.0
        Known upper bound ``B`` of the reward support.
    custom_optim : bool, default=True
        If True, first test whether the dual maximum is attained at a boundary
        (the common case, roughly a 2x speed-up) and otherwise solve
        ``jac = 0`` with Brent's method. On non-convergence, and when False,
        fall back to :func:`scipy.optimize.minimize_scalar`.

    Returns
    -------
    float
        The value of :math:`K_{\\inf}`, or ``inf`` if the optimizer failed —
        which makes the arm ineligible for this round.

    See Also
    --------
    klBern, klGauss : Parametric alternatives, used when the reward family is known.     

    Examples
    --------
    >>> bool(KLinf_threshold([0.5, 0.5, 0.5], 0.5) < 1e-6)
    True
    """
    # Kinf is computed via its concave dual problem
    #         max_{0<=lambda<=1/(B-mu^*)} E[log(1-(X-mu^*)*lambda)],
    #         where E is taken w.r.t the empirical measure hat{F}_k(t).
    X = np.array(reward_history)
    # Faster optimization: many times, the maximum of the concave
    # dual objective is attained on the boundary 0 or 1/(B-mu).
    # ~x2 speedup on some bandit instances.
    # If problem, fall back to standard minimize_scalar.

    # Pb when X>= upper_bound
    l_plus = 1e12 if mean_threshold == upper_bound else 1 / (upper_bound - mean_threshold)
    l_plus -= 1e-12  # To avoid reaching upper_bound?

    fallback = False
    if custom_optim:
        def f(lam):
            return np.mean(np.log(1 - (X - mean_threshold) * lam))

        def jac(lam):
            return -np.mean((X - mean_threshold) / (1 - (X - mean_threshold) * lam))

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
        def f(lam):
            return -np.mean(np.log(1 - (X - mean_threshold) * lam))

        ret = minimize_scalar(
            f, method='bounded', bounds=(0, l_plus)
        )
        if ret.success:
            kinf = -ret.fun
        else:
            # if error, just make this arm not eligible this turn
            kinf = np.inf
    return kinf
