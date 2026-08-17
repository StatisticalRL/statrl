import numpy as np

from statrl.settings.bandits.batch.environment import BatchMAB

from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit, BinomialBandit, GaussianBandit, TruncatedGaussianBandit


import math

# Batch-size schedules — ell is the round index (0-based)
_B_CONST     = lambda ell: 10
_B_LINEAR    = lambda ell: int(ell + 1)
_B_QUADRATIC = lambda ell: int((ell + 1) ** 2)
_B_CUBIC     = lambda ell: int((ell + 1) ** 3)
_B_EXP       = lambda ell: int(2 ** ell)
_B_SEXP      =  lambda ell: int(math.exp(ell**1.5))
_B_DOUBLE_EXP = lambda ell: int(math.exp(2 ** ell))
_B_ABRUPT = lambda ell: 100 if (ell % 4==1) else int((ell+1)**3)

def exotic_schedule(t):
    schedule= {0:_B_LINEAR, 1: _B_EXP, 2:_B_CONST, 3:_B_CUBIC}
    return schedule[(t % 4)](t)

_B_EXOTIC = exotic_schedule
#_B_EXOTIC2 = exotic_schedule2

from statrl.settings.bandits.batch.agents.baba_schedule import compute_baba_grid
def baba_schedule(horizon, nbArms):
    batch_sizes, _, _, _ = compute_baba_grid(horizon, nbArms, None, alpha=3)
    return batch_sizes


schedule_catalogue= {"constant": _B_CONST, "linear": _B_LINEAR, "quadratic": _B_QUADRATIC, "cubic":_B_CUBIC,
            "exp": _B_EXP, "surexp": _B_SEXP,"doubleexp":_B_DOUBLE_EXP, "abrupt":_B_ABRUPT, "exotic":_B_EXOTIC
                     }

mean_catalogue = {"simple4": [0.1, 0.4, 0.7, 0.9],
                  "simple6": [0.2, 0.6, 0.8, 0.8, 0.95, 0.9]
                  }

class BatchGaussianBandit(BatchMAB):
    def __init__(self, means, vars, batchschedule="constant", name="BMAB-Gaussian"):
        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchGaussianBandit, self).__init__(
                GaussianBandit(means= mean_catalogue[means], vars=vars), schedule)
        else:
            super(BatchGaussianBandit, self).__init__(
            GaussianBandit(means=means, vars=vars),schedule)


class BatchBernoulliBandit(BatchMAB):
    def __init__(self, means, batchschedule="constant", name="BMAB-Bernoulli"):
        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchBernoulliBandit, self).__init__(
                BernoulliBandit(means=mean_catalogue[means]), schedule)
        else:
            super(BatchBernoulliBandit, self).__init__(
            BernoulliBandit(means=means),schedule)


class BatchTruncatedGaussianBandit(BatchMAB):
    def __init__(self, means, sigma: float = 0.5, low: float = -1.0, high: float = 1.0, batchschedule="constant", name="BMAB-TGaussian"):

        if (type(batchschedule) is str):
            if ("," in batchschedule):
                fct, horiz = batchschedule.split(",")
                horizon = int(horiz)
                schedule = baba_schedule(horizon, len(means))
            else:
                schedule = schedule_catalogue[batchschedule]
        if (type(means) is str):
            super(BatchTruncatedGaussianBandit, self).__init__(
                TruncatedGaussianBandit(means=mean_catalogue[means],sigma= sigma, low=low, high=high), schedule)
        else:
            super(BatchTruncatedGaussianBandit, self).__init__(
            TruncatedGaussianBandit(means=means, sigma= sigma, low=low, high=high),schedule)