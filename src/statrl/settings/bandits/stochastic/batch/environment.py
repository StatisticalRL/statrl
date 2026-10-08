
from itertools import groupby
from typing import Any, Optional

from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv as MAB
from statrl.settings.bandits.stochastic.anytime.envs.distributions import Arm

class BatchMAB(MAB):
    """Wrap a stochastic bandit into a batched one.

    Turns any
    :class:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv`
    into an environment whose :meth:`step` consumes a whole list of arms and
    returns a list of rewards.  

    A round is a *batch*, not a pull, so the horizon of an experiment counts
    batches.

    Parameters
    ----------
    mab : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
        The underlying bandit supplying the arms.
    batchsize : callable or sequence of int
        Batch schedule: either ``batchsize(round) -> int``, or a sequence
        indexed by round which falls back to ``1`` once exhausted. A plain
        sequence is picklable, and so survives the process pool used by
        :func:`~statrl.experiments.parallelruns.multicoreRuns`; a lambda does
        not.

    Attributes
    ----------
    round : int
        Index of the next batch, counted from 0 since the last :meth:`reset`.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> env = BatchMAB(BernoulliBandit([0.2, 0.9]), batchsize=[2, 4, 8])
    >>> info = env.reset()
    >>> info["nextbatchsize"]
    2
    >>> rewards, info = env.step([0, 1])
    >>> len(rewards), info["nextbatchsize"]
    (2, 4)
    """

    def __init__(self, mab: MAB, batchsize: Any) -> None:
        self.mab = mab
        # Accept a plain list (picklable for multiprocessing) or a callable.
        if callable(batchsize):
            self.batchsize = batchsize
        else:
            _sizes = list(batchsize)
            self.batchsize = lambda ell: _sizes[ell] if ell < len(_sizes) else 1
        self.name = "B"+self.mab.name+"-batch-"+str(self.batchsize(0))+"-"+str(self.batchsize(1))+"-"+str(self.batchsize(2))
        self.round = 0
        super(BatchMAB, self).__init__(self.mab.rewarddistributions, name=self.name)
        # A batch's "last" is a (arms, rewards) pair of lists, unlike the single
        # (arm, reward) pair used by the unbatched MAB this class wraps.
        self.last: tuple[Optional[list], Optional[list]] = (None, None)  # type: ignore[assignment]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> dict:  # type: ignore[override]
        """Start a new run and announce the size of the first batch.

        Parameters
        ----------
        seed : int, optional
            Seed for the underlying bandit's generator.
        options : dict, optional
            Unused; accepted for :class:`gymnasium.Env` compatibility.

        Returns
        -------
        dict
            ``{"nextbatchsize": int, "mean": 0}``. Unlike the unbatched
            :meth:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv.reset`,
            an info dict is returned rather than a dummy observation, because
            the agent cannot act without knowing the batch size.
        """
        super().reset(seed=seed, options=options)  # MAB.reset returns an unused dummy observation
        self.round = 0
        info = {"nextbatchsize": self.batchsize(self.round), "mean":0}
        self.last = (None,None)
        return info

    def step(self, action: list) -> tuple:  # type: ignore[override]
        """Pull every arm of one batch and return all their rewards.

        Parameters
        ----------
        action : list of int
            Arms to pull, one per slot of the current batch. Its length must
            equal the ``nextbatchsize`` announced by the previous call.

        Returns
        -------
        batchreward : list of float
            Reward of each pull, in the order the arms were given.
        info : dict
            ``{"nextbatchsize": int, "mean": float}``, where ``mean`` is the
            *sum* of the true means of the arms pulled — the batch's expected
            score, which the interaction loop accumulates for regret.

        Raises
        ------
        AssertionError
            If ``action`` does not have exactly ``nextbatchsize`` entries.
        """
        B= self.batchsize(self.round)
        assert len(action)==B
        batchreward: list[float] = []
        batchmean = []
        for arm, group in groupby(action):
            distribution = self.mab.rewarddistributions[arm]
            run_size = sum(1 for _ in group)
            if isinstance(distribution, Arm):
                batchreward.extend(distribution.sample_many(run_size))
            else:
                batchreward.extend(self.mab.step(arm) for _ in range(run_size))
            batchmean.extend([distribution.mean] * run_size)

        self.last = (action, batchreward)
        if batchreward:
            self.mab.last = (action[-1], batchreward[-1])
        self.round=self.round+1
        info = {"nextbatchsize": self.batchsize(self.round), "mean": sum(batchmean)}
        return (batchreward,info)
