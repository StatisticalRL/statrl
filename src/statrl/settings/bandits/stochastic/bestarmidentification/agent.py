from abc import ABC, abstractmethod

from gymnasium.utils import seeding

class BAIBanditAgent(ABC):
    """Base class for Best-arm-identification stochastic bandit agents.

    Subclasses implement the protocol shared by every agent in
    this setting: :meth:`reset` to start an independent run,
    :meth:`select_arm` to choose an arm, and :meth:`update` to learn from the
    observed reward.

    Parameters
    ----------
    name : str
        Label used in logfiles and plot legends. Must be unique within an
        experiment, since :func:`~statrl.experiments.utils.dump` builds
        filenames from it.
    seed : int, default=1
        Seed for the agent's own randomness. :meth:`reset` re-derives
        ``np_random`` from it, so replicates are reproducible.

    Attributes
    ----------
    np_random : numpy.random.Generator
        Agent-local generator, available after the first :meth:`reset`.

    See Also
    --------
    statrl.settings.bandits.stochastic.knownhorizon.agent.BanditAgent :
        The counterpart whose ``reset`` receives the horizon.

    Notes
    -----
    Agents are deep-copied once per replicate by
    :func:`~statrl.experiments.parallelruns.multicoreRuns`, so an agent may
    hold arbitrary state as long as :meth:`reset` fully reinitializes it.
    """

    def __init__(self, name: str, seed: int = 1) -> None:
        self.name = name
        self.seed =seed


    def reset(self) -> None:
        """Start a new independent run.

        Reseeds ``np_random`` and clears any statistics accumulated by a
        previous run.
        """
        self.np_random, self.seed = seeding.np_random(self.seed)

    @abstractmethod
    def select_arm(self) -> int:
        """Choose the arm to pull next.

        Returns
        -------
        int
            Index of the selected arm, in ``range(env.number_arms)``.
        """
        ...

    @abstractmethod
    def recommend_arm(self)-> int:
        """Choose the arm recommender as best arm"
        Returns
        -------
        int
            Index of the recommended arm, in ``range(env.number_arms)``.
        """
        ...

    @abstractmethod
    def is_done(self)  -> bool:
        """
        Decide if enough samples are gathered for final decision
        :return:
        """
        ...


    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Learn from the reward observed for the arm just pulled.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Reward sampled by
            :meth:`~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv.step`.
            Only this realized reward is observed — never the arm's mean.
        """
