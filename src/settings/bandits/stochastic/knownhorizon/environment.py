from abc import ABC, abstractmethod


class StochasticBanditEnv(ABC):
    """
    Abstract stochastic multi-armed bandit environment.

    Each arm corresponds to an independent reward distribution.
    Pulling an arm produces an independent sample from its
    associated distribution.

    The environment is stateless: interactions do not modify
    the underlying distributions.
    """

    @property
    @abstractmethod
    def n_arms(self) -> int:
        """Number of available arms."""

    @property
    @abstractmethod
    def means(self):
        """
        Mean reward of every arm.

        Intended only for evaluation and oracle construction.
        """

    @property
    def optimal_mean(self):
        return max(self.means)

    @property
    def optimal_arm(self):
        return int(np.argmax(self.means))

    @abstractmethod
    def pull(self, arm: int):
        """
        Sample one reward from the specified arm.
        """

    def expected_reward(self, arm):
        return self.means[arm]

    def reset(self, seed=None):
        """
        Reset the random generator.
        """
