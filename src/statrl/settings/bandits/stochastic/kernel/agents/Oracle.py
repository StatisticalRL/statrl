

from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class Oracle(KernelBanditAgent):
    """Baseline that always plays the best arm.

    The oracle knows the latent function values and so incurs no regret.

    Parameters
    ----------
    env : ~statrl.settings.bandits.stochastic.kernel.environment.KernelBanditEnv
        The environment.

    See Also
    --------
    statrl.settings.bandits.stochastic.kernel.agents._Random.Random :
        The opposite baseline, which never exploits.
    """

    def __init__(self, env: KernelBanditEnv) -> None:
        self.env = env
        KernelBanditAgent.__init__(
            self,
            name="Oracle",
            arms=env.arm_features,
            kernel=env.kernel,
            noise_std=env.noise_std,
        )

    @property
    def policy(self) -> list[int]:
        """list of int: The optimal arm, as a one-element list."""
        return [self.env.optimal_arm]

    def reset(self) -> None:
        """Start a new run. The oracle keeps no statistics."""
        pass

    def select_arm(self) -> int:
        """Play the arm with the highest mean.

        Returns
        -------
        int
            ``env.optimal_arm``.
        """
        return self.env.optimal_arm

    def update(self, action: int, reward: float) -> None:
        """Ignore the observed reward (the oracle has nothing to learn).

        Parameters
        ----------
        action : int
            Index of the arm that was pulled.
        reward : float
            Observed reward.
        """
        pass