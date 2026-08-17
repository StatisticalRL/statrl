import numpy as np

from gymnasium import Env as Environment
from gymnasium.utils import seeding





class KernelBanditEnv(Environment):
    """
    Finite kernelized bandit environment.

    A latent reward function is sampled from a Gaussian Process

        f ~ GP(0,k)

    over the arm feature vectors.

    Pulling arm i returns

        f(x_i) + noise

    where x_i is the feature vector associated with arm i.

    Usage:
    kernel = lambda x, y: np.exp(-0.5 * np.linalg.norm(x-y)**2)
    arms = np.linspace(0, 1, 100).reshape(-1,1)
    env = KernelBanditEnv(
    arm_features=arms,
    kernel=kernel,
    noise_std=0.1,
    name="RBFKernelBandit")
    """

    def __init__(
        self,
        arm_features,
        kernel,
        noise_std: float = 0.0,
        name: str = "KernelBandit",
        function_seed: int = 1,
    ):

        self.arm_features = np.asarray(arm_features, dtype=float)
        self.kernel = kernel
        self.noise_std = noise_std
        self.name = name

        self.function_seed = function_seed
        self.renderers = []
        self.last = (None, 0.0)

        self._sample_function()

    @property
    def number_arms(self) -> int:
        return len(self.arm_features)

    @property
    def means(self):
        if self._means is None:
            raise RuntimeError(
                "Environment must be reset before accessing means."
            )
        return self._means

    @property
    def optimal_mean(self) -> float:
        return float(np.max(self.means))

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    def _covariance_matrix(self) -> np.ndarray:
        n = self.number_arms

        K = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(
                    self.arm_features[i],
                    self.arm_features[j]
                )

        return K

    def _sample_function(self) -> None:

        # Dedicated RNG for the latent GP realization
        rng = np.random.default_rng(self.function_seed)

        K = self._covariance_matrix()

        K += 1e-10 * np.eye(self.number_arms)

        self._means = rng.multivariate_normal(
            mean=np.zeros(self.number_arms),
            cov=K,
        )

    def reset(self, seed=None, options=None):

        # RNG used for the interaction noise
        self.np_random, self.seed = seeding.np_random(seed)

        self.last = (None, 0.0)

        return 0

    def expected_reward(self, arm: int) -> float:
        return float(self._means[arm])

    def step(self, arm: int)-> float:

        reward = self._means[arm]

        if self.noise_std > 0:
            reward += self.np_random.normal(
                0.0,
                self.noise_std
            )

        self.last = (arm, reward)

        return float(reward)

    def render(self, mode="human"):

        for renderer in self.renderers:
            renderer.render(self, self.last)

    def close(self):

        for renderer in self.renderers:
            renderer.stop(self)