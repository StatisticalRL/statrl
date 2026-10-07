from __future__ import annotations

from abc import ABC

import numpy as np

from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class _KernelTSBase(KernelBanditAgent, ABC):
    """
    Base implementation for Kernel Thompson Sampling.

    The kernel regression state is maintained incrementally.

    For a fixed regularization parameter lambda,

        A_t = K_t + lambda I,

    and the inverse A_t^{-1} is updated by a rank-one/block
    inverse formula whenever a new observation is received.
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        C: float = 5.0,
        delta: float = 0.1 / 12,
        name: str = "Kernel TS",
        seed: int = 1,
    ):
        super().__init__(
            arms=env.arm_features,
            kernel=env.kernel,
            noise_std=env.noise_std,
            name=name,
            seed=seed,
        )

        self.env = env
        self.C = float(C)
        self.delta = float(delta)

        self.arms = np.asarray(env.arm_features, dtype=float)
        self.number_arms = self.arms.shape[0]

        self.observed_arms: list[int] = []
        self.rewards: list[float] = []

        self.lambda_: float | None = None

        # Inverse of K_t + lambda I.
        self._A_inv: np.ndarray | None = None

        # Current regularized regression solution:
        #
        # alpha_t = (K_t + lambda I)^{-1} Y_t
        self._alpha: np.ndarray | None = None

        # log determinant of I + K_t / lambda.
        self._logdet: float = 0.0

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------

    def _kernel_vector(self, arm: int) -> np.ndarray:
        """
        Kernel vector between an arm and all observed arms.
        """
        if not self.observed_arms:
            return np.empty(0, dtype=float)

        x = self.arms[arm]

        return np.asarray(
            [
                self.kernel(
                    x,
                    self.arms[i],
                )
                for i in self.observed_arms
            ],
            dtype=float,
        )

    def _kernel_vector_to_point(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Kernel vector between arbitrary point x and observations.
        """
        if not self.observed_arms:
            return np.empty(0, dtype=float)

        return np.asarray(
            [
                self.kernel(
                    x,
                    self.arms[i],
                )
                for i in self.observed_arms
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Incremental inverse update
    # ------------------------------------------------------------------

    def _initialize_regression(self) -> None:
        """
        Initialize the regression state for the current lambda.
        """
        assert self.lambda_ is not None

        self._A_inv = np.empty((0, 0), dtype=float)
        self._alpha = np.empty(0, dtype=float)
        self._logdet = 0.0

    def _add_observation_fixed_lambda(
        self,
        arm: int,
        reward: float,
    ) -> None:
        """
        Add one observation using a block inverse update.

        This assumes lambda has not changed since the previous
        observation.
        """
        assert self.lambda_ is not None

        lam = self.lambda_

        x = self.arms[arm]

        # First observation.
        if self._A_inv is None or self._A_inv.size == 0:
            diagonal = self.kernel(x, x) + lam

            self._A_inv = np.array(
                [[1.0 / diagonal]],
                dtype=float,
            )

            self._alpha = np.array(
                [reward / diagonal],
                dtype=float,
            )

            self._logdet = np.log1p(
                self.kernel(x, x) / lam
            )

            return

        assert self._alpha is not None

        A_inv = self._A_inv

        k = self._kernel_vector(arm)

        # q = A_{t-1}^{-1} k
        q = A_inv @ k

        # Schur complement
        s = (
            self.kernel(x, x)
            + lam
            - np.dot(k, q)
        )

        # Numerical protection.
        s = max(float(s), 1e-12)

        n = A_inv.shape[0]

        new_A_inv = np.empty(
            (n + 1, n + 1),
            dtype=float,
        )

        new_A_inv[:n, :n] = (
            A_inv
            + np.outer(q, q) / s
        )

        new_A_inv[:n, n] = -q / s
        new_A_inv[n, :n] = -q / s
        new_A_inv[n, n] = 1.0 / s

        self._A_inv = new_A_inv

        # Rather than multiplying the new inverse by all rewards,
        # update alpha using the same block structure.
        #
        # alpha_new =
        #
        # [ A^{-1} y + q * residual / s ]
        # [ residual / s ]
        #
        residual = (
            reward
            - np.dot(k, self._alpha)
        )

        new_alpha = np.empty(
            n + 1,
            dtype=float,
        )

        new_alpha[:n] = (
            self._alpha
            + q * residual / s
        )

        new_alpha[n] = residual / s

        self._alpha = new_alpha

        # Matrix determinant lemma.
        #
        # det(A_new)
        #   = det(A_old) * s
        #
        # But information gain uses
        #
        # det(I + K/lambda)
        #   = det(K + lambda I) / lambda^t.
        #
        # Therefore the incremental contribution is
        #
        # log(s / lambda).
        self._logdet += np.log(s / lam)

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    def _posterior_all_arms(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior mean and regularized kernel covariance
        over all finite arms using the maintained inverse.

        Returns
        -------
        mean:
            Posterior mean for all arms.

        covariance:
            Regularized kernel covariance matrix
            K_lambda(x,x').
        """
        assert self.lambda_ is not None

        if not self.observed_arms:
            covariance = np.asarray(
                [
                    [
                        self.kernel(x, y)
                        for y in self.arms
                    ]
                    for x in self.arms
                ],
                dtype=float,
            )

            return (
                np.zeros(self.number_arms),
                covariance,
            )

        assert self._A_inv is not None
        assert self._alpha is not None

        # K_XA : K x t
        K_XA = np.asarray(
            [
                self._kernel_vector_to_point(x)
                for x in self.arms
            ],
            dtype=float,
        )

        mean = K_XA @ self._alpha

        # K_XA A^{-1} K_AX
        projection = (
            K_XA
            @ self._A_inv
            @ K_XA.T
        )

        K_AA = np.asarray(
            [
                [
                    self.kernel(x, y)
                    for y in self.arms
                ]
                for x in self.arms
            ],
            dtype=float,
        )

        covariance = K_AA - projection

        covariance = (
            covariance + covariance.T
        ) / 2.0

        return mean, covariance

    # ------------------------------------------------------------------
    # Information gain
    # ------------------------------------------------------------------

    def _information_gain(self) -> float:
        """
        gamma_t(lambda)
            = 1/2 log det(I + K_t/lambda).

        The determinant is maintained incrementally.
        """
        return 0.5 * self._logdet

    # ------------------------------------------------------------------
    # Confidence factor
    # ------------------------------------------------------------------

    def _B(
        self,
        sigma_plus: float,
        delta: float | None = None,
    ) -> float:
        assert self.lambda_ is not None

        if delta is None:
            delta = self.delta

        gamma = self._information_gain()

        return (
            np.sqrt(self.lambda_) * self.C
            + sigma_plus
            * np.sqrt(
                2.0 * np.log(1.0 / delta)
                + 2.0 * gamma
            )
        )

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def _sample_function(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        v: float,
        sigma_plus: float,
    ) -> np.ndarray:
        """
        Draw

            f_tilde ~ N(
                mean,
                v^2 sigma_plus^2/lambda K_lambda
            ).
        """
        assert self.lambda_ is not None

        covariance = (
            covariance + covariance.T
        ) / 2.0

        covariance = covariance + (
            1e-10 * np.eye(self.number_arms)
        )

        posterior_covariance = (
            v ** 2
            * sigma_plus ** 2
            / self.lambda_
            * covariance
        )

        return self.np_random.multivariate_normal(
            mean=mean,
            cov=posterior_covariance,
        )

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------

    def update(
        self,
        arm: int,
        reward: float,
    ) -> None:
        """
        Add an observation.

        The subclass is responsible for ensuring that lambda
        has the appropriate value before this method is called.
        """
        self.observed_arms.append(int(arm))
        self.rewards.append(float(reward))

        self._add_observation_fixed_lambda(
            arm=int(arm),
            reward=float(reward),
        )


class KernelTSKnownVariance(_KernelTSBase):
    """
    Kernel Thompson Sampling with known noise variance.
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        C: float = 5.0,
        sigma: float | None = None,
        delta: float = 0.1 / 12,
        name: str = "Kernel TS Known Variance",
        seed: int = 1,
    ):
        super().__init__(
            env=env,
            C=C,
            delta=delta,
            name=name,
            seed=seed,
        )

        if sigma is None:
            raise ValueError(
                "sigma must be provided for "
                "KernelTSKnownVariance."
            )

        if sigma <= 0:
            raise ValueError(
                "sigma must be strictly positive."
            )

        self.sigma = float(sigma)

        self.lambda_ = (
            self.sigma ** 2
            / self.C ** 2
        )

    def reset(self) -> None:
        super().reset()

        self.observed_arms = []
        self.rewards = []

        self.lambda_ = (
            self.sigma ** 2
            / self.C ** 2
        )

        self._initialize_regression()

    def select_arm(self) -> int:
        assert self.lambda_ is not None

        mean, covariance = (
            self._posterior_all_arms()
        )

        B = self._B(
            sigma_plus=self.sigma,
            delta=self.delta,
        )

        v = B / self.sigma

        sampled_function = (
            self._sample_function(
                mean=mean,
                covariance=covariance,
                v=v,
                sigma_plus=self.sigma,
            )
        )

        return int(
            np.argmax(sampled_function)
        )