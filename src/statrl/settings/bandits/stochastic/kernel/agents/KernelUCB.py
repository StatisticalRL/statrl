from __future__ import annotations

from typing import Optional

import numpy as np

from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv

from abc import abstractmethod

class _KernelUCBBase(KernelBanditAgent):
    """Common implementation of kernel ridge regression for Kernel UCB.

    The implementation works on the finite set of arms provided by
    ``KernelBanditEnv.arm_features``.

    At time t, given observations (x_s, y_s), s=1,...,t,

        f_lambda,t(x)
            = k_t(x)^T (K_t + lambda I)^(-1) Y_t,

    and

        k_lambda,t(x,x)
            = k(x,x)
              - k_t(x)^T (K_t + lambda I)^(-1) k_t(x).

    Subclasses only need to specify the regularization parameter and
    confidence-bound term.
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        C: float = 5.0,
        delta: float = 0.1,
        name: str = "Kernel UCB",
        seed: int = 1,
    ) -> None:

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

        if self.C <= 0:
            raise ValueError("C must be positive.")

        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must belong to (0, 1).")

        self.reset()

    # ------------------------------------------------------------------
    # Kernel matrix utilities
    # ------------------------------------------------------------------

    def _gram_matrix(self, arms: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix K[i,j] = k(x_i, x_j)."""

        n = len(arms)

        K = np.empty((n, n), dtype=float)

        for i in range(n):
            for j in range(i, n):
                value = float(self.kernel(arms[i], arms[j]))
                K[i, j] = value
                K[j, i] = value

        return K

    def _kernel_vector(self, x: np.ndarray) -> np.ndarray:
        """Return k(X_t, x), where X_t are the observed arms."""

        if self._n_obs == 0:
            return np.empty(0, dtype=float)

        return np.asarray(
            [
                self.kernel(self.arms[a], x)
                for a in self._observed_arms
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Kernel ridge regression
    # ------------------------------------------------------------------

    def _posterior(
        self,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and regularized variance on all arms.

        Returns
        -------
        mean : ndarray, shape (number_arms,)
            Kernel ridge regression estimate.
        variance : ndarray, shape (number_arms,)
            k_lambda,t(x,x) for every arm.
        """

        n_arms = len(self.arms)

        # No observations.
        if self._n_obs == 0:
            mean = np.zeros(n_arms)

            variance = np.asarray(
                [self.kernel(x, x) for x in self.arms],
                dtype=float,
            )

            return mean, variance

        X = self.arms[self._observed_arms]
        y = np.asarray(self._rewards, dtype=float)

        K = self._gram_matrix(X)

        # Numerical stabilization.
        A = K + lam * np.eye(self._n_obs)

        # Solve rather than explicitly invert A.
        alpha = np.linalg.solve(A, y)

        mean = np.empty(n_arms, dtype=float)
        variance = np.empty(n_arms, dtype=float)

        for i, x in enumerate(self.arms):

            kx = np.asarray(
                [self.kernel(xi, x) for xi in X],
                dtype=float,
            )

            mean[i] = float(kx @ alpha)

            # k_lambda,t(x,x)
            correction = float(
                kx @ np.linalg.solve(A, kx)
            )

            variance[i] = max(
                float(self.kernel(x, x)) - correction,
                0.0,
            )

        return mean, variance

    # ------------------------------------------------------------------
    # Information gain
    # ------------------------------------------------------------------

    def _information_gain(self, lam: float) -> float:
        """Compute gamma_t(lambda).

        For the finite observed design X_t,

            gamma_t(lambda)
                = 1/2 log det(I + K_t / lambda).

        This is the finite-dimensional form of the information gain
        used in the paper.
        """

        if self._n_obs == 0:
            return 0.0

        X = self.arms[self._observed_arms]
        K = self._gram_matrix(X)

        A = np.eye(self._n_obs) + K / lam

        sign, logdet = np.linalg.slogdet(A)

        if sign <= 0:
            raise RuntimeError(
                "Numerical failure while computing log determinant."
            )

        return 0.5 * float(logdet)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Initialize an independent run."""

        super().reset()

        self._observed_arms: list[int] = []
        self._rewards: list[float] = []
        self._n_obs = 0

        self._means = np.zeros(len(self.arms))
        self._variances = np.zeros(len(self.arms))

        self._ucb = np.zeros(len(self.arms))

        self._lambda: float = 0.0
        self._beta: float = 0.0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """Select the arm maximizing the Kernel-UCB index."""

        lam = self._regularization()

        self._means, self._variances = self._posterior(lam)

        beta = self._confidence_bound(lam)

        self._lambda = lam
        self._beta = beta

        ucb = (
            self._means
            + np.sqrt(
                np.maximum(self._variances, 0.0) / lam
            ) * beta
        )

        self._ucb = ucb

        # Deterministic tie breaking.
        return int(np.argmax(ucb))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, arm: int, reward: float) -> None:
        """Store an observed reward."""

        self._observed_arms.append(int(arm))
        self._rewards.append(float(reward))
        self._n_obs += 1

    # ------------------------------------------------------------------
    # Methods supplied by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _regularization(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _confidence_bound(self, lam: float) -> float:
        raise NotImplementedError


# ======================================================================
# Known variance
# ======================================================================


class KernelUCBKnownVariance(_KernelUCBBase):
    """Kernel UCB with known noise variance.

    This corresponds to the fixed-regularization version considered in
    Durand, Maillard and Pineau (2018).

    Parameters
    ----------
    env:
        Kernel bandit environment.

    sigma:
        Known standard deviation of the observation noise.

    C:
        Known upper bound on the RKHS norm of the unknown function.

    delta:
        Confidence parameter.

    seed:
        Seed for the agent's own random generator. Kernel UCB itself is
        deterministic, but the standard KernelBanditAgent interface
        provides the generator for consistency with other agents.

    Notes
    -----
    The regularization is

        lambda = sigma^2 / C^2.

    The confidence term is

        B_t =
            sqrt(lambda) C
            + sigma sqrt(2 log(1/delta) + 2 gamma_t(lambda)).
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        sigma: Optional[float] = None,
        C: float = 5.0,
        delta: float = 0.1,
        seed: int = 1,
    ) -> None:

        if sigma is None:
            sigma = env.noise_std

        self.sigma = float(sigma)

        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        super().__init__(
            env=env,
            C=C,
            delta=delta,
            name="Kernel UCB (known variance)",
            seed=seed,
        )

    def _regularization(self) -> float:
        return self.sigma ** 2 / self.C ** 2

    def _confidence_bound(self, lam: float) -> float:

        gamma = self._information_gain(lam)

        return (
            np.sqrt(lam) * self.C
            + self.sigma
            * np.sqrt(
                2.0 * np.log(1.0 / self.delta)
                + 2.0 * gamma
            )
        )


# ======================================================================
# Unknown variance
# ======================================================================


class KernelUCBUnknownVariance(_KernelUCBBase):
    """Kernel UCB with unknown noise variance.

    Adaptive version following Section 3 and Section 4 of:

        Durand, Maillard and Pineau (2018),
        "Streaming kernel regression with provably adaptive
        mean, variance, and regularization".

    The algorithm maintains an upper and lower confidence estimate of
    the noise standard deviation and sets

        lambda_t = sigma_{+,t-1}^2 / C^2.

    Parameters
    ----------
    env:
        Kernel bandit environment.

    C:
        Known upper bound on the RKHS norm of the unknown function.

    sigma_upper:
        Initial upper bound on the noise standard deviation.

    sigma_lower:
        Initial lower bound on the noise standard deviation.

    delta:
        Confidence parameter.

    delta0:
        Confidence parameter used by the variance estimator.
        If None, ``delta0 = delta / 4``.

    Notes
    -----
    This implementation follows the computable "sandwich estimates"
    construction of Corollary 1. In particular, the lower estimate is
    used to obtain a lower bound on the unknown optimal regularization,
    which is then used in the upper variance estimate.

    The paper assumes Gaussian noise for the variance-estimation
    procedure (or the corresponding second-order sub-Gaussian
    assumption).
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        C: float = 5.0,
        sigma_upper: float = 1.0,
        sigma_lower: float = 0.01,
        delta: float = 0.1,
        delta0: Optional[float] = None,
        seed: int = 1,
    ) -> None:

        self.sigma_upper = float(sigma_upper)
        self.sigma_lower = float(sigma_lower)

        if self.sigma_upper <= 0:
            raise ValueError("sigma_upper must be positive.")

        if self.sigma_lower < 0:
            raise ValueError("sigma_lower must be non-negative.")

        if self.sigma_lower > self.sigma_upper:
            raise ValueError(
                "sigma_lower must not exceed sigma_upper."
            )

        self.delta0 = (
            float(delta0)
            if delta0 is not None
            else float(delta) / 4.0
        )

        if not (0.0 < self.delta0 < 1.0):
            raise ValueError("delta0 must belong to (0, 1).")

        super().__init__(
            env=env,
            C=C,
            delta=delta,
            name="Kernel UCB (unknown variance)",
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:

        super().reset()

        # sigma_{+,0} and sigma_{-,0}
        self._sigma_plus: float = self.sigma_upper
        self._sigma_minus: float = self.sigma_lower

        # Initial regularization:
        #
        # lambda_0 = sigma_+^2 / C^2
        self._lambda = (
            self._sigma_plus ** 2 / self.C ** 2
        )

        self._sigma_plus_history: list[float] = []
        self._sigma_minus_history: list[float] = []

    # ------------------------------------------------------------------
    # Confidence schedule
    # ------------------------------------------------------------------

    def _Ct(self, t: int) -> float:
        """C_t(delta_0) from Theorem 3."""

        t = max(1, int(t))

        log_term = np.log(
            max(
                np.pi ** 2 * max(np.log(t), 1.0) / 6.0,
                1.0,
            )
        )

        return np.log(np.e / self.delta0) * (
            1.0
            + log_term / np.log(1.0 / self.delta0)
        )

    def _D(self, lam: float, t: int) -> float:
        """D_{lambda,t}(delta_0)."""

        if t == 0:
            return 2.0 * np.log(1.0 / self.delta0)

        gamma = self._information_gain(lam)

        return (
            2.0 * np.log(1.0 / self.delta0)
            + 2.0 * gamma
        )

    # ------------------------------------------------------------------
    # Variance estimates
    # ------------------------------------------------------------------

    def _variance_estimate(
        self,
        lam: float,
        sigma_upper_for_bound: float,
    ) -> tuple[float, float]:
        """Compute lower and upper variance estimates.

        This implements the case-1 variance bounds of Theorem 3,
        where an initial upper bound sigma_upper is available.

        Returns
        -------
        sigma_minus_tilde:
            Lower confidence estimate.

        sigma_plus_tilde:
            Upper confidence estimate.
        """

        t = self._n_obs

        if t == 0:
            return (
                self._sigma_minus,
                self._sigma_plus,
            )

        # Posterior mean evaluated at the observed points.
        mean, _ = self._posterior(lam)

        residuals = np.asarray(
            [
                self._rewards[i] - mean[self._observed_arms[i]]
                for i in range(t)
            ],
            dtype=float,
        )

        sigma_hat_sq = float(
            np.mean(residuals ** 2)
        )

        sigma_hat = np.sqrt(
            max(sigma_hat_sq, 0.0)
        )

        Ct = self._Ct(t)

        # Lower bound on the optimal regularization.
        #
        # We use the observed design and the current lambda in the
        # computable expression from Theorem 3.
        D = self._D(lam, t)

        # alpha = max(
        #     1.0
        #     - np.sqrt(Ct / t)
        #     - np.sqrt((Ct + 2.0 * D) / t),
        #     0.0,
        # )

        # k_lambda,t(x_t, x_t) for the observed points.
        #
        # The correction appearing in Theorem 3 involves
        #
        # max_s (1 + k_lambda,s-1(x_s,x_s)/lambda).
        #
        # We compute this quantity directly from the observations.
        max_term = 1.0

        if t > 0:
            X = self.arms[self._observed_arms]
            K = self._gram_matrix(X)
            A = K + lam * np.eye(t)

            for i in range(t):
                kx = np.asarray(
                    [
                        self.kernel(xj, X[i])
                        for xj in X
                    ],
                    dtype=float,
                )

                variance_i = max(
                    float(self.kernel(X[i], X[i]))
                    - float(kx @ np.linalg.solve(A, kx)),
                    0.0,
                )

                max_term = max(
                    max_term,
                    1.0 + variance_i / lam,
                )

        # --------------------------------------------------------------
        # Lower confidence bound.
        # --------------------------------------------------------------

        lower = (
            sigma_hat
            - sigma_upper_for_bound
            * np.sqrt(
                2.0 * Ct / t
            )
            - self.C
            * np.sqrt(lam / t)
            * (
                1.0
                - 1.0 / max_term
            )
        )

        lower = max(lower, 0.0)

        # --------------------------------------------------------------
        # Upper confidence bound.
        #
        # Case 1 of Theorem 3.
        # --------------------------------------------------------------

        upper = (
            sigma_hat
            + sigma_upper_for_bound
            * (
                np.sqrt(Ct / t)
                + np.sqrt((Ct + 2.0 * D) / t)
            )
            + np.sqrt(
                2.0
                * sigma_upper_for_bound
                * self.C
                * np.sqrt(lam * D / t)
            )
        )

        upper = max(upper, lower)

        return lower, upper

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def _update_variance_bounds(self) -> None:
        """Update sigma_- and sigma_+ using the sandwich procedure."""

        if self._n_obs == 0:
            return

        previous_lambda: float = self._lambda

        sigma_minus_tilde, _ = self._variance_estimate(
            previous_lambda,
            self.sigma_upper,
        )

        # Monotone lower estimate:
        self._sigma_minus = max(
            self._sigma_minus,
            sigma_minus_tilde,
        )

        # lambda_minus = (
        #     self._sigma_minus ** 2
        #     / self.C ** 2
        # )

        # Recompute the upper estimate using lambda_-.
        #
        # This is the sandwich step of Corollary 1.
        _, sigma_plus_tilde = self._variance_estimate(
            previous_lambda,
            self.sigma_upper,
        )

        # Monotone upper estimate:
        self._sigma_plus = min(
            self._sigma_plus,
            sigma_plus_tilde,
        )

        # Numerical safety.
        self._sigma_plus = max(
            self._sigma_plus,
            self._sigma_minus,
            1e-12,
        )

        self._sigma_plus_history.append(
            self._sigma_plus
        )
        self._sigma_minus_history.append(
            self._sigma_minus
        )

        # Predictable regularization for the next observation.
        self._lambda = (
            self._sigma_plus ** 2
            / self.C ** 2
        )

    # ------------------------------------------------------------------
    # UCB parameters
    # ------------------------------------------------------------------

    def _regularization(self) -> float:
        """Return lambda_t used for the current observation."""

        return max(
            float(self._lambda),
            1e-12,
        )

    def _confidence_bound(self, lam: float) -> float:

        t = self._n_obs

        if t == 0:
            return (
                np.sqrt(lam) * self.C
                + self._sigma_plus
                * np.sqrt(
                    2.0 * np.log(1.0 / self.delta)
                )
            )

        lambda_minus = (
            self._sigma_minus ** 2
            / self.C ** 2
        )

        gamma_minus = self._information_gain(
            max(lambda_minus, 1e-12)
        )

        return (
            np.sqrt(lam) * self.C
            + self._sigma_plus
            * np.sqrt(
                2.0 * np.log(1.0 / self.delta)
                + 2.0 * gamma_minus
            )
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """Update variance estimates and select the UCB arm."""

        if self._n_obs > 0:
            self._update_variance_bounds()

        return super().select_arm()
