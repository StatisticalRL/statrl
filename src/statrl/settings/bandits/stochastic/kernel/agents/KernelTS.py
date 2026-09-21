from __future__ import annotations

from abc import ABC
#from typing import Callable

import numpy as np

from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class _KernelTSBase(KernelBanditAgent, ABC):
    """
    Base implementation for Kernel Thompson Sampling.

    The implementation follows Durand, Maillard & Pineau (2018),
    Algorithm 1, for a finite set of arms.

    The kernel posterior is based on

        f_lambda,t(x)
            = k_t(x)^T (K_t + lambda I)^(-1) Y_t

    and

        k_lambda,t(x, x)
            = k(x,x)
              - k_t(x)^T (K_t + lambda I)^(-1) k_t(x).

    The Thompson covariance used by the algorithm is

        Sigma_t(x, x')
            = sigma_plus^2 / lambda
              * k_lambda,t(x, x').

    A Gaussian sample is then drawn from

        N(mu_t, v_t^2 Sigma_t)

    and the arm maximizing the sampled function is selected.
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
        self.sigma_plus: float | None = None

    # ------------------------------------------------------------------
    # Kernel computations
    # ------------------------------------------------------------------

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix K(X,Y).
        """
        return np.asarray(
            [[self.kernel(x, y) for y in Y] for x in X],
            dtype=float,
        )

    def _gram_matrix(self) -> np.ndarray:
        """
        Gram matrix of the observations.
        """
        X = self.arms[np.asarray(self.observed_arms, dtype=int)]

        if len(X) == 0:
            return np.empty((0, 0), dtype=float)

        return self._kernel_matrix(X, X)

    def _posterior(
        self,
        lambda_: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior mean and regularized kernel covariance
        over all arms.

        Returns
        -------
        mean:
            Vector of size K.

        covariance:
            K x K matrix containing k_lambda,t(x,x').
        """
        n = len(self.observed_arms)

        if n == 0:
            Kxx = self._kernel_matrix(self.arms, self.arms)
            mean = np.zeros(self.number_arms)
            return mean, Kxx

        X = self.arms[np.asarray(self.observed_arms, dtype=int)]
        y = np.asarray(self.rewards, dtype=float)

        K = self._kernel_matrix(X, X)

        A = K + lambda_ * np.eye(n)

        # Cholesky factorization.
        L = np.linalg.cholesky(A + 1e-10 * np.eye(n))

        # alpha = A^{-1} y
        alpha = np.linalg.solve(
            L.T,
            np.linalg.solve(L, y),
        )

        # K_XA has shape (K, n).
        K_XA = self._kernel_matrix(self.arms, X)

        mean = K_XA @ alpha

        # A^{-1} K_XA^T
        V = np.linalg.solve(L, K_XA.T)

        K_AA = self._kernel_matrix(self.arms, self.arms)

        covariance = K_AA - V.T @ V

        # Numerical symmetrization / clipping.
        covariance = 0.5 * (covariance + covariance.T)

        return mean, covariance

    # ------------------------------------------------------------------
    # Information gain
    # ------------------------------------------------------------------

    def _information_gain(self, lambda_: float) -> float:
        """
        Compute

            gamma_t(lambda)
                = 1/2 log det(I + K_t / lambda).

        """
        n = len(self.observed_arms)

        if n == 0:
            return 0.0

        K = self._gram_matrix()

        A = np.eye(n) + K / lambda_

        sign, logdet = np.linalg.slogdet(A)

        if sign <= 0:
            raise np.linalg.LinAlgError(
                "Information-gain matrix is not positive definite."
            )

        return 0.5 * logdet

    # ------------------------------------------------------------------
    # Confidence factor
    # ------------------------------------------------------------------

    def _B(
        self,
        lambda_: float,
        sigma_plus: float,
        delta: float | None = None,
    ) -> float:
        """
        Confidence factor B_{lambda,t}(delta).

        For known variance this corresponds to

            sqrt(lambda) C
            + sigma sqrt(2 log(1/delta) + 2 gamma_t(lambda)).

        For adaptive variance, lambda may be the current adaptive
        regularization and sigma_plus is its corresponding upper
        noise estimate.
        """
        if delta is None:
            delta = self.delta

        gamma = self._information_gain(lambda_)

        return (
            np.sqrt(lambda_) * self.C
            + sigma_plus
            * np.sqrt(
                2.0 * np.log(1.0 / delta)
                + 2.0 * gamma
            )
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_function(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        v: float,
    ) -> np.ndarray:
        """
        Draw

            f_tilde ~ N(mean, v^2 covariance).
        """
        covariance = 0.5 * (covariance + covariance.T)

        # Small jitter for numerical stability.
        jitter = 1e-10
        covariance = covariance + jitter * np.eye(self.number_arms)

        sampled = self.np_random.multivariate_normal(
            mean=mean,
            cov=(v ** 2) * covariance,
        )

        return sampled

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------

    def update(self, arm: int, reward: float) -> None:
        self.observed_arms.append(int(arm))
        self.rewards.append(float(reward))


class KernelTSKnownVariance(_KernelTSBase):
    """
    Kernel Thompson Sampling with known noise variance.

    Durand, Maillard & Pineau (2018), Algorithm 1, oracle setting.

    Parameters
    ----------
    env:
        Kernel bandit environment.

    C:
        Known upper bound on the RKHS norm ||f*||_K.

    sigma:
        Known noise standard deviation.

    delta:
        Confidence parameter used in the inflation factor.

    seed:
        Random seed for Thompson sampling.
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
                "KernelTSKnownVariance requires an explicit "
                "known noise standard deviation `sigma`."
            )

        if sigma <= 0:
            raise ValueError("sigma must be strictly positive.")

        self.sigma = float(sigma)

        # Oracle regularization:
        #
        # lambda = sigma^2 / C^2
        self.lambda_ = self.sigma ** 2 / self.C ** 2
        self.sigma_plus = self.sigma

    def reset(self) -> None:
        super().reset()

        self.observed_arms = []
        self.rewards = []

        self.lambda_ = self.sigma ** 2 / self.C ** 2
        self.sigma_plus = self.sigma

    def select_arm(self) -> int:
        """
        Sample a function from the inflated posterior and return
        its maximizing arm.
        """
        assert self.lambda_ is not None
        mean, covariance = self._posterior(self.lambda_)

        B = self._B(
            lambda_=self.lambda_,
            sigma_plus=self.sigma,
            delta=self.delta,
        )

        # Paper:
        #
        # v_t = B_{lambda,t-1}(delta) / sigma.
        v = B / self.sigma

        sampled_function = self._sample_function(
            mean=mean,
            covariance=(self.sigma ** 2 / self.lambda_) * covariance,
            v=v,
        )

        return int(np.argmax(sampled_function))


class KernelTSUnknownVariance(_KernelTSBase):
    """
    Kernel Thompson Sampling with unknown noise variance and
    adaptive variance estimation.

    Implements the adaptive regularization scheme of
    Durand, Maillard & Pineau (2018), Algorithm 1 together
    with Theorem 3 / Corollary 1.

    Parameters
    ----------
    env:
        Kernel bandit environment.

    C:
        Known upper bound on ||f*||_K.

    sigma_plus:
        Initial upper bound on the noise standard deviation.

    sigma_minus:
        Initial lower bound on the noise standard deviation.

    delta:
        Confidence parameter used by Kernel TS.

    delta0:
        Confidence parameter used for variance estimation.

    seed:
        Random seed for Thompson sampling.
    """

    def __init__(
        self,
        env: KernelBanditEnv,
        C: float = 5.0,
        sigma_plus: float = 1.0,
        sigma_minus: float = 0.01,
        delta: float = 0.1 / 12,
        delta0: float = 0.1 / 12,
        name: str = "Kernel TS Unknown Variance",
        seed: int = 1,
    ):
        super().__init__(
            env=env,
            C=C,
            delta=delta,
            name=name,
            seed=seed,
        )

        if C <= 0:
            raise ValueError("C must be strictly positive.")

        if sigma_plus <= 0:
            raise ValueError(
                "sigma_plus must be strictly positive."
            )

        if sigma_minus < 0:
            raise ValueError(
                "sigma_minus must be non-negative."
            )

        if sigma_minus > sigma_plus:
            raise ValueError(
                "sigma_minus must not exceed sigma_plus."
            )

        self.sigma_plus_initial = float(sigma_plus)
        self.sigma_minus_initial = float(sigma_minus)

        self.delta0 = float(delta0)

        self.sigma_plus = self.sigma_plus_initial
        self.sigma_minus = self.sigma_minus_initial

        # lambda_0 = sigma_+^2 / C^2
        self.lambda_ = (
            self.sigma_plus ** 2 / self.C ** 2
        )

    def reset(self) -> None:
        super().reset()

        self.observed_arms = []
        self.rewards = []

        self.sigma_plus = self.sigma_plus_initial
        self.sigma_minus = self.sigma_minus_initial

        self.lambda_ = (
            self.sigma_plus ** 2 / self.C ** 2
        )

    # ------------------------------------------------------------------
    # Variance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _C_t(t: int, delta0: float) -> float:
        """
        C_t(delta_0) from Theorem 3.

        The theorem's expression contains log(log(t)); the variance
        update is therefore started only once t >= 2.
        """
        if t < 2:
            raise ValueError(
                "C_t(delta0) is only defined here for t >= 2."
            )

        return np.log(np.e / delta0) * (
            1.0
            + np.log(
                np.pi ** 2 * np.log(t) / 6.0
            )
            / np.log(1.0 / delta0)
        )

    def _D(
        self,
        lambda_: float,
        t: int,
        delta0: float,
    ) -> float:
        """
        D_{lambda,t}(delta_0)
            = 2 log(1/delta_0) + 2 gamma_t(lambda).
        """
        return (
            2.0 * np.log(1.0 / delta0)
            + 2.0 * self._information_gain(lambda_)
        )

    def _variance_estimate(
        self,
        lambda_: float,
    ) -> float:
        """
        Empirical variance estimate

            sigma_hat^2
                = (1/t) sum_s
                    (y_s - f_lambda,t(x_s))^2.

        The posterior mean is computed using all observations.
        """
        t = len(self.observed_arms)

        if t == 0:
            return 0.0

        mean, _ = self._posterior(lambda_)

        observed = np.asarray(
            self.observed_arms,
            dtype=int,
        )

        residuals = (
            np.asarray(self.rewards, dtype=float)
            - mean[observed]
        )

        variance = np.mean(residuals ** 2)

        return float(max(variance, 0.0))

    def _max_leverage(
        self,
        lambda_: float,
    ) -> float:
        """
        Compute

            max_{t' <= t}
                [1 + lambda^{-1}
                    k_{lambda,t'-1}(x_t', x_t')].

        This is the quantity appearing in Theorem 3's lower
        variance bound.
        """
        t = len(self.observed_arms)

        if t == 0:
            return 1.0

        maximum = 1.0

        for n in range(1, t + 1):
            X = self.arms[
                np.asarray(
                    self.observed_arms[:n - 1],
                    dtype=int,
                )
            ]

            x = self.arms[
                self.observed_arms[n - 1]
            ]

            if len(X) == 0:
                k_lambda = self.kernel(x, x)
            else:
                K = self._kernel_matrix(X, X)
                k = np.asarray(
                    [self.kernel(xi, x) for xi in X],
                    dtype=float,
                )

                A = K + lambda_ * np.eye(len(X))

                L = np.linalg.cholesky(
                    A + 1e-10 * np.eye(len(X))
                )

                z = np.linalg.solve(L, k)

                k_lambda = (
                    self.kernel(x, x)
                    - np.dot(z, z)
                )

            k_lambda = max(float(k_lambda), 0.0)

            leverage = 1.0 + k_lambda / lambda_

            maximum = max(maximum, leverage)

        return maximum

    def _sigma_minus_tilde(
        self,
        lambda_: float,
    ) -> float:
        """
        Lower variance estimate from Theorem 3, case 1:

          sigma_tilde^- =
              sigma_hat
              - sigma_+ sqrt(2 C_t / t)
              - C sqrt(lambda/t)
                [1 - 1 / max leverage].
        """
        t = len(self.observed_arms)

        if t < 2:
            return self.sigma_minus

        C_t = self._C_t(t, self.delta0)

        sigma_hat = np.sqrt(
            self._variance_estimate(lambda_)
        )

        maximum_leverage = self._max_leverage(lambda_)

        correction = (
            1.0
            - 1.0 / maximum_leverage
        )

        estimate = (
            sigma_hat
            - self.sigma_plus
            * np.sqrt(2.0 * C_t / t)
            - self.C
            * np.sqrt(lambda_ / t)
            * correction
        )

        return max(float(estimate), 0.0)

    def _sigma_plus_tilde(
        self,
        lambda_: float,
        lambda_minus: float,
    ) -> float:
        """
        Upper variance estimate from Theorem 3, case 1:

          sigma_tilde^+
              = sigma_hat
                + sigma_+
                  [sqrt(C_t/t)
                   + sqrt((C_t + 2D_{lambda*,t})/t)]
                + sqrt(
                    2 sigma_+ C
                    sqrt(lambda D_{lambda*,t}/t)
                  ).

        Since lambda* is unknown, Corollary 1 replaces it by
        lambda_minus <= lambda*.
        """
        t = len(self.observed_arms)

        if t < 2:
            return self.sigma_plus_initial

        C_t = self._C_t(t, self.delta0)

        D = self._D(
            lambda_=lambda_minus,
            t=t,
            delta0=self.delta0,
        )

        sigma_hat = np.sqrt(
            self._variance_estimate(lambda_)
        )

        first = np.sqrt(C_t / t)

        second = np.sqrt(
            (C_t + 2.0 * D) / t
        )

        third = np.sqrt(
            2.0
            * self.sigma_plus_initial
            * self.C
            * np.sqrt(
                lambda_ * D / t
            )
        )

        estimate = (
            sigma_hat
            + self.sigma_plus_initial
            * (first + second)
            + third
        )

        return float(estimate)

    def _update_variance_bounds(self) -> None:
        """
        Perform the Corollary 1 update after observing y_t.

        Given the previous predictable regularization lambda_t:

          sigma_minus,t
              = max(
                    sigma_minus_tilde,t(lambda_t),
                    sigma_minus,t-1
                )

          lambda_minus
              = sigma_minus,t^2 / C^2

          sigma_plus,t
              = min(
                    sigma_plus_tilde,t(
                        lambda_t, lambda_minus
                    ),
                    sigma_plus,t-1
                )

          lambda_{t+1}
              = sigma_plus,t^2 / C^2.
        """
        t = len(self.observed_arms)

        if t < 2:
            return

        assert self.lambda_ is not None
        lambda_previous = self.lambda_

        # --------------------------------------------------------------
        # Lower bound on sigma
        # --------------------------------------------------------------

        sigma_minus_tilde = self._sigma_minus_tilde(
            lambda_=lambda_previous,
        )

        self.sigma_minus = max(
            sigma_minus_tilde,
            self.sigma_minus,
        )

        # Corresponding lower bound on lambda*
        lambda_minus = (
            self.sigma_minus ** 2
            / self.C ** 2
        )

        # --------------------------------------------------------------
        # Upper bound on sigma
        # --------------------------------------------------------------

        sigma_plus_tilde = self._sigma_plus_tilde(
            lambda_=lambda_previous,
            lambda_minus=lambda_minus,
        )

        self.sigma_plus = min(
            sigma_plus_tilde,
            self.sigma_plus,
        )

        # --------------------------------------------------------------
        # Predictable regularization for next round
        # --------------------------------------------------------------

        self.lambda_ = (
            self.sigma_plus ** 2
            / self.C ** 2
        )

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """
        Draw from the inflated posterior and select its maximizer.

        At time t, lambda_t and sigma_{+,t-1} depend only on
        observations up to t-1.
        """
        assert self.lambda_ is not None

        mean, covariance = self._posterior(
            self.lambda_
        )

        B = self._B(
            lambda_=self.lambda_,
            sigma_plus=self.sigma_plus,
            delta=self.delta,
        )

        # Algorithm 1:
        #
        # v_t =
        #     B_{lambda_t,t-1}(delta)
        #     / sigma_{+,t-1}
        #
        v = B / self.sigma_plus

        posterior_covariance = (
            self.sigma_plus ** 2
            / self.lambda_
        ) * covariance

        sampled_function = self._sample_function(
            mean=mean,
            covariance=posterior_covariance,
            v=v,
        )

        return int(np.argmax(sampled_function))

    def update(self, arm: int, reward: float) -> None:
        """
        Store the observation, then update the variance estimates
        and the regularization parameter for the next round.
        """
        super().update(arm, reward)

        self._update_variance_bounds()