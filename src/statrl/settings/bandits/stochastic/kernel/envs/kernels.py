from __future__ import annotations

import numpy as np

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def RBFKernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Squared-exponential / RBF kernel."""

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        d2 = np.sum((x - y) ** 2)
        return variance * np.exp(-d2 / (2.0 * lengthscale ** 2))

    return kernel


def Matern32Kernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Matérn-3/2 kernel."""

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        r = np.linalg.norm(x - y)
        z = np.sqrt(3.0) * r / lengthscale
        return variance * (1.0 + z) * np.exp(-z)

    return kernel


def Matern52Kernel(lengthscale: float = 0.2, variance: float = 1.0):
    """Matérn-5/2 kernel."""

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        r = np.linalg.norm(x - y)
        z = np.sqrt(5.0) * r / lengthscale
        return variance * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)

    return kernel


kernel_dictionary = {"RBFKernel": RBFKernel, "Matern32Kernel": Matern32Kernel, "Matern52Kernel":Matern52Kernel}

# ---------------------------------------------------------------------------
# Kernel bandit constructor
# ---------------------------------------------------------------------------
from collections.abc import Sequence

def KernelBandit(
    nb_arms: int | Sequence[int],
    kernel: str,
    lengthscale: float = 0.2,
    variance: float = 1.0,
    noise_std: float = 0.1,
    name: str = "KernelBandit",
    function_seed: int =1,
) -> KernelBanditEnv:
    """Generate a kernel bandit on a regular grid in [0, 1]^d."""

    if isinstance(nb_arms, int):
        nb_arms = [nb_arms]

    axes = [
        np.linspace(0.0, 1.0, n)
        for n in nb_arms
    ]

    mesh = np.meshgrid(*axes, indexing="ij")

    arms = np.stack(
        [m.ravel() for m in mesh],
        axis=-1,
    )

    name = (
        f"{name}"
        f"-{kernel}"
        f"-K{list(nb_arms)}"
        f"-lengthscale{lengthscale}"
        f"-variance{variance}"
        f"-noise{noise_std}"
    )


    return KernelBanditEnv(
        arm_features=arms,
        kernel=kernel_dictionary[kernel](
            lengthscale=lengthscale,
            variance=variance,
        ),
        noise_std=noise_std,
        name=name,
        function_seed=function_seed
    )
