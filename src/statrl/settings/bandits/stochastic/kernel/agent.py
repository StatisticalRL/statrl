from abc import ABC, abstractmethod
from typing import Callable

from gymnasium.utils import seeding
import numpy as np

class KernelBanditAgent(ABC):

    def __init__(
        self,
        arms: np.ndarray,
        kernel: Callable,
        noise_std: float = 0.1,
        name: str = "Kernel Bandit Agent",
        seed: int = 1,
    ) -> None:
        self.arms = np.asarray(arms)
        self.kernel = kernel
        self.noise_std = noise_std
        self.name = name
        self.seed = seed

    def reset(self) -> None:
        self.np_random, self.seed = seeding.np_random(self.seed)

    @abstractmethod
    def select_arm(self) -> int:
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        ...