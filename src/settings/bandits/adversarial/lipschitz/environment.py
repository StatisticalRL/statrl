import numpy as np
from typing import Callable, Any, Optional


class LipschitzAdversarialEnv:
    """
    Gymnasium-like environment for adversarial Lipschitz online optimization.

    The environment exposes a sequence of reward functions f_t(x),
    chosen by an adversary (or precomputed generator).

    Each step:
        action = learner.play(observation)
        reward = f_t(action)
    """

    def __init__(
        self,
        action_space: Any,
        reward_function_sequence: Callable[[int], Callable[[np.ndarray], float]],
        horizon: int,
        observation_fn: Optional[Callable[[int], Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        action_space :
            Continuous metric space (e.g. Box in Gymnasium terms).

        reward_function_sequence :
            Function mapping time t -> reward function f_t(x).

        horizon :
            Number of rounds T.

        observation_fn :
            Optional function producing observation at time t.
            If None, observation is trivial (bandit setting).
        """
        self.action_space = action_space
        self.reward_function_sequence = reward_function_sequence
        self.horizon = horizon
        self.observation_fn = observation_fn

        self.t = 0
        self.current_f: Optional[Callable[[np.ndarray], float]] = None

    def reset(self, seed: Optional[int] = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.current_f = self.reward_function_sequence(self.t)

        obs = self._get_observation()
        info: dict = {}

        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Executes one round of interaction.

        Parameters
        ----------
        action :
            x_t chosen by learner

        Returns
        -------
        obs : next observation
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """

        # Evaluate adversarial reward function at chosen action
        assert self.current_f is not None, "step() called before reset()"
        reward = float(self.current_f(action))

        # Advance time
        self.t += 1

        terminated = self.t >= self.horizon
        truncated = False

        if not terminated:
            self.current_f = self.reward_function_sequence(self.t)

        obs = self._get_observation()
        info = {
            "time": self.t,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Any:
        if self.observation_fn is None:
            return None
        return self.observation_fn(self.t)