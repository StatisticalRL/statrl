import numpy as np
from typing import Any, Optional

from statrl.settings.bandits.adversarial.lipschitz.agent import Agent

class ALFLearner(Agent):
    """
    ALF: Adversarial Lipschitz Forecaster (discretization + Hedge)

    Learns in a continuous Lipschitz bandit/online optimization setting
    by reducing to a finite expert set.
    """

    def __init__(
        self,
        action_space: Any,
        epsilon: float,
        eta: float,
        horizon: int,
        metric: Optional[Any] = None,
        sampling: str = "argmax",
    ) -> None:
        """
        Parameters
        ----------
        action_space:
            Continuous domain (assumed metric space or Box).

        epsilon:
            Discretization resolution.

        eta:
            Learning rate for exponential weights.

        horizon:
            Time horizon T (may be used for tuning epsilon/eta).

        metric:
            Optional metric for cover construction.

        sampling:
            'argmax' or 'sample'
        """

        self.action_space = action_space
        self.epsilon = epsilon
        self.eta = eta
        self.horizon = horizon
        self.metric = metric
        self.sampling = sampling

        # ------------------------------------------------------------
        # Step 1: build finite discretization (ε-net / grid / particles)
        # ------------------------------------------------------------
        self.actions = self._build_cover(action_space, epsilon)

        self.n_actions = len(self.actions)

        # ------------------------------------------------------------
        # Step 2: initialize uniform weights
        # ------------------------------------------------------------
        self.weights = np.ones(self.n_actions) / self.n_actions

        self.time = 0

    # ================================================================
    # CORE INTERFACE
    # ================================================================

    def select_arm(self, observation: Optional[Any] = None) -> np.ndarray:
        """
        Selects an action according to exponential weights.
        """

        probs = self._get_probabilities()

        if self.sampling == "sample":
            idx = np.random.choice(self.n_actions, p=probs)
        else:
            idx = int(np.argmax(probs))

        self.last_idx = idx
        return self.actions[idx]

    def update(self, action: np.ndarray, reward: float, observation: Optional[Any] = None) -> None:
        """
        Hedge update on discretized expert corresponding to chosen action.
        """

        idx = self.last_idx

        # importance-weighted or direct reward (bandit/full-info abstraction)
        r = float(reward)

        # ------------------------------------------------------------
        # Exponential weights update
        # ------------------------------------------------------------
        self.weights[idx] *= np.exp(self.eta * r)

        # normalize
        self.weights /= np.sum(self.weights)

        self.time += 1

    # ================================================================
    # INTERNALS
    # ================================================================

    def _get_probabilities(self) -> np.ndarray:
        """
        Convert weights to probability distribution.
        """
        w = np.array(self.weights)
        return w / np.sum(w)

    def _build_cover(self, action_space: Any, epsilon: float) -> np.ndarray:
        """
        Constructs ε-discretization of the action space.

        NOTE: In the paper this is abstract (metric cover).
        Here we instantiate a simple version for Box spaces.
        """

        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            low, high = action_space.low, action_space.high

            # grid resolution based on epsilon
            dims = len(low)
            steps = max(2, int(1.0 / epsilon))

            grids = [np.linspace(lo, hi, steps) for lo, hi in zip(low, high)]

            mesh = np.meshgrid(*grids)
            points = np.stack(mesh, axis=-1).reshape(-1, dims)

            return points

        raise NotImplementedError("General metric cover not implemented.")