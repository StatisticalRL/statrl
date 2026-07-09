import numpy as np

from statrl.settings.bandits.adversarial.lipschitz.agent import Agent

class ALFLearner(Agent):
    """
    ALF: Adversarial Lipschitz Forecaster (discretization + Hedge)

    Learns in a continuous Lipschitz bandit/online optimization setting
    by reducing to a finite expert set.
    """

    def __init__(
        self,
        action_space,
        epsilon: float,
        eta: float,
        horizon: int,
        metric=None,
        sampling: str = "argmax",
    ):
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

    def play(self, observation=None):
        """
        Selects an action according to exponential weights.
        """

        probs = self._get_probabilities()

        if self.sampling == "sample":
            idx = np.random.choice(self.n_actions, p=probs)
        else:
            idx = np.argmax(probs)

        self.last_idx = idx
        return self.actions[idx]

    def update(self, action, reward, observation=None):
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

    def _get_probabilities(self):
        """
        Convert weights to probability distribution.
        """
        w = np.array(self.weights)
        return w / np.sum(w)

    def _build_cover(self, action_space, epsilon):
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

            grids = [np.linspace(low[d], high[d], steps) for d in range(dims)]

            mesh = np.meshgrid(*grids)
            points = np.stack(mesh, axis=-1).reshape(-1, dims)

            return points

        raise NotImplementedError("General metric cover not implemented.")