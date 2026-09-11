from statrl.settings.utils import categorical_sample
import numpy as np

from gymnasium import Env, spaces
from gymnasium.utils import seeding

class DiscreteMDP(Env):
    """Finite Markov decision process with no exploitable structure.

    Parameters
    ----------
    nS : int
        Number of states.
    nA : int
        Number of actions.
    P : dict
        Transition kernel as a dict of dicts, with
        ``P[s][a] == [(probability, nextstate, done), ...]``. The
        probabilities of each ``P[s][a]`` must sum to one.
    R : dict
        Reward distributions, with ``R[s][a]`` exposing ``rvs()`` and
        ``mean()``, a :class:`~statrl.settings.utils.Dirac` for a
        deterministic reward, or any frozen :mod:`scipy.stats` distribution.
    isd : array-like of float, shape (nS,)
        Initial state distribution.
    nameActions : list of str, default=[]
        Human-readable action labels, used by the renderers.
    seed : int, optional
        Seed for the environment's generator.
    name : str, default='DiscreteMDP'
        Label used in logfiles, plot titles, and dump filenames.

    Attributes
    ----------
    s : int
        Current state.
    last : tuple
        Most recent ``(state, action, reward)``, consumed by the renderers.
    renderers : list
        Renderers notified on every :meth:`render` call.
    reward_range : tuple
        ``(0, 1)``, assumed rather than derived from ``R``.

    See Also
    --------
    statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim.RiverSwim :
        The standard hard-exploration instance.
    statrl.settings.markovdecisionprocess.discrete_nostructure.envs.randomMDP.RandomMDP :
        Randomly generated instances.

    Examples
    --------
    >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim import RiverSwim
    >>> env = RiverSwim(5)
    >>> env.nS, env.nA
    (5, 2)
    >>> state, info = env.reset()
    >>> state, reward, done, truncated, info = env.step(0)
    """


    def __init__(self, nS, nA, P, R, isd, nameActions=[], seed=None, name="DiscreteMDP"):
        self.name=name
        self.displayname=f"Markov Decisio Process {nS}x{nA}"
        self.nS = nS
        self.nA = nA
        self.P = P
        self.R = R

        self.isd = isd
        self.reward_range = (0, 1)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)

        # Rendering parameters and variables:

        self.last = (None, None, 0.)

        self.renderers: list = []

        # Initialization
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        """Seed the environment's generator.

        Parameters
        ----------
        seed : int, optional
            Seed to use. ``None`` draws a fresh one.

        Returns
        -------
        list of int
            The seed actually used, in a one-element list.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """Start a new episode by drawing a state from the initial distribution.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's generator.
        options : dict, optional
            Unused; accepted for :class:`gymnasium.Env` compatibility.

        Returns
        -------
        state : int
            The initial state.
        info : dict
            ``{"mean": 0}``, matching the shape of what :meth:`step` returns.
        """
        super().reset(seed=seed, options=options)
        self.np_random, seed = seeding.np_random(seed)
        self.s = categorical_sample(self.isd, self.np_random)

        self.last = (self.s,None, 0.)
        return self.s, {"mean": 0}

    def expected_reward(self, state, arm: int) -> float:
        """Mean reward of a state-action pair.

        Parameters
        ----------
        state : int
            The state.
        arm : int
            The action taken in it.

        Returns
        -------
        float
            The true mean reward. Never pass this to a learner.
        """
        return self.getMeanReward(state,arm)


    def change_rendermode(self,rendermode):
        """Set the render mode and mark the renderer as needing re-initialization.

        Parameters
        ----------
        rendermode : str
            The new render mode.
        """
        self.rendermode = rendermode
        self.initializedRenderer = False

    def step(self, a):
        """Take an action: draw the next state and a reward.

        Parameters
        ----------
        a : int
            Action to take in the current state.

        Returns
        -------
        state : int
            The new state.
        reward : float
            Reward drawn from ``R[s][a]``.
        done : bool
            Whether the transition was terminal.
        truncated : bool
            Always False; this setting has no time limit.
        info : dict
            ``{"mean": float}``, the *mean* reward of the pair. Returned for
            regret accounting only and must not be given to the learner.
        """
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        r = rewarddis.rvs()
        m = rewarddis.mean()
        self.s = s

        self.last = (s, a,r)
        return s, r, d, False, {"mean":m}

    def getTransition(self, s, a):
        """Transition distribution of a state-action pair, as a dense vector.

        Parameters
        ----------
        s : int
            The state.
        a : int
            The action.

        Returns
        -------
        ndarray of shape (nS,)
            Probability of reaching each state. Read by the oracle to solve
            the MDP; not available to a learning agent.
        """
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]] = c[0]
        return transition

    def getMeanReward(self, s, a):
        """Mean reward of a state-action pair.

        Parameters
        ----------
        s : int
            The state.
        a : int
            The action.

        Returns
        -------
        float
            The mean of ``R[s][a]``. Read by the oracle; not available to a
            learning agent.
        """
        rewarddis = self.R[s][a]
        r = rewarddis.mean()
        return r


    def render(self, mode: str = 'human') -> None:
        """Forward the last ``(state, action, reward)`` to every attached renderer.

        Parameters
        ----------
        mode : str, default='human'
            Unused; accepted for :class:`gymnasium.Env` compatibility.
        """
        for re in self.renderers:
            re.render(self,self.last)

    def close(self) -> None:
        """Release every attached renderer at the end of a rendered run."""
        for re in self.renderers:
            re.stop(self)




