from statrl.settings.utils import categorical_sample
import numpy as np
import string

from gymnasium import Env, spaces
from gymnasium.utils import seeding

class DiscreteMDP(Env):
    """
    Parameters
    - nS: number of states
    - nA: number of actions
    - P: transition distributions (*)
    - R: reward distributions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, done), ...]
      R[s][a] == distribution(mean,param)
       One can sample R[s][a] using R[s][a].rvs()
    (**) list or array of length nS


    """


    def __init__(self, nS, nA, P, R, isd, nameActions=[], seed=None, name="DiscreteMDP"):
        self.name=name
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.np_random, seed = seeding.np_random(seed)
        self.s = categorical_sample(self.isd, self.np_random)

        self.last = (self.s,None, 0.)
        return self.s, {"mean": 0}

    def expected_reward(self, state, arm: int) -> float:
        return self.getMeanReward(state,arm)


    def change_rendermode(self,rendermode):
        self.rendermode = rendermode
        self.initializedRenderer = False

    def step(self, a):
        """

        :param a: action
        :return:  (state, reward, IsDone?, IsTruncated?, meanreward)
        The meanreward is returned for information, it should not be given to the learner.
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
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]] = c[0]
        return transition

    def getMeanReward(self, s, a):
        rewarddis = self.R[s][a]
        r = rewarddis.mean()
        return r


    def render(self, mode: str = 'human') -> None:
        for re in self.renderers:
            re.render(self,self.last)

    def close(self) -> None:
        for re in self.renderers:
            re.stop(self)




