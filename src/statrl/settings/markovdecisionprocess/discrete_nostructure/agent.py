import numpy as np
from gymnasium.utils import seeding

class MDPAgent:
    def __init__(self, nS, nA, name="Agent",seed=None):
        self.nS = nS
        self.nA = nA
        self.name= name
        self.seed = seed


    def reset(self,inistate) -> None:
        """Initialize a new independent run."""
        self.np_random, self.seed = seeding.np_random(self.seed)


    def play(self,state):
        return np.random.randint(self.nA)

    def update(self, state, action, reward, observation):
        ()