
from statrl.settings.markovdecisionprocess.agent import MDPAgent
class Random(MDPAgent):
    def __init__(self,env):
        self.env=env
        self.name= "Random"
        super(Random, self).__init__(env.nS, env.nA, self.name)

    def reset(self,inistate):
        pass

    def play(self,state):
        return self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        pass