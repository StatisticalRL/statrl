def keyboard_waitfor(list):
    while True:
        x = input()
        if x in list:
            i = list.index(x)
            return x,i

from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
class Human(MDPAgent):
    def __init__(self,env):
        self.env=env.env
        self.name="Human"
        super(Human, self).__init__(env.nS, env.nA, self.name)

    def reset(self,inistate):
        pass

    def play(self,state):
        print("Current state is: ",state)
        print("Please choose an action in the following list (confirm with entry): ", self.env.nameActions)
        (name_action,action) = keyboard_waitfor(list(self.env.nameActions))
        print("Chosen action is: ",name_action)#, "(number ",action,")")

        return action

    def update(self, state, action, reward, observation):
        pass