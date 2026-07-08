
import settings.bandits.stochastic.anytime.environment as envA
import settings.bandits.stochastic.knownhorizon.environment as envK
#" Environment interface are the same, only agents interface are different"
import settings.bandits.stochastic.anytime.agent as agentA
import settings.bandits.stochastic.knownhorizon.agent as agentK

class AnytimeToKnownHorizonAgentWrapper(
        agentK.BanditAgent):

    def __init__(self, learner):
        self.learner = learner
        super().__init__(self.learner.name+"-anytime")

    def reset(self, horizon):
        self.learner.reset()

    def select_arm(self):
        return self.learner.select_arm()

    def update(self, arm, reward):
        self.learner.update(arm, reward)


    @property
    def policy(self):
        return self.learner.policy


class KnownHorizonToAnytimeAgentWrapper(
        agentA.BanditAgent):

    def __init__(self, learner, horizon, name: str):
        """
        Fixes an internal horizon at initilization.
        """
        self.learner = learner
        self.horizon= horizon
        super().__init__(name)

    def reset(self):
        self.learner.reset(self.horizon)

    def select_arm(self):
        return self.learner.select_arm()

    def update(self, arm, reward):
        self.learner.update(arm, reward)

    @property
    def policy(self):
        return self.learner.policy


