
import statrl.settings.bandits.stochastic.anytime.environment as envA
import statrl.settings.bandits.stochastic.anytime.agent as agentA
import statrl.settings.bandits.stochastic.knownhorizon.environment as envK
import statrl.settings.bandits.stochastic.knownhorizon.agent as agentK


class AnytimeToKnownHorizonEnvironmentWrapper(
    envA.StochasticBanditEnv
):
    """
    View a known-horizon environment as an anytime environment.
    """

    def __init__(self, env):
        self.env = env

    @property
    def n_arms(self):
        return self.env.n_arms

    @property
    def means(self):
        return self.env.means

    def pull(self, arm):
        return self.env.pull(arm)

    def reset(self, seed=None):
        return self.env.reset(seed)


class KnownHorizonToAnytimeEnvironmentWrapper(
    envK.StochasticBanditEnv
):
    """
    View an anytime  environment as a known-horizon  environment.
    """

    def __init__(self, env):
        self.env = env

    @property
    def n_arms(self):
        return self.env.n_arms

    @property
    def means(self):
        return self.env.means

    def pull(self, arm):
        return self.env.pull(arm)

    def reset(self, seed=None):
        return self.env.reset(seed)

class AnytimeToKnownHorizonAgentWrapper(
        agentK.BanditAgent):

    def __init__(self, learner):
        self.learner = learner

    def reset(self, horizon):
        self.learner.reset()

    def select_arm(self):
        return self.learner.select_arm()

    def update(self, arm, reward):
        self.learner.update(arm, reward)


class KnownHorizonToAnytimeAgentWrapper(
        agentA.BanditAgent):

    def __init__(self, learner,horizon):
        """
        Fixes an internal horizon at initilization.
        """
        self.learner = learner
        self.horizon= horizon

    def reset(self):
        self.learner.reset(self.horizon)

    def select_arm(self):
        return self.learner.select_arm()

    def update(self, arm, reward):
        self.learner.update(arm, reward)