
import numpy as np

from src.settings.bandits.stochastic.knownhorizon.environment import StochasticBanditEnv
from src.settings.bandits.stochastic.knownhorizon.agent import BanditAgent




from src.experiments import Interaction
class BanditInteraction(Interaction):

    def run(self, env:StochasticBanditEnv, learner:BanditAgent, horizon):
        env.reset()
        learner.reset(horizon)

        score = []

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.pull(arm)

            learner.update(arm, reward)

            if t > 0:
                score.append(env.expected_reward(arm) + score[t - 1])
            else:
                score.append(env.expected_reward(arm))

        return np.array(score)
