

import numpy as np
from src.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
from src.settings.bandits.stochastic.anytime.agent import BanditAgent


from src.settings.bandits.stochastic.anytime.renderers.textrenderer import Textrenderer
from src.experiments.onerun import Interaction
class BanditInteraction(Interaction):

    def run(self, env:StochasticBanditEnv, learner:BanditAgent, horizon):
        env.reset()
        learner.reset()

        score = []

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            if t > 0:
                score.append(env.expected_reward(arm) + score[t - 1])
            else:
                score.append(env.expected_reward(arm))

        return np.array(score)

    def renderrun(self, env: StochasticBanditEnv, learner: BanditAgent, horizon):

        env.renderers= [Textrenderer()]
        env.reset()
        learner.reset()

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()

        env.close()







