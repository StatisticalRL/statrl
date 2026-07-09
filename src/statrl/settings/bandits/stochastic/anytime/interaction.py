

import numpy as np
from statrl.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
from statrl.settings.bandits.stochastic.anytime.agent import BanditAgent


from statrl.settings.bandits.stochastic.anytime.renderers.textrenderer import Textrenderer
from statrl.experiments.onerun import Interaction
class BanditInteraction(Interaction):

    def run(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> np.ndarray:
        env.reset()
        learner.reset()

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            steps_scores[t] = env.expected_reward(arm)

        return np.cumsum(steps_scores)

    def renderrun(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> None:

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







