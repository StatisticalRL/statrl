

import numpy as np
from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv
from statrl.settings.bandits.stochastic.kernel.agent import KernelBanditAgent


from statrl.settings.bandits.stochastic.kernel.renderers.textrenderer import Textrenderer
from statrl.settings.bandits.stochastic.kernel.renderers.plotrenderer import  PlotRenderer

from statrl.experiments.onerun import Interaction

class KernelBanditInteraction(Interaction):

    def run(self, env: KernelBanditEnv, learner: KernelBanditAgent, horizon: int) -> np.ndarray:
        env.reset()
        learner.reset()

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            steps_scores[t] = env.expected_reward(arm)

        return np.cumsum(steps_scores)

    def renderrun(self, env: KernelBanditEnv, learner: KernelBanditAgent, horizon: int) -> None:

        env.renderers= [Textrenderer(),PlotRenderer()]
        env.reset()
        learner.reset()

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()

        env.close()

    @property
    def plotlabels(self):
        return ("Time step", "Regret")






