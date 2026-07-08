

import numpy as np
from settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
from settings.bandits.stochastic.anytime.agent import BanditAgent


from settings.bandits.stochastic.anytime.renderers.textRenderer import textRenderer
from experiments.onerun import Interaction
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

        env.renderers.append(textRenderer())
        env.reset()
        learner.reset()
        env.render()

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()






if __name__ == "__main__":

    from settings.bandits.stochastic.anytime.envs.parametric import  BernoulliBandit
    from settings.bandits.stochastic.anytime.agents.IMED import IMED
    from settings.bandits.stochastic.anytime.agents._Oracle import Oracle

    means=[0.2,0.9,0.7,0.5]
    nA=len(means)

    env = BernoulliBandit(means)
    agent1 = IMED(nA)
    oracle = Oracle(env)
    interaction = BanditInteraction()

    interaction.renderrun(env, agent1, 10)

    scores1=interaction.run(env, agent1, horizon=10)
    print(f"{env.name}:{agent1.name}:{scores1}")
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:{scores0}")



    from experiments.massiveruns import runLargeMulticoreExperiment
    from settings.utils import klBern,klGauss
    env =BernoulliBandit(means)
    agents = [IMED(nA,klBern),
              IMED(nA,klGauss)]
    oracle = Oracle(env)
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)


