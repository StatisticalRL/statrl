

import numpy as np
from src.settings.bandits.stochastic.anytime.environment import StochasticBanditEnv
from src.settings.bandits.stochastic.anytime.agent import BanditAgent
from src.settings.utils import klBern,klGauss

from src.experiments.onerun import Interaction
class BanditInteraction(Interaction):

    def run(self, env:StochasticBanditEnv, learner:BanditAgent, horizon):
        env.reset()
        learner.reset()

        score = []

        for t in range(horizon):
            arm = learner.play()

            reward = env.step(arm)

            learner.update(arm, reward)

            if t > 0:
                score.append(env.expected_reward(arm) + score[t - 1])
            else:
                score.append(env.expected_reward(arm))

        return np.array(score)





if __name__ == "__main__":

    from settings.bandits.stochastic.anytime.envs.parametric import  BernoulliBandit
    from settings.bandits.stochastic.anytime.agents.IMED import IMED
    from settings.bandits.stochastic.anytime.agents._Oracle import Oracle

    means=[0.2,0.9,0.7,0.5]
    env = BernoulliBandit(means)
    agent1 = IMED(len(means))
    oracle = Oracle(env)
    interaction = BanditInteraction()
    scores1=interaction.run(env, agent1, horizon=10)
    print(f"{env.name}:{agent1.name}:{scores1}")
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:{scores0}")


    from experiments.massiveruns import runLargeMulticoreExperiment
    env= [BernoulliBandit, {"means":means}]
    agents=[[IMED, {"nbArms": len(means), "kullback": klBern}], [IMED, {"nbArms": len(means), "kullback": klGauss}]]
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)


