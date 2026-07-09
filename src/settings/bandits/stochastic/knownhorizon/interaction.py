
import numpy as np

from settings.bandits.stochastic.knownhorizon.environment import StochasticBanditEnv
from settings.bandits.stochastic.knownhorizon.agent import BanditAgent


from settings.bandits.stochastic.anytime.renderers.textrenderer import Textrenderer

from experiments.onerun import Interaction
class BanditInteraction(Interaction):

    def run(self, env:StochasticBanditEnv, learner:BanditAgent, horizon: int) -> np.ndarray:
        env.reset()
        learner.reset(horizon)

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            steps_scores[t] = env.expected_reward(arm)

        return np.cumsum(steps_scores)

    def renderrun(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> None:

        env.renderers.append(Textrenderer())
        env.reset()
        learner.reset(horizon)

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()

        env.close()





if __name__ == "__main__":
    # These are all ANYTIME environments and agents.
    from settings.bandits.stochastic.anytime.envs.parametric import  BernoulliBandit
    from settings.bandits.stochastic.anytime.agents.IMED import IMED
    from settings.bandits.stochastic.anytime.agents._Oracle import Oracle

    from settings.bandits.stochastic.knownhorizon.wrappers.wrapper_anytime_knownhorizon import AnytimeToKnownHorizonAgentWrapper

    means=[0.2,0.9,0.7,0.5]
    nA=len(means)

    env = BernoulliBandit(means) #Anytime environments are compatible with Knownhorizon environment, by knownhorizon.environment.
    agent1 = AnytimeToKnownHorizonAgentWrapper(IMED(nA))
    oracle = AnytimeToKnownHorizonAgentWrapper(Oracle(env))
    interaction = BanditInteraction() #Knownhorizon interaction


    interaction.renderrun(env, agent1, 10)

    scores1=interaction.run(env, agent1, horizon=10)
    print(f"{env.name}:{agent1.name}:\t{scores1}")#Notice the wrapper updated the name of the algorithm.
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:\t{scores0}")



    from experiments.massiveruns import runLargeMulticoreExperiment
    from settings.utils import klBern,klGauss
    env = BernoulliBandit(means)
    agents = [AnytimeToKnownHorizonAgentWrapper(IMED(nA, klBern)),
              AnytimeToKnownHorizonAgentWrapper(IMED(nA, klGauss))]
    oracle = AnytimeToKnownHorizonAgentWrapper(Oracle(env))
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)
