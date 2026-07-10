



import numpy as np
from statrl.settings.markovdecisionprocess.gridworld.environment import DiscreteMDP
from statrl.settings.markovdecisionprocess.gridworld.agent import  MDPAgent


from statrl.settings.markovdecisionprocess.gridworld.renderers.textRenderer import GridworldRenderer,GridworldWithWallRenderer

from statrl.experiments.onerun import Interaction

class MDPInteraction(Interaction):

    def run(self, env: DiscreteMDP, learner: MDPAgent, horizon: int) -> np.ndarray:
        observation, info = env.reset()
        learner.reset(observation)

        steps_scores = np.empty(horizon)
        for t in range(horizon):
            state = observation
            action = learner.play(state)  # Get action
            observation, reward, done, truncated, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners

            steps_scores[t] = env.expected_reward(state,action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()


        return np.cumsum(steps_scores)

    def renderrun(self, env: DiscreteMDP, learner: MDPAgent, horizon: int) -> None:
        env.renderers= [GridworldRenderer()]
        observation, info = env.reset()
        learner.reset(observation)

        env.render()
        for t in range(horizon):
            state = observation
            action = learner.play(state)  # Get action
            observation, reward, done, truncated, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()

            env.render()

        env.close()


