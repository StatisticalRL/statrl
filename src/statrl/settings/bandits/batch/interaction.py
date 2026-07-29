

import numpy as np
from statrl.settings.bandits.batch.environment import BatchMAB
from statrl.settings.bandits.batch.agent import BatchBanditAgent

from statrl.experiments.onerun import Interaction

class BatchBanditInteraction(Interaction):

    def run(self, env: BatchMAB, learner: BatchBanditAgent, horizon: int) -> np.ndarray:
        observation, info = env.reset()
        learner.reset()
        B = info["nextbatchsize"]

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            batchaction = learner.batchplay(B)  # Get action

            batchreward, info = env.step(batchaction)  # Get response
            learner.batchupdate(batchaction, batchreward)  # Update learners

            B = info["nextbatchsize"]
            steps_scores[t] = info["mean"]

        return np.cumsum(steps_scores)

    def renderrun(self, env: BatchMAB, learner: BatchBanditAgent, horizon: int) -> None:

        env.renderers= []
        observation, info = env.reset()
        learner.reset()
        B = info["nextbatchsize"]

        env.render()
        for t in range(horizon):
            batchaction = learner.batchplay(B)  # Get action

            batchreward, info = env.step(batchaction)  # Get response
            learner.batchupdate(batchaction, batchreward)  # Update learners

            B = info["nextbatchsize"]
            env.render()

        env.close()