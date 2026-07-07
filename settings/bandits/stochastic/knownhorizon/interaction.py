
import numpy as np

from statrl.settings.bandits.stochastic.knownhorizon.environment import StochasticBanditEnv
from statrl.settings.bandits.stochastic.knownhorizon.agent import BanditAgent


def interact(env: StochasticBanditEnv,
    learner: BanditAgent,
    horizon: int
) -> np.ndarray:
    env.reset()

    learner.reset()

    score = []

    for t in range(horizon):
        arm = learner.select_arm()

        _, reward, _, _, _ = env.pull(arm)

        learner.update(arm, reward)

        if t>0:
            score.append(reward+score[t-1])
        else:
            score.append(reward)

    return np.array(score)