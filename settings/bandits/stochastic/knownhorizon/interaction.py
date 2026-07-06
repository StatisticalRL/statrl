
import numpy as np


def interact(env: StochasticBanditEnv,
             learner: BanditAgent,
             horizon: int,
             initialize_score=lambda x: 0,
             compute_score=lambda a, r: r
             ) -> np.ndarray:
    env.reset()

    learner.reset(horizon)

    score = initialize_score(horizon)

    for t in range(horizon):
        arm = learner.select_arm()

        reward = env.pull(arm)

        learner.update(arm, reward)

        score[t] = compute_score(arm, reward)

    return score