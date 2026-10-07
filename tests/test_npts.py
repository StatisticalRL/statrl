import numpy as np

from statrl.settings.bandits.stochastic.anytime.agents.NPTS import NPTS


def test_npts_runs_past_first_round_and_favors_best_arm():
    np.random.seed(0)
    means = [0.2, 0.5, 0.8]
    agent = NPTS(len(means))
    agent.reset()
    for _ in range(2000):
        arm = agent.select_arm()
        assert 0 <= arm < len(means)
        agent.update(arm, float(np.random.rand() < means[arm]))
    assert agent.nbDraws.sum() == 2000
    assert np.argmax(agent.nbDraws) == 2
