import math

import numpy as np
import pytest

from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
from statrl.settings.utils import klBern, klGauss


@pytest.mark.parametrize("divergence", [klBern, klGauss])
def test_vectorized_indexes_match_scalar_formula(divergence):
    agent = IMED(4, kullback=divergence)
    agent.reset()

    for arm, reward in [(0, 0.8), (1, 0.2), (2, 0.6), (0, 0.4), (3, 0.1)]:
        agent.update(arm, reward)
        expected = np.array([
            (
                agent.nbDraws[a] * divergence(agent.means[a], agent.maxMeans)
                + math.log(agent.nbDraws[a])
            )
            if agent.nbDraws[a] > 0
            else 0.0
            for a in range(agent.nA)
        ])
        np.testing.assert_allclose(agent.indexes, expected)

