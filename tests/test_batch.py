from statrl.settings.bandits.batch.environment import BatchMAB
from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit


def test_batchmab_reset_and_step() -> None:
    mab = BernoulliBandit([0.2, 0.9, 0.7, 0.5])
    env = BatchMAB(mab, [2, 3])  # round 0 -> 2 pulls, round 1 -> 3 pulls, then 1

    _, info = env.reset()
    assert info["nextbatchsize"] == 2

    _, batchreward, info = env.step([0, 1])
    assert len(batchreward) == 2
    assert info["nextbatchsize"] == 3
    assert "mean" in info

    _, batchreward, info = env.step([0, 1, 2])
    assert len(batchreward) == 3
