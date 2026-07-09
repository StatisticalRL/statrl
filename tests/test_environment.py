from statrl.settings.bandits.stochastic.anytime.envs.parametric import (
    BernoulliBandit, GaussianBandit,
)


def test_stochastic_env_properties():
    env = BernoulliBandit(means =[0.1, 0.7, 0.3], name="test_env")
    assert env.number_arms == 3
    assert env.means == [0.1, 0.7, 0.3]
    assert env.optimal_mean == 0.7
    assert env.optimal_arm == 1
    assert env.expected_reward(2) == 0.3
    assert env.name == "test_env-means-0.1-0.7-0.3" 


def test_gaussian_bandit_means():
    env = GaussianBandit([1.0, -1.0], vars=[1.0, 2.0])
    assert env.means == [1.0, -1.0]
    assert env.optimal_arm == 0
    
