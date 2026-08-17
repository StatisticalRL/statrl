from statrl.settings.bandits.stochastic.kernel.envs.kernels import KernelBandit

from statrl.settings.bandits.stochastic.kernel.agents._Random import Random
from statrl.settings.bandits.stochastic.kernel.agents._Oracle import Oracle
from statrl.settings.bandits.stochastic.kernel.interaction import KernelBanditInteraction
from statrl.experiments.massiveruns import runLargeMulticoreExperiment


def test_load() -> None:

    from statrl.experiments.utils import load, make
    envs = load("envs/environments.yaml")
    env = make(envs["rbf_1d_100"])
    #env = make(envs["rbf_2d_20"])

    random = Random(env)
    oracle = Oracle(env)
    interaction = KernelBanditInteraction()

    interaction.renderrun(env, random, 10)




if __name__ == "__main__":
    #test_render()
    #test_run()
    test_load()
    #test_massive()