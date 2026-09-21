
from statrl.settings.bandits.stochastic.kernel.agents._Random import Random
from statrl.settings.bandits.stochastic.kernel.agents._Oracle import Oracle
from statrl.settings.bandits.stochastic.kernel.interaction import KernelBanditInteraction
from statrl.experiments.massiveruns import runLargeMulticoreExperiment


def test_load() -> None:

    from statrl.experiments.utils import load, make
    envs = load("envs/environments.yaml")
    #env = make(envs["rbf_1d_100"])
    env = make(envs["rbf_2d_20"])

    random = Random(env)
    _oracle = Oracle(env)
    interaction = KernelBanditInteraction()

    interaction.renderrun(env, random, 10)



def test_massive() -> None:
    from statrl.settings.bandits.stochastic.kernel.agents.KernelUCB import KernelUCBKnownVariance
    from statrl.settings.bandits.stochastic.kernel.agents.KernelTS import KernelTSKnownVariance
    from statrl.settings.bandits.stochastic.kernel.agents.KernelUCB import KernelUCBUnknownVariance
    from statrl.settings.bandits.stochastic.kernel.agents.KernelTS import KernelTSUnknownVariance

    from statrl.experiments.utils import load, make
    envs = load("envs/environments.yaml")
    env = make(envs["rbf_1d_100"])

    interaction = KernelBanditInteraction()

    agents = [Random(env),
              KernelUCBKnownVariance(env,sigma=env.noise_std),
              KernelTSKnownVariance(env,sigma=env.noise_std),
              KernelUCBUnknownVariance(env),
              KernelTSUnknownVariance(env),
              ]
    oracle = Oracle(env)
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=100,  nbReplicates=10)


if __name__ == "__main__":
    #test_render()
    #test_run()
    #test_load()
    test_massive()