from settings.multiarmedbandit.stochastic.anytime.envs.parametric import BernoulliBandit
from settings.multiarmedbandit.stochastic.anytime.agents._Random import Random
from settings.multiarmedbandit.stochastic.anytime.agents._Oracle import Oracle
from settings.multiarmedbandit.stochastic.anytime.interaction import BanditInteraction
from experiments.massiveruns import runLargeMulticoreExperiment


def test_render():

    means=[0.2,0.9,0.7,0.5]

    env = BernoulliBandit(means)
    random = Random(env)
    oracle = Oracle(env)
    interaction = BanditInteraction()

    interaction.renderrun(env, oracle, 10)
    interaction.renderrun(env, random, 10)

def test_load():

    from experiments.utils import load, make
    envs = load("envs/environments.yaml")
    env = make(envs["bernoulli_hard"])

    random = Random(env)
    interaction = BanditInteraction()

    interaction.renderrun(env, random, 10)

def test_run():

    means=[0.2,0.9,0.7,0.5]

    env = BernoulliBandit(means)
    interaction = BanditInteraction()
    random = Random(env)
    oracle = Oracle(env)

    scores1=interaction.run(env, random, horizon=10)
    print(f"{env.name}:{random.name}: \t{scores1}")
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:\t{scores0}")


def test_massive():
    from settings.multiarmedbandit.stochastic.anytime.agents.IMED import IMED
    from src.settings.utils import klBern,klGauss

    means=[0.2,0.9,0.7,0.5]
    nA=len(means)

    env =BernoulliBandit(means)
    interaction = BanditInteraction()
    agents = [IMED(nA,klBern),
              IMED(nA,klGauss)]
    oracle = Oracle(env)
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)


if __name__ == "__main__":
    test_render()
    test_run()
    test_load()
    test_massive()
