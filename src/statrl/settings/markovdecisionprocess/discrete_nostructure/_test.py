from statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim import RiverSwim
from statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Random import Random
from statrl.settings.markovdecisionprocess.discrete_nostructure.agents._Oracle import Opti_swimmer as Oracle
from statrl.settings.markovdecisionprocess.discrete_nostructure.interaction import MDPInteraction
from statrl.experiments.massiveruns import runLargeMulticoreExperiment


def test_render() -> None:

    nS=6
    env = RiverSwim(nS)
    random = Random(env)
    oracle = Oracle(env)
    interaction = MDPInteraction()

    interaction.renderrun(env, oracle, 10)
    interaction.renderrun(env, random, 10)

def test_load() -> None:

    from statrl.experiments.utils import load, make
    envs = load("envs/environments.yaml")
    env = make(envs["riverswim_6"])

    random = Random(env)
    interaction = MDPInteraction()

    interaction.renderrun(env, random, 10)

def test_run() -> None:
    nS=6

    env = RiverSwim(nS)
    interaction = MDPInteraction()
    random = Random(env)
    #agent= IMEDRL(env.nS, env.nA)
    oracle = Oracle(env)

    scores1=interaction.run(env, random, horizon=10)
    print(f"{env.name}:{random.name}: \t{scores1}")
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:\t{scores0}")


def test_massive() -> None:
    from statrl.settings.markovdecisionprocess.discrete_nostructure.agents.IMED_RL import IMEDRL
    from statrl.settings.markovdecisionprocess.discrete_nostructure.agents.PSRL import PSRL

    nS=6
    env =RiverSwim(nS)
    nA=env.nA

    interaction = MDPInteraction()
    agents = [IMEDRL(nS, nA),PSRL(nS, nA)]
    oracle = Oracle(env)
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)


if __name__ == "__main__":
    test_render()
    test_run()
    test_load()
    test_massive()
