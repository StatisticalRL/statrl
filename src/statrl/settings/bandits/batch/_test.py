from statrl.settings.bandits.batch.envs.parametric import BatchBernoulliBandit
from statrl.settings.bandits.batch.agents._Oracle import Oracle
from statrl.settings.bandits.batch.agents._Random import Random
from statrl.settings.bandits.batch.interaction import BatchBanditInteraction

from statrl.experiments.massiveruns import runLargeMulticoreExperiment




def test_run() -> None:

    means=[0.2,0.9,0.7,0.5]

    env = BatchBernoulliBandit(means,batchschedule="quadratic")
    interaction = BatchBanditInteraction()
    oracle = Oracle(env)

    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:\t{oracle.name}:\t{scores0}")


    random = Random(env)
    scores0=interaction.run(env, random, horizon=10)
    print(f"{env.name}:\t{random.name}:\t{scores0}")




def test_massive() -> None:

    from statrl.settings.bandits.batch.agents.BIMED import BIMED
    from statrl.settings.bandits.batch.agents.BatchIMED import BatchIMED
    means=[0.2,0.9,0.7,0.5]

    env = BatchBernoulliBandit(means,batchschedule="constant")
    interaction = BatchBanditInteraction()
    oracle = Oracle(env)
    agents = [BIMED(env.number_arms),
              BatchIMED(env.number_arms,bound=1,batchagnostic=True),
              BatchIMED(env.number_arms,bound=1,batchagnostic=False)]
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=50,  nbReplicates=30)



if __name__ == "__main__":
    #test_render()
    test_run()
    #test_load()
    test_massive()