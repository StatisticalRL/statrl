
import numpy as np

from statrl.settings.bandits.stochastic.knownhorizon.environment import StochasticBanditEnv
from statrl.settings.bandits.stochastic.knownhorizon.agent import BanditAgent


from statrl.settings.bandits.stochastic.anytime.renderers.textrenderer import Textrenderer

from statrl.experiments.onerun import Interaction
class BanditInteraction(Interaction):
    """Interaction loop for the known-horizon stochastic bandit setting.

    Identical to the anytime loop except that the horizon is passed to
    ``learner.reset(horizon)`` instead of being withheld.

    See Also
    --------
    statrl.settings.bandits.stochastic.anytime.interaction.BanditInteraction :
        The anytime counterpart.

    Examples
    --------
    >>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    >>> from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
    >>> from statrl.settings.bandits.stochastic.knownhorizon.wrappers.wrapper_anytime_knownhorizon import (
    ...     AnytimeToKnownHorizonAgentWrapper)
    >>> env = BernoulliBandit([0.2, 0.9, 0.5])
    >>> agent = AnytimeToKnownHorizonAgentWrapper(IMED(env.number_arms))
    >>> BanditInteraction().run(env, agent, horizon=50).shape
    (50,)
    """

    def run(self, env:StochasticBanditEnv, learner:BanditAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance; the same class as in the anytime setting,
            since only the agent interface differs between the two.
        learner : ~statrl.settings.bandits.stochastic.knownhorizon.agent.BanditAgent
            The agent, reset with ``horizon`` so it can plan against it.
        horizon : int
            Number of rounds to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative sum of the *expected* rewards of the arms played.
        """
        env.reset()
        learner.reset(horizon)

        steps_scores = np.empty(horizon)

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            steps_scores[t] = env.expected_reward(arm)

        return np.cumsum(steps_scores)

    def renderrun(self, env: StochasticBanditEnv, learner: BanditAgent, horizon: int) -> None:
        """Run one interaction, printing each round to stdout.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance. A
            :class:`~statrl.settings.bandits.stochastic.anytime.renderers.textrenderer.Textrenderer`
            is *appended* to its ``renderers``, so calling this twice on the
            same environment prints every round twice.
        learner : ~statrl.settings.bandits.stochastic.knownhorizon.agent.BanditAgent
            The agent.
        horizon : int
            Number of rounds to play.
        """
        env.renderers.append(Textrenderer())
        env.reset()
        learner.reset(horizon)

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            env.render()

        env.close()


    @property
    def plotlabels(self):
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots."""
        return ("Time step", "Regret")





if __name__ == "__main__":
    # These are all ANYTIME environments and agents.
    from statrl.settings.bandits.stochastic.anytime.envs.parametric import  BernoulliBandit
    from statrl.settings.bandits.stochastic.anytime.agents.IMED import IMED
    from statrl.settings.bandits.stochastic.anytime.agents.Oracle import Oracle

    from statrl.settings.bandits.stochastic.knownhorizon.wrappers.wrapper_anytime_knownhorizon import AnytimeToKnownHorizonAgentWrapper

    means=[0.2,0.9,0.7,0.5]
    nA=len(means)

    env = BernoulliBandit(means) #Anytime environments are compatible with Knownhorizon environment, by knownhorizon.environment.
    agent1 = AnytimeToKnownHorizonAgentWrapper(IMED(nA))
    oracle = AnytimeToKnownHorizonAgentWrapper(Oracle(env))
    interaction = BanditInteraction() #Knownhorizon interaction


    interaction.renderrun(env, agent1, 10)

    scores1=interaction.run(env, agent1, horizon=10)
    print(f"{env.name}:{agent1.name}:\t{scores1}")#Notice the wrapper updated the name of the algorithm.
    scores0=interaction.run(env, oracle, horizon=10)
    print(f"{env.name}:{oracle.name}:\t{scores0}")



    from statrl.experiments.massiveruns import runLargeMulticoreExperiment
    from statrl.settings.utils import klBern,klGauss
    env = BernoulliBandit(means)
    agents = [AnytimeToKnownHorizonAgentWrapper(IMED(nA, klBern)),
              AnytimeToKnownHorizonAgentWrapper(IMED(nA, klGauss))]
    oracle = AnytimeToKnownHorizonAgentWrapper(Oracle(env))
    runLargeMulticoreExperiment(env,agents,oracle, interaction,timeHorizon=1000,  nbReplicates=50)
