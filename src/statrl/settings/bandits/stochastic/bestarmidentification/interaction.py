

import numpy as np
from statrl.settings.bandits.stochastic.bestarmidentification.environment import StochasticBanditEnv
from statrl.settings.bandits.stochastic.bestarmidentification.agent import BAIBanditAgent


from statrl.settings.bandits.stochastic.bestarmidentification.renderers.textrenderer import Textrenderer
from statrl.experiments.onerun import Interaction
class BAIInteraction(Interaction):
    """Interaction loop for the anytime stochastic bandit setting.

    Drives ``select_arm`` -> ``step`` -> ``update`` for a fixed number of
    rounds and returns the cumulative *expected* score.

    See Also
    --------
    statrl.settings.bandits.stochastic.knownhorizon.interaction.BanditInteraction :
        The counterpart that passes the horizon to ``reset``.

    Examples
    --------
    # TODO: update the example.
    #>>> from statrl.settings.bandits.stochastic.anytime.envs.parametric import BernoulliBandit
    #>>> from statrl.settings.bandits.stochastic.anytime.agents.TopTwoTS import TopTwoTS
    #>>> env = BernoulliBandit([0.2, 0.9, 0.5])
    #>>> scores = BAIInteraction().run(env, TopTwoTS(env.number_arms), horizon=100)
    #>>> scores.shape
    (100,)
    """

    def run(self, env: StochasticBanditEnv, learner: BAIBanditAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its best-arm identification score.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance. Reset at the start of the run.
        learner : ~statrl.settings.bandits.stochastic.anytime.agent.BanditAgent
            The agent. Reset at the start of the run, so a single instance can
            be reused across replicates.
        horizon : int
            Maximum number of rounds to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Identifcation loss the *expected* rewards of the arms played, i.e.
            entry ``t`` is :math:`\\sum_{s \\le t} \\mu_{a_s}`. Accumulating
            means rather than realized rewards removes the reward noise from
            the regret curve.
        """
        env.reset()
        learner.reset()

        steps_scores=[]

        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            a = learner.recommend_arm()
            score = 1 if (env.expected_reward(a) >= env.optimal_mean) else 0.

            steps_scores.append(score)
            if (learner.is_done()):
                break

        return np.array(steps_scores)

    def renderrun(self, env: StochasticBanditEnv, learner: BAIBanditAgent, horizon: int) -> None:
        """Run one interaction, printing each round to stdout.

        Same loop as :meth:`run`, but a
        :class:`~statrl.settings.bandits.stochastic.anytime.renderers.textrenderer.Textrenderer`
        is attached to the environment and no score is returned. Intended for
        inspecting short runs by eye, not for benchmarking.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The bandit instance. Its ``renderers`` list is overwritten.
        learner : ~statrl.settings.bandits.stochastic.anytime.agent.BanditAgent
            The agent.
        horizon : int
            Number of rounds to play.
        """
        env.renderers= [Textrenderer()]
        env.reset()
        learner.reset()

        env.render()
        for t in range(horizon):
            arm = learner.select_arm()

            reward = env.step(arm)

            learner.update(arm, reward)

            if (learner.is_done()):
                break

            env.render()

        env.close()

    @property
    def plotlabels(self):
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots."""
        # Remember that we plot the regret, so 1[a' chosen by orcle is opt]-1[a is opt] = 1-1[a is opt] = 1[a is not opt]
        return ("Time step", "1[a is not opt]")






