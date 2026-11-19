import sys
import string

from typing import Optional

from statrl.settings.bandits.stochastic.bestarmidentification.environment import StochasticBanditEnv


class Textrenderer():
    """Print each bandit round to stdout, one line per pull.

    Attributes
    ----------
    started : bool
        Whether the header has been printed yet. The header is emitted lazily
        on the first :meth:`render` call.
    """

    def __init__(self) -> None:
        self.started = False

    def start(self, env: StochasticBanditEnv) -> None:
        """Print the header naming the environment and its arms.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The environment being rendered.
        """
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.displayname) + "\n")
        self.outfile.write("Actions: " + str(self._nameActions(env)) + "\n")
        self.outfile.write("-" * 30 + "\n")

    def stop(self, env: StochasticBanditEnv) -> None:
        """Print the closing rule at the end of a rendered run.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The environment being rendered.
        """
        self.outfile.write("-" * 30 + "\n")

    def _nameActions(self, env: StochasticBanditEnv) -> str:
        return string.ascii_uppercase[:env.number_arms]

    def render(self, env: StochasticBanditEnv, last: tuple[Optional[int], float]) -> None:
        """Print the arm just played and the reward it returned.

        Parameters
        ----------
        env : ~statrl.settings.bandits.stochastic.anytime.environment.StochasticBanditEnv
            The environment being rendered.
        last : tuple of (int or None, float)
            The ``(arm, reward)`` pair recorded by the environment. A ``None``
            arm means no pull has happened yet, and nothing is printed.
        """
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        actionNames = self._nameActions(env)
        if lastaction is not None:
            suffix = "'" * int(lastaction % 26)
            self.outfile.write(f"({actionNames[lastaction % 26] + suffix})\tr={lastreward:0.2f}\n")
