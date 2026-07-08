import sys
import string

from typing import Optional

from settings.bandits.stochastic.anytime.environment import StochasticBanditEnv


class Textrenderer():

    def __init__(self) -> None:
        self.started = False


    def start(self, env: StochasticBanditEnv) -> None:
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.name) + "\n")
        self.outfile.write("Actions: "+ str(self._nameActions(env)) + "\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env: StochasticBanditEnv) -> None:
        self.outfile.write("-"*30+"\n")

    def _nameActions(self, env: StochasticBanditEnv) -> str:
        return string.ascii_uppercase[:env.number_arms]

    def render(self, env: StochasticBanditEnv, last: tuple[Optional[int], float]) -> None:
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        actionNames = self._nameActions(env)
        if lastaction is not None:
            self.outfile.write(f"({actionNames[lastaction % 26]})\tr={lastreward:0.2f}\n")
