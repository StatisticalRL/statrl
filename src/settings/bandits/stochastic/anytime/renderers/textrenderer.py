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

    def _nameActions(self, env: StochasticBanditEnv) -> list[str]:
        return list(string.ascii_uppercase)[0:min(env.number_arms, 26)]

    def render(self, env: StochasticBanditEnv, last: tuple[Optional[int], float]) -> None:
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        actionNames = self._nameActions(env)
        if lastaction:
            self.outfile.write("({})\tr={}\n".format(actionNames[lastaction % 26],str("{:01.2f}".format(lastreward))))
