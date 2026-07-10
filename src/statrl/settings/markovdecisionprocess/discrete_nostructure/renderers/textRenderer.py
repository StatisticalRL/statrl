
import sys
from six import StringIO
from gymnasium import utils
import string

class TextRenderer:

    def __init__(self):
        self.started = False

    def start(self, env) -> None:
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.name) + "\n")
        self.outfile.write("Actions: "+ str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, Blue=possible next states\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env) -> None:
        self.outfile.write("-"*30+"\n")

    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]


    def render(self,env,last):

        current, lastaction, lastreward = last
        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)

        desc = [str(s) for s in env.states]

        desc[current] = utils.colorize(desc[current], "red", highlight=True)
        for a in env.actions:
            for ssl in env.P[current][a]:
                if (ssl[0] > 0):
                    desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)


        actionNames = self._nameActions(env)
        #print(f"\t{current},{lastaction},{lastreward}")
        if lastaction is not None:
            self.outfile.write(f"({actionNames[lastaction % 26]})\tr={lastreward:0.2f}\t")
            self.outfile.write("".join(desc))
            self.outfile.write("\n")
        else:
            self.outfile.write("\t\t\t")
            self.outfile.write("".join(desc))
            self.outfile.write("\n")