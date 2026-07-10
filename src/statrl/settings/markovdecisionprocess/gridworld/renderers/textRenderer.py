
import sys
from six import StringIO
from gymnasium import utils
import string

class GridworldWithWallRenderer:

    def __init__(self):
        self.started = False

    def start(self, env) -> None:
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.name) + "\n")
        self.outfile.write("Actions: " + str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, X=wall, G=goal state\n")
        self.outfile.write("-" * 30 + "\n")

    def stop(self, env) -> None:
        self.outfile.write("-" * 30 + "\n")


    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]

    def render(self,env,current,lastaction,lastreward):

        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        #outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(current)
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            self.outfile.write("  ({})\n".format(env.nameActions[lastaction]))
        else:
            self.outfile.write("\n")
        self.outfile.write("\n".join(''.join(line) for line in desc) + "\n")





class GridworldRenderer:
    def __init__(self):
        self.started = False

    def start(self, env) -> None:
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.name) + "\n")
        self.outfile.write("Actions: " + str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, X=wall, G=goal state\n")
        self.outfile.write("-" * 30 + "\n")

    def stop(self, env) -> None:
        self.outfile.write("-" * 30 + "\n")


    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]

    def render(self,env,current,lastaction,lastreward):

        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        #outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(env.mapping[current])
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)


        #desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            self.outfile.write("\t({})\tr={}\n\n".format(env.nameActions[lastaction],str(lastreward)))
        else:
            self.outfile.write(" \n")

        self.outfile.write("\n".join(''.join(line) for line in desc) + "\n")


