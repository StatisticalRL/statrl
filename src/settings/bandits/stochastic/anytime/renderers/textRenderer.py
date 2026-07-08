import sys
import string


class textRenderer():

    def __init__(self):
        self.initializedRender = False


    def initRender(self, env):
        outfile = sys.stdout
        outfile.write("Environment: " + str(env.name) + "\n")
        outfile.write("Actions: "+ str(self._nameActions(env)) + "\n")

    def _nameActions(self,env):
        return list(string.ascii_uppercase)[0:min(env.number_arms, 26)]

    def render(self,env,last):
        lastaction, lastreward = last

        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout



        #desc.append(" \t\tr=" + str(lastreward))
        actionNames = self._nameActions(env)

        if lastaction is not None:
            outfile.write("({})\tr={}\n".format(actionNames[lastaction % 26],str("{:01.2f}".format(lastreward))))
        else:
            outfile.write("")
        #desc = ["."]
        #outfile.write("".join(''.join(line) for line in desc) + "\t")

        #if mode != 'text':
        #    return outfile