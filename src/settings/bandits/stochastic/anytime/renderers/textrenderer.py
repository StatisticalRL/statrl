import sys
import string


class Textrenderer():

    def __init__(self):
        self.started = False


    def start(self, env):
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.name) + "\n")
        self.outfile.write("Actions: "+ str(self._nameActions(env)) + "\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env):
        self.outfile.write("-"*30+"\n")

    def _nameActions(self,env):
        return list(string.ascii_uppercase)[0:min(env.number_arms, 26)]

    def render(self,env,last):
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        #outfile = StringIO() if mode == 'ansi' else sys.stdout



        #desc.append(" \t\tr=" + str(lastreward))
        actionNames = self._nameActions(env)

        if lastaction:
            self.outfile.write("({})\tr={}\n".format(actionNames[lastaction % 26],str("{:01.2f}".format(lastreward))))
        else:
            self.outfile.write("")
        #desc = ["."]
        #outfile.write("".join(''.join(line) for line in desc) + "\t")

        #if mode != 'text':
        #    return outfile