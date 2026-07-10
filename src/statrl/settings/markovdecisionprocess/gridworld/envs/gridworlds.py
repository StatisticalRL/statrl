

###  The following is simply copoy paste+debuged to compile.
###  But Gridworld MDPs are specifically structured MDPs:  We can a specific rendering function, and we my have specific algos too.
###  So I suggest we make a new settings dedicated to gridworldMDPs.


import numpy as np
#import sys
#from six import StringIO, b
import scipy.stats as stat
#import matplotlib.pyplot as plt

from statrl.settings.markovdecisionprocess.discrete_nostructure.environment import  DiscreteMDP

#from gym import utils
#from gym.src.toy_text import discrete
#import src.MDPs_discrete.gymWrapper
from gymnasium import spaces
from gymnasium.utils import seeding
#import string
from statrl.settings.utils import categorical_sample, Dirac



# Maze maps used to build grid-worlds MDPs:

def randomMap(sizeX, sizeY, density, lengthofwalks, np_random=np.random):
    maze = np.ones((sizeX, sizeY))
    s = [np_random.integers(sizeX), np_random.integers(sizeY)]
    for i in range((int)(density * sizeX * sizeY)):
        p = np.exp(-np.log( 2) / lengthofwalks)  # probability p to continue building the current wall, 1-p  to start a new one.
        b = np_random.binomial(1, p)
        if (b == 0):
            next = np_random.integers(4)
            if (next == 0):     s = [(s[0] + 1) % sizeX, s[1]]
            if (next == 1):     s = [(s[0] - 1) % sizeX, s[1]]
            if (next == 2):     s = [s[0], (s[1] + 1) % sizeY]
            if (next == 3):     s = [s[0], (s[1] - 1) % sizeY]
        else:
            s = [np_random.integers(sizeX), np_random.integers(sizeY)]
        maze[s[0]][s[1]] = 0.
    return maze

def fourRoomMap(X, Y):
    Y2 = (int) (Y/2)
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
        maze[x][Y2] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
        maze[X2][(int) (Y2/2)] = 1.
        maze[X2][(int) (3*Y2/2)] = 1.
        maze[(int) (X2/2)][Y2] = 1.
        maze[(int) (3*X2/2)][Y2] = 1.
    return maze

def twoRoomMap(X, Y):
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
    maze[X2][ (int) (Y/2)] = 1.
    return maze


class GridWorldWithWall(DiscreteMDP):
    """


    """

#    metadata = {'render.modes': ['text', 'pylab', 'maze'], 'maps': ['random','2-room', '4-room']}

    def __init__(self, sizeX,sizeY, map_name="random", slippery=0.1,nbGoals=1,rewardStd=0.,density=0.2, lengthofwalks=5, initialSingleStateDistribution=False,start=None, goal=None, seed=None,name="GridWorldWithWall"):
        """

        :param sizeX: length of the 2-d grid
        :param sizeY: height of the 2-d grid
        :param map_name: random, 2-room or 4-room
        :param slippery: real-value in [0,1], makes transitions more (1) or less (0) stochastic.
        :param nbGoals: number og goal states to be generated
        :param rewardStd: standard deviation of rewards.
        :param density: density of walls (for random map)
        :param lengthofwalks: average lengh of walls (for random map)
        :param initialSingleStateDistribution: If set to True, the initial distribution is a Dirac at one state, chosen uniformly randomly amongts valid non-goal states, If set to False, initial Distribution is uniform random amongst non-goal states.
        :param seed:
        """

        self.sizeX, self.sizeY = sizeX, sizeY
        self.reward_range = (0, 1)
        self.rewardStd=rewardStd

        self.nA = 4
        self.nS = sizeX * sizeY
        self.nameActions= ["Up", "Down", "Left", "Right"]

        self.seed(seed)

        #stochastic transitions
        slip=min(slippery,1./3.)

        self.massmap = [[slip, 1.-3*slip, slip, 0., slip], # up : left up right  down stay
                   [slip, 0., slip, 1.-3*slip, slip],  # down : left up down right stay
                   [1.-3*slip, slip, 0., slip, slip],  # left : left up right down stay
                   [0., slip, 1.-3*slip, slip, slip]]  # right : left up right down stay


        if (map_name=="2-room"):
            self.maze=twoRoomMap(sizeX, sizeY)
        elif (map_name=="4-room"):
            self.maze = fourRoomMap(sizeX, sizeY)
        else:
            self.maze = randomMap(sizeX, sizeY, density, lengthofwalks, np_random=self.np_random)



        if (goal != None):
            self.goalstates = self.makeGoalState(xy = goal)
        else:
            self.goalstates = self.makeGoalStates(nbGoals)
        if (initialSingleStateDistribution):
            isd = self.makeInitialSingleStateDistribution(self.maze,xy=start)#start = [1,1]
        else:
            isd = self.makeInitialDistribution(self.maze)

        P = self.makeTransition(isd)
        R = self.makeRewards()


        super(GridWorldWithWall, self).__init__(self.nS, self.nA, P, R, isd, nameActions=self.nameActions, seed=None,name=name)
        #self.renderers['gw-pyplot'] = gwppRendering.GridworldWithWallRenderer
        #self.renderers['gw-text'] = gwtRendering.GridworldWithWallRenderer
        #self.rendermode='gw-text'

    def to_s(self,rowcol):
            return rowcol[0] * self.sizeY + rowcol[1]

    def from_s(self,s):
            return s//self.sizeY, s%self.sizeY

    def makeGoalStates(self, nb):
        goalstates = []
        for g in range(nb):
            s = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
            while (self.maze[s[0]][s[1]] == 0):
                s = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
            goalstates.append(self.to_s(s))
            self.maze[s[0]][s[1]] = 2.
        return goalstates



    def makeGoalState(self, xy=None):
        goalstates = []
        if (xy == None):
            xy = [np.random.integers(self.sizeX), np.random.integers(self.sizeY)]
            while (self.maze[xy[0]][xy[1]] != 1):
                xy = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
        goalstates.append(self.to_s(xy))
        self.maze[xy[0]][xy[1]] = 2.
        return goalstates


    def makeInitialSingleStateDistribution(self, maze,xy=None):
        if (xy==None):
            xy =[np.random.integers(self.sizeX), np.random.integers(self.sizeY)]
            while (self.maze[xy[0]][xy[1]] != 1):
                xy = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
        isd = np.zeros(self.nS)
        isd[self.to_s(xy)] = 1.
        return isd


    def makeInitialDistribution(self,maze):
         isd = np.array(maze == 1.).astype('float64').ravel()
         isd /= isd.sum()
         return isd

    def makeTransition(self,initialstatedistribution):
            X = self.sizeX
            Y = self.sizeY
            P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
            nbempty=0

            for s in range(self.nS):
                x,y = self.from_s(s)
                if (self.maze[x][y] == 2.):
                    for a in range(self.nA):
                        li = P[s][a]
                        for ns in range(self.nS):
                            if(initialstatedistribution[ns] > 0):
                                li.append((initialstatedistribution[ns],ns,False))
                else:
                    us = [(x - 1) % X, y % Y]
                    ds = [(x + 1) % X, y % Y]
                    ls = [x % X, (y - 1) % Y]
                    rs = [x % X, (y + 1) % Y]
                    ss=[x,y]
                    if (self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0): us = ss
                    if (self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0): ds = ss
                    if (self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0): ls = ss
                    if (self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0): rs = ss

                    for a in range(self.nA):
                        li = P[s][a]
                        li.append((self.massmap[a][0],self.to_s(ls),False))
                        li.append((self.massmap[a][1],self.to_s(us),False))
                        li.append((self.massmap[a][2],self.to_s(rs),False))
                        li.append((self.massmap[a][3],self.to_s(ds),False))
                        li.append((self.massmap[a][4],self.to_s(ss),False))

            return P

    def makeRewards(self):
        R = {s: {a: Dirac(0.) for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(s)
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    mymean=0.99
                    if (self.rewardStd > 0):
                        ma, mb = (0 - mymean) / self.rewardStd, (1 - mymean) / self.rewardStd
                        R[s][a]= stat.truncnorm(ma, mb, loc=mymean, scale=self.rewardStd)
                    else:
                        R[s][a] = Dirac(mymean)
        return R

    def getTransition(self,s,a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]]=c[0]
        return transition



# Upgrade of the previous class, walls are no longer visible as unaccessible states for the learner (they're no longer existing for the learner).
class GridWorld(DiscreteMDP):
 #   metadata = {'render.modes': ['text', 'ansi', 'pylab', 'maze'], 'maps': ['random', '2-room', '4-room']}

    def __init__(self, sizeX, sizeY, map_name="random", slippery=0.1, nbGoals=1, rewardStd=0., density=0.2,
                 lengthofwalks=5, initialSingleStateDistribution=False,start=None, goal=None,seed=None,name="GridWorld"):
        """

        :param sizeX: length of the 2-d grid
        :param sizeY: height of the 2-d grid
        :param map_name: random, 2-room or 4-room
        :param slippery: real-value in [0,1], makes transitions more (1) or less (0) stochastic.
        :param nbGoals: number og goal states to be generated
        :param rewardStd: standard deviation of rewards.
        :param density: density of walls (for random map)
        :param lengthofwalks: average lengh of walls (for random map)
        :param initialSingleStateDistribution: True: the initial distribution is a Dirac at one state, chosen uniformly randomly amongts valid non-goal states; False: initial Distribution is uniform random amongst non-goal states.
        :param seed:
        """

        # desc = maps[map_name]
        self.sizeX, self.sizeY = sizeX, sizeY
        self.reward_range = (0, 1)
        self.rewardStd = rewardStd
        self.map_name=map_name

        self.nA = 4
        self.nS_all = sizeX * sizeY
        self.nameActions = ["Up", "Down", "Left", "Right"]


        self.seed(seed)

        # stochastic transitions
        slip = min(slippery, 1. / 3.)
        self.massmap = [[slip, 1. - 3 * slip, slip, 0., slip],  # up : up down left right stay
                        [slip, 0., slip, 1. - 3 * slip, slip],  # down
                        [1. - 3 * slip, slip, 0., slip, slip],  # left
                        [0., slip, 1. - 3 * slip, slip, slip]]  # right

        if (map_name=="2-room"):
            self.maze=twoRoomMap(sizeX, sizeY)
        elif (map_name=="4-room"):
            self.maze = fourRoomMap(sizeX, sizeY)
        else:
            self.maze = randomMap(sizeX, sizeY, density, lengthofwalks, np_random=self.np_random)

        self.mapping = []
        self.revmapping = []#np.zeros(sizeX*sizeY)
        cpt=0
        for x in range(sizeX):
            for y in range(sizeY):
                #xy = self.to_s((x, y))
                xy = x * self.sizeY + y
                if self.maze[x, y] >= 1:
                    self.mapping.append(xy)
                    self.revmapping.append((int) (cpt))
                    cpt=cpt+1
                else:
                    self.revmapping.append((int) (-1))

        #print(self.revmapping)
        self.nS = len(self.mapping)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        if (goal != None):
            self.goalstates = self.makeGoalState(xy = goal)
        #if (map_name == "2-room"):
        #    self.goalstates = self.makeGoalState(xy = [sizeX - 2, sizeY - 2])
        #elif (map_name=="4-room"):
        #    self.goalstates = self.makeGoalState(xy= [sizeX-2,sizeY-2])
        else:
            self.goalstates = self.makeGoalStates(nbGoals)
        if (initialSingleStateDistribution):
            isd = self.makeInitialSingleStateDistribution(self.maze,xy=start)#start = [1,1]
        else:
            isd = self.makeInitialDistribution(self.maze)
        P = self.makeTransition(isd)
        R = self.makeRewards()


        super(GridWorld, self).__init__(self.nS, self.nA, P, R, isd, nameActions=self.nameActions, seed=None,name=name)
        self.renderers = []
        #self.renderers['gw-pyplot'] = gwppRendering.GridworldRenderer
        #self.renderers['gw-text'] = gwtRendering.GridworldRenderer
        #self.rendermode = 'gw-text'

    def to_s(self, rowcol):
        return rowcol[0] * self.sizeY + rowcol[1]

    def from_s(self, s):
        return s // self.sizeY, s % self.sizeY

    def step(self, a):
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        r = rewarddis.rvs()
        m = rewarddis.mean()
        self.s = s
        self.lastaction = a
        self.lastreward = r
        return s, r, d,False, {"mean":m}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.np_random, seed = seeding.np_random(seed)
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s, {"mean":0}

    def makeGoalStates(self, nb):
        goalstates = []
        for g in range(nb):
            s = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
            while (self.maze[s[0]][s[1]] == 0):
                s = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
            goalstates.append(self.revmapping[self.to_s(s)])
            self.maze[s[0]][s[1]] = 2.
        return goalstates


    def makeGoalState(self, xy=None):
        goalstates = []
        if (xy == None):
            xy = [np.random.integers(self.sizeX), np.random.integers(self.sizeY)]
            while (self.maze[xy[0]][xy[1]] != 1):
                xy = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
        goalstates.append(self.revmapping[self.to_s(xy)])
        self.maze[xy[0]][xy[1]] = 2.
        return goalstates

    def makeInitialSingleStateDistribution(self, maze,xy=None):
        if (xy==None):
            xy =[np.random.integers(self.sizeX), np.random.integers(self.sizeY)]
            while (self.maze[xy[0]][xy[1]] != 1):
                xy = [self.np_random.integers(self.sizeX), self.np_random.integers(self.sizeY)]
        isd = np.zeros(self.nS)
        isd[self.revmapping[self.to_s(xy)]] = 1.
        return isd

    def makeInitialDistribution(self, maze):
        isd = np.ones(self.nS)
        for g in self.goalstates:
            isd[g] = 0
            #isd = np.array(maze == 1.).astype('float64').ravel()
        isd /= isd.sum()
        return isd

    def makeTransition(self, initialstatedistribution):
        X = self.sizeX
        Y = self.sizeY
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(self.mapping[s])
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    li = P[s][a]
                    for ns in range(self.nS):
                        if (initialstatedistribution[ns] > 0):
                            li.append((initialstatedistribution[ns], ns, False))
            else:
                us = [(x - 1) % X, y % Y]
                ds = [(x + 1) % X, y % Y]
                ls = [x % X, (y - 1) % Y]
                rs = [x % X, (y + 1) % Y]
                ss = [x, y]
                if (self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0): us = ss
                if (self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0): ds = ss
                if (self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0): ls = ss
                if (self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0): rs = ss
                for a in range(self.nA):
                    li = P[s][a]
                    li.append((self.massmap[a][0], self.revmapping[self.to_s(ls)], False))
                    li.append((self.massmap[a][1], self.revmapping[self.to_s(us)], False))
                    li.append((self.massmap[a][2], self.revmapping[self.to_s(rs)], False))
                    li.append((self.massmap[a][3], self.revmapping[self.to_s(ds)], False))
                    li.append((self.massmap[a][4], self.revmapping[self.to_s(ss)], False))

        return P


    def makeRewards(self):
        R = {s: {a: Dirac(0.) for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(self.mapping[s])
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    mymean=0.99
                    if (self.rewardStd > 0):
                        ma, mb = (0 - mymean) / self.rewardStd, (1 - mymean) / self.rewardStd
                        R[s][a]= stat.truncnorm(ma, mb, loc=mymean, scale=self.rewardStd)
                    else:
                        R[s][a] = Dirac(mymean)
        return R

    def getTransition(self, s, a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            p,ss,isA = c
            transition[ss]+=p
        return transition
