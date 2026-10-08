import numpy as np
from statrl.settings.utils import categorical_sample

from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
def build_opti(name, env, nS, nA):
    """Build the oracle for an environment.

    Parameters
    ----------
    name : str
        Environment name. Currently ignored: the per-map hand-coded oracles
        are commented out, so every environment gets the generic solver.
    env : DiscreteMDP
        The environment to solve.
    nS, nA : int
        Numbers of states and actions.

    Returns
    -------
    Opti_controller
        An oracle whose policy was computed by value iteration.
    """
    #if ("2-room" in name):
    #    return  Opti_911_2room(env)
    #elif ("4-room" in name):
    #    return  Opti_77_4room(env)
    #elif ("RiverSwim" in name):
    #    return Opti_swimmer(env)
    #else:
    return Opti_controller(env, nS, nA)



class Opti_controller(MDPAgent):
    """Oracle that solves the MDP exactly and follows its optimal policy.

    Reads the true transitions and mean rewards from the environment at
    construction and runs value iteration on them, so it plays optimally from
    the first step. 

    Parameters
    ----------
    env : DiscreteMDP
        The environment. Must expose ``getTransition`` and ``getMeanReward``;
        otherwise ``extractRewardsAndTransitions`` is tried as a fallback.
    nS, nA : int
        Numbers of states and actions.
    epsilon : float, default=0.001
        Stopping precision recorded on the instance. The value iteration run
        at construction uses a much tighter ``1e-7`` regardless, so the policy
        is effectively exact.
    max_iter : int, default=100
        Iteration cap recorded on the instance; the construction-time run uses
        100000.

    Attributes
    ----------
    policy : ndarray of shape (nS, nA)
        Optimal stochastic policy, uniform over tied optimal actions.
    u : ndarray of shape (nS,)
        Bias function from value iteration.

    """

    def __init__(self, env, epsilon=0.001, max_iter=100):
        self.name="Oracle"
        self.env = env
        self.nS = env.nS
        self.nA = env.nA
        super(Opti_controller, self).__init__(self.nS, self.nA, self.name)
        self.u = np.zeros(self.nS)
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.not_converged = True
        self.transitions = np.zeros((self.nS, self.nA, self.nS))
        self.meanrewards = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS, self.nA))

        try:
            for s in range(self.nS):
                for a in range(self.nA):
                    self.transitions[s, a] = self.env.getTransition(s, a)
                    self.meanrewards[s, a] = self.env.getMeanReward(s, a)
                    self.policy[s,a] = 1. / self.nA
        except AttributeError:
            for s in range(self.nS):
                for a in range(self.nA):
                    self.transitions[s, a], self.meanrewards[s, a] = self.extractRewardsAndTransitions(s, a)
                    self.policy[s, a] = 1. / self.nA

        self.VI(epsilon=0.0000001, max_iter=100000)


    def extractRewardsAndTransitions(self,s,a):
        """Reader for the transitions and mean reward of a pair.

        Parameters
        ----------
        s : int
            The state.
        a : int
            The action.

        Returns
        -------
        transition : ndarray of shape (nS,)
            Probability of reaching each state.
        reward : float
            Mean reward of the pair.
        """
        transition  = self.env.getTransition(s,a)
        reward = self.env.getMeanReward(s,a)
        #transition = np.zeros(self.nS)
        #reward = 0.
        #for c in self.env.P[s][a]: #c= proba, nexstate, reward, done
        #    transition[c[1]]=c[0]
        #    reward = c[2]
        return transition, reward

    def reset(self, inistate):
        """Start a new run. The policy is fixed, so nothing is cleared.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        ()

    def play(self, state):
        """Sample an action from the optimal policy for this state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        int
            An optimal action, drawn uniformly among ties.
        """
        a = categorical_sample([self.policy[state,a] for a in range(self.nA)], np.random)
        return a

    def update(self, state, action, reward, observation):
        """Ignore the transition (the oracle has nothing to learn).

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached.
        """
        ()

    def VI(self, epsilon=0.01, max_iter=1000):
        """Solve the true MDP by value iteration and store the greedy policy.

        Parameters
        ----------
        epsilon : float, default=0.01
            Stopping threshold on the span of successive bias differences.
        max_iter : int, default=1000
            Iteration cap. On reaching it the current iterate is kept and a
            non-convergence warning is printed.
        """
        u0 = self.u - min(self.u)  # np.zeros(self.nS)
        itera = 0
        while True:
            action_values = (
                self.meanrewards
                + 0.999 * (self.transitions @ u0)
            )
            u1 = np.max(action_values, axis=1)
            optimal_actions = action_values == u1[:, None]
            self.policy = optimal_actions / np.sum(
                optimal_actions, axis=1, keepdims=True
            )

            diff = np.abs(u1 - u0)
            if np.ptp(diff) < epsilon:
                self.u = u1 - np.min(u1)
                break
            elif itera > max_iter:
                self.u = u1 - np.min(u1)
                print("[Opt] No convergence in VI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1 - np.min(u1)
                itera += 1







class Opti_swimmer(MDPAgent):
    """Hand-coded oracle for RiverSwim: always swim right.

    Skips value iteration by encoding the known optimal policy directly.

    Parameters
    ----------
    env : DiscreteMDP
        The RiverSwim instance.
    """

    def __init__(self, env):
        self.env = env
        self.policy = np.zeros(self.env.nS)
        self.name="Opti_swimmer"
        super(Opti_swimmer, self).__init__(env.nS, env.nA, self.name)

    def reset(self, inistate):
        """Start a new run. The policy is fixed, so nothing is cleared.

        Parameters
        ----------
        inistate : int
            Initial state; ignored.
        """
        ()

    def play(self, state):
        """Always take action 0, "swim right".

        Parameters
        ----------
        state : int
            Current state; ignored.

        Returns
        -------
        int
            Always ``0``.
        """
        return 0

    def update(self, state, action, reward, observation):
        """Ignore the transition (the oracle has nothing to learn).

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Reward observed.
        observation : int
            State reached.
        """
        ()
