"""
Posterior Sampling Reinforcement Learning (PSRL).

This module implements Posterior Sampling Reinforcement Learning (PSRL),
also known as Thompson Sampling for reinforcement learning, for finite
Markov Decision Processes with Bernoulli rewards.

The algorithm maintains Bayesian posterior distributions over both the
reward function and the transition kernel of the unknown MDP. At the
beginning of every episode, a complete MDP model is sampled from the
posterior. The optimal policy of this sampled model is then computed
using Value Iteration and executed until an episode stopping criterion
is met.

The implementation follows the episodic version introduced in

    Osband, Russo and Van Roy,
    "(More) Efficient Reinforcement Learning via Posterior Sampling",
    NeurIPS 2013.

Reward distributions are modeled by Beta posteriors, while transition
probabilities are modeled by Dirichlet posteriors.
"""


import numpy as np
import scipy.stats as stat
from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
from statrl.settings.utils import categorical_sample


class PSRL(MDPAgent):
    """
    Posterior Sampling Reinforcement Learning.

    PSRL is a Bayesian model-based reinforcement learning algorithm for
    finite Markov Decision Processes.

    Rather than constructing optimistic confidence sets, PSRL maintains
    posterior distributions over the unknown reward function and
    transition probabilities.

    At the beginning of each episode, a complete MDP is sampled from the
    posterior. The optimal policy of this sampled MDP is computed and
    executed until a stopping criterion triggers the beginning of a new
    episode.

    This implementation assumes

    * finite state and action spaces;
    * Bernoulli rewards;
    * Beta priors over rewards;
    * Dirichlet priors over transitions.

    Parameters
    ----------
    nS : int
        Number of states.

    nA : int
        Number of actions.

    delta : float, optional
        Confidence parameter. Present for compatibility with the common
        learner interface but not explicitly used by the PSRL algorithm.

    Attributes
    ----------
    t : int
        Global interaction time.

    policy : ndarray of shape (nS, nA)
        Current stochastic policy computed on the sampled MDP.

    u : ndarray of shape (nS,)
        Bias (relative value) function obtained by Value Iteration.

    Nk : ndarray of shape (nS, nA)
        Cumulative number of visits to every state-action pair over all
        completed episodes.

    vk : ndarray of shape (nS, nA)
        State-action visit counts during the current episode.

    r_successCounts :
        Beta posterior α parameters.

    r_failureCounts :
        Beta posterior β parameters.

    p_pseudoCounts :
        Dirichlet pseudo-counts defining the posterior transition model.

    r_sampled :
        Reward function sampled from the posterior.

    p_sampled :
        Transition kernel sampled from the posterior.
    """

    def __init__(self, nS, nA, delta=0.05):
        """
        Construct a Posterior Sampling Reinforcement Learning learner.

        The learner starts with independent conjugate priors for every
        state-action pair.

        Rewards are modeled using Beta distributions

            Beta(1,1),

        corresponding to a uniform prior over Bernoulli means.

        Transition probabilities are modeled using symmetric Dirichlet
        priors

            Dirichlet(1,...,1),

        corresponding to a uniform prior over successor states.
        """
        MDPAgent.__init__(self, nS, nA, name="PSRL")
        # ---------------------------------------------------------
        # Problem dimensions
        # ---------------------------------------------------------

        self.nS = nS
        self.nA = nA

        # ---------------------------------------------------------
        # Global interaction statistics
        # ---------------------------------------------------------

        self.t = 1
        self.delta = delta

        # Complete interaction history
        self.observations = [[], [], []]

        # ---------------------------------------------------------
        # Episode statistics
        # ---------------------------------------------------------

        # Number of visits during the current episode.
        self.vk = np.zeros((self.nS, self.nA))

        # Cumulative visits over previous episodes.
        self.Nk = np.zeros((self.nS, self.nA))

        # ---------------------------------------------------------
        # Planning variables
        # ---------------------------------------------------------

        # Current policy computed on the sampled MDP.
        self.policy = np.zeros((self.nS, self.nA))

        # Relative value (bias) function.
        self.u = np.zeros(self.nS)

        # ---------------------------------------------------------
        # Bayesian posterior over rewards
        # ---------------------------------------------------------

        # Alpha parameters of Beta posteriors.
        self.r_successCounts = np.ones((self.nS, self.nA))

        # Beta parameters of Beta posteriors.
        self.r_failureCounts = np.ones((self.nS, self.nA))

        # ---------------------------------------------------------
        # Bayesian posterior over transitions
        # ---------------------------------------------------------

        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))

        # ---------------------------------------------------------
        # Sampled MDP
        # ---------------------------------------------------------

        # Reward function sampled from the posterior.
        self.r_sampled = np.zeros((self.nS, self.nA))

        # Transition kernel sampled from the posterior.
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))

    def reset(self, inistate):
        """Clear every statistic, restore the priors, and open the first episode.

        Reward counts restart at ``Beta(1, 1)`` and transition pseudo-counts
        at a flat ``Dirichlet(1, ..., 1)``, i.e. uniform priors, before an
        initial model is sampled and solved.

        Parameters
        ----------
        inistate : int
            State the environment was reset to.
        """
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

        self.r_successCounts = np.ones((self.nS, self.nA))
        self.r_failureCounts = np.ones((self.nS, self.nA))
        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))


        self.r_sampled = np.zeros((self.nS, self.nA))
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))

        self.new_episode()


    def VI(self, epsilon=0.01, max_iter=1000):
        """Solve the sampled MDP by average-reward (relative) value iteration.

        Iterates the Bellman operator on the bias function until the span of
        successive iterate differences contracts below ``epsilon``
        (Puterman & Chan, "Markov Decision Processes and Reinforcement
        Learning", Ch. 7, Algorithm 7.1), then computes the greedy policy for
        the converged bias function. Ties among greedy actions are broken
        uniformly among the least-visited ones, to keep exploring where the
        sampled model cannot distinguish actions.

        Parameters
        ----------
        epsilon : float, default=0.01
            Stopping threshold on the span of successive bias differences.
        max_iter : int, default=1000
            Maximum number of sweeps. If reached without convergence, the
            current iterate is kept and a warning is printed.
        """
        u0 = self.u - min(self.u)

        for _ in range(max_iter):
            action_values = self.r_sampled + self.p_sampled @ u0
            u1 = np.max(action_values, axis=1)

            diff = u1 - u0
            if np.ptp(diff) < epsilon:
                break
            u0 = u1 - np.min(u1)
        else:
            print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")

        self.u = u1 - np.min(u1)

        # Greedy policy w.r.t. the converged bias function, tie-breaking by
        # choosing: Uniform(Argmin(Nk)) among the greedy actions.
        action_values = self.r_sampled + self.p_sampled @ self.u
        u1 = np.max(action_values, axis=1)
        for s in range(self.nS):
            greedy_actions = np.flatnonzero(action_values[s] == u1[s])
            greedy_counts = self.Nk[s, greedy_actions]
            choice = greedy_actions[greedy_counts == np.min(greedy_counts)]
            self.policy[s] = 0.0
            self.policy[s, choice] = 1.0 / len(choice)

    def new_episode(self):
        """Fold in the last episode's counts, sample a fresh MDP, and solve it.

        Draws a reward mean per state-action pair from its Beta posterior and
        a transition row from its Dirichlet posterior, then runs :meth:`VI` on
        that sampled model. Sampling a *whole* MDP rather than perturbing each
        pair independently is what makes the exploration coherent across
        states — the policy commits to one plausible world for the episode.

        The value-iteration precision tightens as ``1 / t``, so early episodes
        are solved coarsely and later ones exactly.
        """
        self.updateN()

        for s in range(self.nS):
            for a in range(self.nA):
                self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
                p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
                p=p[0]
                self.p_sampled[s,a] = [p[ns] for ns in range(self.nS)]


        self.VI(epsilon=1. / max(1, self.t))

    ###### Steps and updates functions ######
 
    def updateN(self):
        """Auxiliary function to update N the current state-action count.

        Adds ``vk`` into ``Nk`` and zeroes ``vk`` for the next episode.
        """
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]
                self.vk[s, a] = 0

     
    def play(self, state):
        """Sample an action from the current policy, starting an episode if due.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        int
            The chosen action.

        Notes
        -----
        The doubling stopping criterion ends an episode as soon as a
        state-action pair has been visited as often within it as in all
        previous episodes combined. This bounds the number of episodes
        logarithmically, so the cost of resampling and re-solving the MDP
        stays negligible against the horizon.
        """
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stopping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action 
    
    def update(self, state, action, reward, observation):
        """Update the learner (the Beta and Dirichlet posteriors) with one transition (one step of the current policy).

        Parameters
        ----------
        state : int
            State the action was taken in.
        action : int
            Action taken.
        reward : float
            Observed reward, treated as Bernoulli: it is added to the success
            count and its complement to the failure count. Rewards outside
            :math:`[0, 1]` therefore corrupt the posterior.
        observation : int
            State reached; increments that transition's pseudo-count.
        """
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)

        self.r_successCounts[state,action] += reward
        self.r_failureCounts[state,action] += 1.-reward

        #print(state,action,observation)
        self.p_pseudoCounts[state,action,observation] +=1

        self.t += 1


