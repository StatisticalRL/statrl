"""
Implementation of the IMED-RL algorithm.

This module implements the **IMED-RL (Indexed Minimum Empirical Divergence for
Reinforcement Learning)** algorithm introduced in:

    Fabien Pesquerel and Odalric-Ambrym Maillard.
    *IMED-RL: Regret Optimal Learning of Ergodic Markov Decision Processes.*
    Advances in Neural Information Processing Systems (NeurIPS), 2022.

IMED-RL is an optimistic reinforcement learning algorithm for finite,
ergodic Markov Decision Processes (MDPs). It extends the IMED principle,
originally developed for stochastic multi-armed bandits, to the average-reward
reinforcement learning setting.

Unlike classical optimistic algorithms such as UCRL2 or UCRL3, IMED-RL
constructs, for every state-action pair, an information-theoretic index
measuring the statistical evidence that the action may still be optimal.
Actions with the smallest empirical divergence are preferentially explored,
leading to asymptotically optimal regret.

This implementation maintains empirical estimates of both the reward function
and transition kernel, repeatedly computes the optimal bias function of the
empirical MDP through Value Iteration, and evaluates IMED indices via a
convex optimization problem.

References
----------
Fabien Pesquerel and Odalric-Ambrym Maillard.
"IMED-RL: Regret Optimal Learning of Ergodic Markov Decision Processes."
NeurIPS 2022.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent
from statrl.settings.utils import randmin


class IMEDRL(MDPAgent):
    """
    Indexed Minimum Empirical Divergence Reinforcement Learning (IMED-RL).

    IMED-RL is a model-based reinforcement learning algorithm designed for
    finite ergodic Markov Decision Processes. The algorithm combines:

    * empirical estimation of rewards and transition probabilities;
    * dynamic programming through Value Iteration;
    * information-theoretic exploration based on the IMED principle.

    At every decision step, IMED-RL:

    1. Maintains an empirical MDP from observed transitions and rewards.
    2. Computes the bias function of the empirical optimal policy.
    3. Evaluates an IMED exploration index for every available action.
    4. Selects the action minimizing this index.

    To reduce computational complexity, the algorithm also maintains a
    *skeleton* of sufficiently explored actions for every state. Value
    Iteration is performed only over this reduced action set.

    Parameters
    ----------
    nbr_states : int
        Number of states of the MDP.

    nbr_actions : int
        Number of available actions in every state.

    name : str, optional
        Name of the learner.

    max_iter : int, default=3000
        Maximum number of iterations allowed during Value Iteration.

    epsilon : float, default=1e-3
        Stopping tolerance used during Value Iteration.

    max_reward : float, default=1
        Known upper bound on instantaneous rewards.

    Attributes
    ----------
    nS : int
        Number of states.

    nA : int
        Number of actions.

    dirac : ndarray of shape (nS, nS)
        Identity matrix whose rows correspond to Dirac distributions over
        successor states. It allows transition updates to be performed using
        simple running averages.

    actions : ndarray of shape (nA,)
        Array containing all action identifiers.

    max_iteration : int
        Maximum number of Value Iteration updates.

    epsilon : float
        Convergence tolerance for Value Iteration.

    max_reward : float
        Known upper bound on rewards.

    state_action_pulls : ndarray of shape (nS, nA)
        Number of observations collected for every state-action pair.

    state_visits : ndarray of shape (nS,)
        Number of visits to every state.

    rewards : ndarray of shape (nS, nA)
        Empirical mean reward function.

        The array is initialized to 0.5, corresponding to an optimistic
        uninformed estimate before any observation is collected.

    transitions : ndarray of shape (nS, nA, nS)
        Empirical transition probabilities.

        Each transition model is initially uniform over all successor states.

    all_selected : ndarray of shape (nS,)
        Boolean indicator specifying whether every action has been sampled
        at least once in each state.

    phi : ndarray of shape (nS,)
        Current estimate of the optimal bias function of the empirical MDP.

    skeleton : dict[int, ndarray]
        Reduced action set used during planning.

        For every state, the skeleton contains only sufficiently sampled
        actions, thereby reducing the computational cost of Value Iteration.

    index : ndarray of shape (nA,)
        Temporary storage of the IMED indices computed for the current state.

    rewards_distributions : dict
        Empirical reward distributions associated with every state-action
        pair.

        Instead of storing only empirical means, IMED-RL keeps the complete
        empirical distribution of observed rewards because the multinomial
        IMED index requires the full reward support.

    s : int or None
        Current state of the learner.
    """

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="IMED-RL",
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
    ):
        """
        Construct an IMED-RL learner.

        The constructor initializes all algorithmic hyperparameters together
        with the data structures required to estimate the empirical MDP and
        compute IMED exploration indices.

        Notes
        -----
        The empirical reward function is initialized optimistically to 0.5,
        while empirical transition probabilities are initialized as uniform
        distributions over successor states. These initial values provide
        non-degenerate estimates before any interaction with the environment.
        """
        MDPAgent.__init__(self, nbr_states, nbr_actions, name=name)

        # ------------------------------------------------------------------
        # Problem dimensions
        # ------------------------------------------------------------------

        self.nS = nbr_states
        self.nA = nbr_actions

        # ------------------------------------------------------------------
        # Frequently reused constant structures
        # ------------------------------------------------------------------

        # Dirac distributions over successor states.
        self.dirac = np.eye(self.nS, dtype=int)

        # Array containing every action identifier.
        self.actions = np.arange(self.nA, dtype=int)

        # ------------------------------------------------------------------
        # Algorithm parameters
        # ------------------------------------------------------------------

        self.max_iteration = max_iter
        self.epsilon = epsilon
        self.max_reward = max_reward

        # ------------------------------------------------------------------
        # Empirical MDP estimates
        # ------------------------------------------------------------------

        # Number of observations of every state-action pair.
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)

        # Number of visits of every state.
        self.state_visits = np.zeros(self.nS, dtype=int)

        # Empirical reward function.
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5

        # Empirical transition kernel.
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS

        # Indicates whether every action has been observed at least once.
        self.all_selected = np.zeros(self.nS, dtype=bool)

        # Current estimate of the bias function.
        self.phi = np.zeros(self.nS)

        # Reduced action set used during planning.
        self.skeleton = {
            s: np.arange(self.nA, dtype=int)
            for s in range(self.nS)
        }

        # IMED indices for the current state.
        self.index = np.zeros(self.nA)

        # Empirical reward distributions.
        self.rewards_distributions = {
            s: {
                a: {}
                for a in range(self.nA)
            }
            for s in range(self.nS)
        }

        # Current state.
        self.s = None

    def reset(self, state):
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
        self.all_selected = np.zeros(self.nS, dtype=bool)
        self.phi = np.zeros(self.nS)
        self.skeleton = {s: np.arange(self.nA, dtype=int) for s in range(self.nS)}
        self.index = np.zeros(self.nA)
        self.rewards_distributions = {s: {a: {1: 0, 0.5: 1} for a in range(self.nA)} for s in range(self.nS)}

        self.s = state

    def value_iteration(self):
        ctr = 0
        stop = False
        phi = np.copy(self.phi)
        phip = np.copy(self.phi)
        while not stop:
            ctr += 1
            for state in range(self.nS):
                u = - np.inf
                for action in self.skeleton[state]:
                    psa = self.transitions[state, action]
                    rsa = self.rewards[state, action]
                    u = max(u, rsa + psa @ phi)
                phip[state] = u
            phip = phip - np.min(phip)
            delta = np.max(np.abs(phi - phip))
            phi = np.copy(phip)
            stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
        self.phi = np.copy(phi)

    def update(self, state, action, reward, observation):
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (na + 2)

        if reward in self.rewards_distributions[state][action].keys():
            self.rewards_distributions[state][action][reward] += 1
        else:
            self.rewards_distributions[state][action][reward] = 1

        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na)**2
        self.skeleton[state] = self.actions[mask]

        self.s = observation

        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

    def multinomial_imed(self, state):
        upper_bound = self.max_reward + np.max(self.phi)
        q = self.rewards[state] + self.transitions[state] @ self.phi
        mu = np.max(q)
        u = upper_bound / (upper_bound - mu) - 1e-2

        for a in range(self.nA):
            if q[a] >= mu:
                self.index[a] = np.log(self.state_action_pulls[state, a])
            else:
                r_d = self.rewards_distributions[state][a]
                vr = np.fromiter(r_d.keys(), dtype=float)
                pr = np.fromiter(r_d.values(), dtype=float)
                pr = pr / pr.sum()

                pt = self.transitions[state][a]

                p = np.zeros(len(pr)*self.nS)
                v = np.zeros(len(pr)*self.nS)
                k = 0
                for i in range(self.nS):
                    for j in range(len(pr)):
                        p[k] = pt[i]*pr[j]
                        v[k] = self.phi[i] + vr[j]
                        k += 1

                delta = v - mu

                h = lambda x: - np.sum(p * np.log(upper_bound - delta*x))

                res = minimize_scalar(h, bounds=(0, u), method='bounded')
                x = - res.fun
                n = self.state_action_pulls[state, a]
                self.index[a] = n * x + np.log(n)

    def play(self, state):
        if not self.all_selected[state]:
            action = randmin(self.state_action_pulls[state])
        else:
            self.value_iteration()
            self.multinomial_imed(state)
            action = randmin(self.index)
        return action
