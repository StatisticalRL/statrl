from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import MDPAgent


class UCRL3Lazy(MDPAgent):
    """UCRL3 agent with lazy NOSS support updates for discrete MDPs.

    Implements the UCRL3 algorithm (Bourel, Maillard, and Talebi, 2020) for
    regret minimization in unknown finite communicating Markov Decision
    Processes under the average-reward criterion, together with the lazy
    support-update variant of EVI-NOSS described in Appendix D.

    UCRL3 improves upon UCRL2 via two key mechanisms:

    1. **Time-uniform component-wise confidence intervals** (Section 3.1):
       Rewards use a combination of a Hoeffding-Laplace mixture bound and a
       time-uniform empirical Bernstein peeling bound. Each transition
       probability :math:`p(s' \mid s, a)` is bounded by intersecting a
       time-uniform Bernstein peeling inequality with a Bernoulli
       sub-Gaussian Laplace mixture bound, plus an atypical-sequence lower
       bound :math:`(1/2)^n` when :math:`\hat{p}_t(s' \mid s, a) = 1` and
       :math:`n > 1` (Appendix D).
    2. **Near-Optimistic Support-adaptive Optimization (EVI-NOSS)**
       (Section 3.2, Algorithms 2 and 3): Instead of optimizing over the full
       state space :math:`\mathcal{S}`, NOSS restricts transition optimization
       for each :math:`(s, a)` to an adaptively augmented subset
       :math:`\widetilde{\mathcal{S}}_{s,a} \supseteq \operatorname{supp}(\hat{p}_t(\cdot \mid s, a))`
       until the outside optimistic contribution is below
       :math:`\min(\kappa, \overline{f}_{s,a}(\widetilde{\mathcal{S}}_{s,a}))`.
       Following Appendix D, the lazy variant recomputes these supports once
       every ``lazy_update_freq`` iterations of extended value iteration.

    Parameters
    ----------
    nS : int
        Number of states :math:`|\mathcal{S}|` in the discrete MDP.
    nA : int
        Number of actions :math:`|\mathcal{A}|` in the discrete MDP.
    delta : float, default=0.05
        High-probability confidence level parameter in ``(0, 1)``. Used with
        confidence scaling :math:`\delta_0 = \delta / (3 + 3 |\mathcal{S}|)`.
    eta : float, default=1.12
        Peeling parameter :math:`\eta > 1` in
        :math:`\ell_n(\delta') = \eta \log(\log(n)\log(\eta n) / (\log^2(\eta)\delta'))`.
    gamma : float, default=10.0
        Scaling constant :math:`\gamma > 0` in the NOSS threshold
        :math:`\kappa_{t,n}(s,a) = \gamma \mathbb{S}(u_n) |\operatorname{supp}(\hat{p}_t(\cdot\mid s,a))| / \max_{s',a'} N_t(s',a')^{2/3}`.
    lazy_update_freq : int, default=5
        Interval :math:`L \ge 1` of EVI iterations between NOSS support
        recomputations (Appendix D). Setting ``lazy_update_freq=1`` recovers
        exact Algorithm 3.
    name : str, default='UCRL3'
        Human-readable label used in logs and plots.
    seed : int, optional
        Random seed used by ``self.np_random`` for uniform tie-breaking among
        :math:`\operatorname{Argmin}_{a \in \mathcal{A}} N_k(s, a)`.

    Attributes
    ----------
    t : int
        Global decision time step :math:`t \ge 1`.
    episode : int
        Current episode index :math:`k \ge 1`.
    N : ndarray of shape (nS, nA)
        Cumulative visit counts :math:`N_t(s, a)`.
    N_ep : ndarray of shape (nS, nA)
        Visit counts :math:`N_{t_k}(s, a)` frozen at the start of episode
        :math:`k`.
    v : ndarray of shape (nS, nA)
        Within-episode visit counts :math:`v_k(s, a)`.
    P_counts : ndarray of shape (nS, nA, nS)
        Cumulative transition counts :math:`N_t(s, a, s')`.
    mean_rewards : ndarray of shape (nS, nA)
        Running empirical mean reward :math:`\hat{\mu}_t(s, a)`.
    m2_rewards : ndarray of shape (nS, nA)
        Running sum of squared reward deviations for Welford variance.
    policy : ndarray of shape (nS,)
        Current near-optimistic policy :math:`\pi^+_{t_k}`.
    u : ndarray of shape (nS,)
        Optimistic relative value vector from the latest EVI-NOSS run.

    See Also
    --------
    statrl.settings.markovdecisionprocess.discrete_nostructure.agent.MDPAgent
    statrl.settings.markovdecisionprocess.discrete_nostructure.interaction.MDPInteraction

    Notes
    -----
    - **Time complexity**: There are :math:`\mathcal{O}(SA \log T)` episodes
      over horizon :math:`T`. Each episode computes component-wise confidence
      bounds in :math:`\mathcal{O}(SA + \sum_{s,a} |\operatorname{supp}(\hat{p}_{s,a})|)`
      scalar solves (since all zero-count transitions for a given :math:`(s, a)`
      share the same upper bound) and runs EVI-NOSS in
      :math:`\mathcal{O}(I \cdot (SA \cdot \bar{K} + \frac{1}{L} SA \cdot S))`
      operations, where :math:`I` is the number of value iterations and
      :math:`\bar{K}` is the average active support size.
    - **Space complexity**: :math:`\mathcal{O}(S^2 A)` to store transition
      counts and confidence intervals.

    References
    ----------
    .. [1] Bourel, H., Maillard, O.-A., and Talebi, M. S. (2020). Tightening
       Exploration in Upper Confidence Reinforcement Learning. *Proceedings of
       the 37th International Conference on Machine Learning (ICML)*, PMLR 119.

    Examples
    --------
    >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.envs.riverswim import RiverSwim
    >>> from statrl.settings.markovdecisionprocess.discrete_nostructure.interaction import MDPInteraction
    >>> env = RiverSwim(nbStates=6, seed=0)
    >>> agent = UCRL3Lazy(nS=env.nS, nA=env.nA, delta=0.05, seed=0)
    >>> scores = MDPInteraction().run(env, agent, horizon=200)
    >>> scores.shape
    (200,)
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        delta: float = 0.05,
        eta: float = 1.12,
        gamma: float = 10.0,
        lazy_update_freq: int = 5,
        name: str = "UCRL3",
        seed: int | None = None,
    ) -> None:
        if nS <= 0:
            raise ValueError(f"nS must be positive, got {nS}.")
        if nA <= 0:
            raise ValueError(f"nA must be positive, got {nA}.")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}.")
        if eta <= 1.0:
            raise ValueError(f"eta must be > 1.0, got {eta}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0.0, got {gamma}.")
        if lazy_update_freq < 1:
            raise ValueError(f"lazy_update_freq must be >= 1, got {lazy_update_freq}.")

        super().__init__(nS=nS, nA=nA, name=name, seed=seed)
        self.delta: float = float(delta)
        self.eta: float = float(eta)
        self.gamma: float = float(gamma)
        self.lazy_update_freq: int = int(lazy_update_freq)

        # Confidence level delta_0 = delta / (3 + 3S) and per-(s,a) level delta_0 / (SA) (Section 3.1).
        self._delta0: float = self.delta / (3.0 + 3.0 * self.nS)
        self._delta_sa: float = self._delta0 / (self.nS * self.nA)
        self._log_eta_sq: float = math.log(self.eta) ** 2
        self._max_evi_iter: int = 5000

        # Mutable episode and empirical statistics buffers (initialized in reset).
        self.t: int = 1
        self.episode: int = 0
        self.N: NDArray[np.int64] = np.zeros((self.nS, self.nA), dtype=np.int64)
        self.N_ep: NDArray[np.int64] = np.zeros((self.nS, self.nA), dtype=np.int64)
        self.v: NDArray[np.int64] = np.zeros((self.nS, self.nA), dtype=np.int64)
        self.P_counts: NDArray[np.int64] = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64)
        self.mean_rewards: NDArray[np.float64] = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.m2_rewards: NDArray[np.float64] = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.policy: NDArray[np.int64] = np.zeros(self.nS, dtype=np.int64)
        self.u: NDArray[np.float64] = np.zeros(self.nS, dtype=np.float64)

    def reset(self, inistate: int) -> None:
        """Reset the agent state and statistics for a new independent run.

        Parameters
        ----------
        inistate : int
            Initial state of the MDP run.
        """
        super().reset(inistate)
        self.t = 1
        self.episode = 0
        self.N.fill(0)
        self.N_ep.fill(0)
        self.v.fill(0)
        self.P_counts.fill(0)
        self.mean_rewards.fill(0.0)
        self.m2_rewards.fill(0.0)
        self.policy.fill(0)
        self.u.fill(0.0)
        self._start_new_episode()

    def play(self, state: int) -> int:
        """Select an action according to the current episode's policy.

        If the doubling criterion ``v_k(state, pi_k(state)) >= max(1, N_{t_k}(state, pi_k(state)))``
        is met (Algorithm 4), a new episode is initiated and EVI-NOSS is
        recomputed before returning the action.

        Parameters
        ----------
        state : int
            Current MDP state in ``{0, ..., nS - 1}``.

        Returns
        -------
        int
            Chosen action in ``{0, ..., nA - 1}``.
        """
        if self.episode == 0:
            self.reset(state)

        action = int(self.policy[state])
        # Algorithm 4 doubling criterion: end episode when v_k(s, a) reaches max(1, N_{t_k}(s, a)).
        if self.v[state, action] >= max(1, int(self.N_ep[state, action])):
            self._start_new_episode()
            action = int(self.policy[state])
        return action

    def update(self, state: int, action: int, reward: float, observation: int) -> None:
        """Update empirical reward and transition statistics from one step.

        Parameters
        ----------
        state : int
            State in which ``action`` was taken.
        action : int
            Action executed.
        reward : float
            Observed scalar reward in ``[0, 1]``.
        observation : int
            Next state reached after executing ``action``.
        """
        self.N[state, action] += 1
        self.v[state, action] += 1
        self.P_counts[state, action, observation] += 1

        # Welford online update for empirical mean and variance of rewards.
        n = int(self.N[state, action])
        delta_r = reward - float(self.mean_rewards[state, action])
        self.mean_rewards[state, action] += delta_r / n
        delta_r2 = reward - float(self.mean_rewards[state, action])
        self.m2_rewards[state, action] += delta_r * delta_r2

        self.t += 1

    def _l_bound(self, n: int) -> float:
        """Compute the time-uniform peeling factor ``l_n(delta_0 / (SA))``.

        Notes
        -----
        Implements :math:`\ell_n(\delta') = \eta \log(\frac{\log(n)\log(\eta n)}{\log^2(\eta)\delta'})`
        from Section 3.1 (Eq. 5). For :math:`n = 1`, the numerator inside the
        logarithm is clamped to at least ``1.0`` (matching Lemma 7's
        ``ceil(log(n)/log(eta)) >= 1``) so ``l_1`` remains finite.
        """
        if n <= 1:
            log_factor = 1.0
        else:
            log_factor = max(1.0, math.log(n) * math.log(self.eta * n) / self._log_eta_sq)
        return self.eta * math.log(log_factor / self._delta_sa)

    def _beta_bound(self, n: int) -> float:
        """Compute the Laplace mixture factor ``beta_n(delta_0 / (SA))`` (Section 3.1)."""
        return math.sqrt(2.0 * (1.0 + 1.0 / n) * math.log(math.sqrt(n + 1.0) / self._delta_sa) / n)

    @staticmethod
    def _g_underline(q: float) -> float:
        """Evaluate the asymmetric Bernoulli variance proxy ``g_underline(q)`` (Section 3.1)."""
        if q <= 0.0 or q >= 1.0:
            return 0.0
        if q >= 0.5:
            return q * (1.0 - q)
        diff = 0.5 - q
        if diff < 1e-6:
            # Smooth Taylor expansion around q = 0.5 to avoid 0/0 cancellation.
            return 0.25 - (diff * diff) / 3.0
        return diff / math.log((1.0 - q) / q)

    def _solve_transition_interval(
        self,
        p_hat: float,
        n: int,
        l_val: float,
        beta_val: float,
    ) -> tuple[float, float]:
        """Compute ``C_{t,delta_0}(s, a, s') = [q_min, q_max]`` (Section 3.1 & Appendix D)."""
        a = l_val / n
        a_third = a / 3.0

        # 1. Time-uniform Bernstein peeling bounds: |p_hat - q| <= sqrt(2*q*(1-q)*a) + a/3.
        if p_hat - a_third <= 0.0:
            q_min_b = 0.0
        else:
            disc_low = (
                2.0 * a * p_hat * (1.0 - p_hat)
                + (a * a) / 3.0
                + (4.0 / 3.0) * (a * a) * p_hat
                - (2.0 / 9.0) * (a ** 3)
            )
            q_min_b = max(
                0.0,
                (p_hat + (2.0 / 3.0) * a - math.sqrt(max(0.0, disc_low))) / (1.0 + 2.0 * a),
            )

        if p_hat + a_third >= 1.0:
            q_max_b = 1.0
        else:
            disc_high = (
                2.0 * a * p_hat * (1.0 - p_hat)
                + (5.0 / 3.0) * (a * a)
                - (4.0 / 3.0) * (a * a) * p_hat
                - (2.0 / 9.0) * (a ** 3)
            )
            q_max_b = min(
                1.0,
                (p_hat + (4.0 / 3.0) * a + math.sqrt(max(0.0, disc_high))) / (1.0 + 2.0 * a),
            )

        # 2. Bernoulli sub-Gaussian Laplace lower bound: p_hat - q <= beta * sqrt(g_underline(q)).
        if p_hat <= 0.0:
            q_min = 0.0
        elif beta_val * math.sqrt(self._g_underline(q_min_b)) - (p_hat - q_min_b) >= 0.0:
            q_min = q_min_b
        else:
            lo = q_min_b
            hi = p_hat
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                if beta_val * math.sqrt(self._g_underline(mid)) - (p_hat - mid) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            q_min = hi

        # 3. Bernoulli sub-Gaussian Laplace upper bound: q - p_hat <= beta * sqrt(g_underline(q)).
        if p_hat >= 1.0:
            q_max = 1.0
        elif beta_val * math.sqrt(self._g_underline(q_max_b)) - (q_max_b - p_hat) >= 0.0:
            q_max = q_max_b
        else:
            lo = p_hat
            hi = q_max_b
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                if beta_val * math.sqrt(self._g_underline(mid)) - (mid - p_hat) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            q_max = lo

        # 4. Atypical-sequence lower bound refinement (Appendix D).
        if n > 1 and p_hat == 1.0:
            q_min = max(q_min, 0.5 ** n)

        return q_min, q_max

    def _compute_confidence_bounds(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.int64],
    ]:
        """Build optimistic reward upper bounds and transition confidence intervals."""
        mu_plus = np.ones((self.nS, self.nA), dtype=np.float64)
        p_min = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64)
        p_max = np.ones((self.nS, self.nA, self.nS), dtype=np.float64)
        supp_hat = self.P_counts > 0
        supp_sizes = np.sum(supp_hat, axis=2, dtype=np.int64)

        zero_ub_cache: dict[int, float] = {}
        l_cache: dict[int, float] = {}
        beta_cache: dict[int, float] = {}

        for s in range(self.nS):
            for a in range(self.nA):
                n = int(self.N_ep[s, a])
                if n == 0:
                    continue

                if n not in l_cache:
                    l_cache[n] = self._l_bound(n)
                    beta_cache[n] = self._beta_bound(n)
                l_val = l_cache[n]
                beta_val = beta_cache[n]

                # Reward confidence radius b^r_{t, delta_0/(SA)}(s, a) (Section 3.1).
                var_hat = max(0.0, float(self.m2_rewards[s, a]) / n)
                b_r = max(
                    0.5 * beta_val,
                    math.sqrt(2.0 * var_hat * l_val / n) + (7.0 * l_val) / (3.0 * n),
                )
                mu_plus[s, a] = min(1.0, float(self.mean_rewards[s, a]) + b_r)

                # Unobserved transitions (p_hat == 0) share the same upper bound for a given n.
                if n not in zero_ub_cache:
                    _, z_ub = self._solve_transition_interval(0.0, n, l_val, beta_val)
                    zero_ub_cache[n] = z_ub
                p_max[s, a, :] = zero_ub_cache[n]

                # Observed transitions in supp(p_hat(.|s, a)).
                obs_states = np.flatnonzero(supp_hat[s, a])
                for s_next in obs_states:
                    p_hat = float(self.P_counts[s, a, s_next]) / n
                    q_lo, q_hi = self._solve_transition_interval(p_hat, n, l_val, beta_val)
                    p_min[s, a, s_next] = q_lo
                    p_max[s, a, s_next] = q_hi

        return mu_plus, p_min, p_max, supp_hat, supp_sizes

    @staticmethod
    def _max_expectation(
        f: NDArray[np.float64],
        order: NDArray[np.int64],
        p_min_sa: NDArray[np.float64],
        p_max_sa: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> float:
        """Compute ``f_bar_{s,a}(mask)`` over positive sub-probability measures (Eq. 2)."""
        total_min = float(np.sum(p_min_sa[mask]))
        budget = 1.0 - total_min
        val = float(np.dot(f[mask], p_min_sa[mask]))
        if budget <= 0.0:
            return val

        for idx in order:
            if not mask[idx]:
                continue
            add = min(budget, float(p_max_sa[idx] - p_min_sa[idx]))
            if add > 0.0:
                val += float(f[idx]) * add
                budget -= add
                if budget <= 1e-15:
                    break
        return val

    def _noss(
        self,
        f: NDArray[np.float64],
        order: NDArray[np.int64],
        supp_sa: NDArray[np.bool_],
        p_min_sa: NDArray[np.float64],
        p_max_sa: NDArray[np.float64],
        kappa: float,
    ) -> NDArray[np.bool_]:
        """Run NOSS (Algorithm 2) to select a near-optimistic support ``S_tilde``."""
        in_tilde = supp_sa.copy()
        in_tilde[order[0]] = True

        outside = ~in_tilde
        if not np.any(outside):
            return in_tilde

        f_bar_in = self._max_expectation(f, order, p_min_sa, p_max_sa, in_tilde)
        f_bar_out = self._max_expectation(f, order, p_min_sa, p_max_sa, outside)

        # Greedily add highest-value excluded states until error falls below min(kappa, f_bar_in).
        ptr = 0
        while f_bar_out > 0.0 and f_bar_out >= min(kappa, f_bar_in):
            while ptr < self.nS and in_tilde[order[ptr]]:
                ptr += 1
            if ptr >= self.nS:
                break
            s_new = order[ptr]
            in_tilde[s_new] = True
            outside[s_new] = False
            if not np.any(outside):
                break
            f_bar_in = self._max_expectation(f, order, p_min_sa, p_max_sa, in_tilde)
            f_bar_out = self._max_expectation(f, order, p_min_sa, p_max_sa, outside)

        return in_tilde

    def _tie_break_argmax(self, q_values: NDArray[np.float64], state: int) -> int:
        """Break ties among optimal actions uniformly over ``Argmin_a N_k(s, a)`` (Appendix D)."""
        max_q = float(np.max(q_values))
        opt_actions = np.flatnonzero(q_values >= max_q - 1e-12)
        if opt_actions.size == 1:
            return int(opt_actions[0])
        counts = self.N_ep[state, opt_actions]
        min_count = int(np.min(counts))
        least_visited = opt_actions[counts == min_count]
        chosen_idx = int(self.np_random.integers(0, least_visited.size))
        return int(least_visited[chosen_idx])

    def _evi_noss(
        self,
        mu_plus: NDArray[np.float64],
        p_min: NDArray[np.float64],
        p_max: NDArray[np.float64],
        supp_hat: NDArray[np.bool_],
        supp_sizes: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Run lazy EVI-NOSS (Algorithm 3 + Appendix D) until ``S(u_{n+1} - u_n) <= 1/sqrt(t_k)``."""
        eps = 1.0 / math.sqrt(max(1, self.t))
        n_max_pow = float(max(1, int(np.max(self.N_ep)))) ** (2.0 / 3.0)

        u = np.zeros(self.nS, dtype=np.float64)
        policy = np.zeros(self.nS, dtype=np.int64)
        active_supports = np.zeros((self.nS, self.nA, self.nS), dtype=np.bool_)
        q_vals = np.empty((self.nS, self.nA), dtype=np.float64)

        for n_iter in range(self._max_evi_iter):
            u_min = float(np.min(u))
            f = u - u_min
            span_u = float(np.max(f))
            order = np.argsort(f, kind="mergesort")[::-1]

            # Lazy support update once every L steps (Appendix D).
            if n_iter % self.lazy_update_freq == 0:
                for s in range(self.nS):
                    for a in range(self.nA):
                        if self.N_ep[s, a] == 0:
                            active_supports[s, a, :] = False
                            active_supports[s, a, order[0]] = True
                        else:
                            # Eq. (3): kappa = gamma * S(u_n) * |supp(p_hat(.|s,a))| / N_max^{2/3}.
                            kappa = self.gamma * span_u * float(supp_sizes[s, a]) / n_max_pow
                            active_supports[s, a] = self._noss(
                                f=f,
                                order=order,
                                supp_sa=supp_hat[s, a],
                                p_min_sa=p_min[s, a],
                                p_max_sa=p_max[s, a],
                                kappa=kappa,
                            )

            u_next = np.empty(self.nS, dtype=np.float64)
            for s in range(self.nS):
                for a in range(self.nA):
                    mask = active_supports[s, a]
                    if self.N_ep[s, a] == 0:
                        q_vals[s, a] = float(mu_plus[s, a]) + float(u[order[0]])
                        continue

                    # Solve argmax {P' u_n : p' in C_tilde(s, a)} (Algorithm 3).
                    p_plus = np.zeros(self.nS, dtype=np.float64)
                    p_plus[mask] = p_min[s, a, mask]
                    budget = max(0.0, 1.0 - float(np.sum(p_plus)))
                    if budget > 0.0:
                        for idx in order:
                            if not mask[idx]:
                                continue
                            add = min(budget, float(p_max[s, a, idx] - p_min[s, a, idx]))
                            if add > 0.0:
                                p_plus[idx] += add
                                budget -= add
                                if budget <= 1e-15:
                                    break
                    total_mass = float(np.sum(p_plus))
                    if abs(total_mass - 1.0) > 1e-12 and total_mass > 0.0:
                        p_plus /= total_mass

                    q_vals[s, a] = float(mu_plus[s, a]) + float(np.dot(p_plus, u))

                u_next[s] = float(np.max(q_vals[s]))

            diff = u_next - u
            diff_span = float(np.max(diff) - np.min(diff))
            u = u_next
            if diff_span <= eps:
                break

        for s in range(self.nS):
            policy[s] = self._tie_break_argmax(q_vals[s], s)

        return u, policy

    def _start_new_episode(self) -> None:
        """Initiate episode ``k``, freeze counts, and compute ``pi^+_{t_k}`` via EVI-NOSS."""
        self.episode += 1
        np.copyto(self.N_ep, self.N)
        self.v.fill(0)

        mu_plus, p_min, p_max, supp_hat, supp_sizes = self._compute_confidence_bounds()
        self.u, self.policy = self._evi_noss(mu_plus, p_min, p_max, supp_hat, supp_sizes)
