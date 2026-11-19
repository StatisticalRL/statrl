# UCRL3Lazy

## Overview

`UCRL3Lazy` implements **UCRL3** (*Upper Confidence Reinforcement Learning 3*) with lazy support updates for finite, discrete, unstructured Markov Decision Processes (MDPs) under the average-reward criterion, introduced by Bourel, Maillard, and Talebi (ICML 2020).

While `UCRL2` achieves rate-optimal asymptotic regret bounds, its $L_1$-norm transition confidence sets and unrestricted Extended Value Iteration (`EVI`) lead to overly conservative exploration and long burn-in phases in practice. `UCRL3Lazy` addresses both bottlenecks via:

1. **Time-uniform component-wise confidence bounds** for rewards and individual transition probabilities $p(s' \mid s, a)$.
2. **Near-Optimistic Support-adaptive Optimization (`EVI-NOSS`)** with lazy support updates every $L$ value-iteration steps.

---

## Core Principle

At the start of each episode $k$ at time $t_k$, let $N_{t_k}(s, a)$ be the visit count of state-action pair $(s, a)$ and set $\delta_0 = \frac{\delta}{3 + 3|\mathcal{S}|}$, $\delta' = \frac{\delta_0}{|\mathcal{S}||\mathcal{A}|}$.

### 1. Time-Uniform Reward Confidence Sets

For each $(s, a) \in \mathcal{S} \times \mathcal{A}$ with $n = N_{t_k}(s, a) \ge 1$:

$$c_{t,\delta_0}(s, a) = \left\{ \mu' \in [0, 1] : |\hat{\mu}_t(s, a) - \mu'| \le b^r_{t,\delta'}(s, a) \right\}$$

where

$$b^r_{t,\delta'}(s, a) = \max\left\{ \frac{1}{2} \beta_n(\delta'), \; \sqrt{\frac{2\hat{\sigma}_t^2(s, a)}{n}\ell_n(\delta')} + \frac{7\ell_n(\delta')}{3n} \right\}$$

with peeling and Laplace mixture functions:

$$\ell_n(\delta') = \eta \log\left( \frac{\log(n)\log(\eta n)}{\log^2(\eta)\delta'} \right), \qquad \beta_n(\delta') = \sqrt{\frac{2(1 + 1/n)\log(\sqrt{n+1}/\delta')}{n}}, \qquad \eta = 1.12.$$

### 2. Component-Wise Transition Confidence Sets

For each $(s, a, s') \in \mathcal{S} \times \mathcal{A} \times \mathcal{S}$, $C_{t,\delta_0}(s, a, s')$ intersects a time-uniform Bernstein peeling bound and a Bernoulli sub-Gaussian Laplace bound:

$$|\hat{p}_t(s' \mid s, a) - q| \le \sqrt{\frac{2q(1-q)}{n}\ell_n(\delta')} + \frac{\ell_n(\delta')}{3n}, \qquad -\sqrt{\underline{g}(q)} \le \frac{\hat{p}_t(s' \mid s, a) - q}{\beta_n(\delta')} \le \sqrt{\underline{g}(q)}$$

where

$$\underline{g}(q) = \begin{cases} \dfrac{1/2 - q}{\log(1/q - 1)} & \text{if } q < 1/2, \\[6pt] q(1 - q) & \text{if } q \ge 1/2. \end{cases}$$

When $n > 1$ and all observed transitions from $(s, a)$ land on the same state $s'$ ($\hat{p}_t(s' \mid s, a) = 1$), the atypical-sequence refinement (Appendix D) enforces a lower bound of at least $(1/2)^n$.

### 3. Near-Optimistic Support Optimization (`EVI-NOSS`)

For a non-negative shifted value function $f = u_n - \min_s u_n(s)$, define the optimistic sub-probability value over $\widetilde{\mathcal{S}} \subseteq \mathcal{S}$:

$$\overline{f}_{s,a}(\widetilde{\mathcal{S}}) = \max_{\widetilde{p} \in \mathcal{X}} \sum_{s' \in \widetilde{\mathcal{S}}} f(s')\widetilde{p}(s'), \quad \mathcal{X} = \left\{ \widetilde{p} : \widetilde{p}(s') \in C_{t,\delta_0}(s,a,s'), \sum_{s' \in \widetilde{\mathcal{S}}} \widetilde{p}(s') \le 1 \right\}.$$

Starting from $\widetilde{\mathcal{S}} = \operatorname{supp}(\hat{p}_t(\cdot \mid s, a)) \cup \{\arg\max_s f(s)\}$, `NOSS` greedily adds the highest-value excluded state while

$$\overline{f}_{s,a}(\mathcal{S} \setminus \widetilde{\mathcal{S}}) \ge \min\left(\kappa_{t,n}(s,a), \; \overline{f}_{s,a}(\widetilde{\mathcal{S}})\right), \qquad \kappa_{t,n}(s,a) = \frac{\gamma \mathbb{S}(u_n) |\operatorname{supp}(\hat{p}_t(\cdot \mid s, a))|}{\max_{s',a'} N_t(s',a')^{2/3}}.$$

In `UCRL3Lazy`, supports $\widetilde{\mathcal{S}}_{s,a}$ are recomputed every $L$ (`lazy_update_freq`) iterations, while value updates iterate until $\mathbb{S}(u_{n+1} - u_n) \le 1/\sqrt{t_k}$. Policy ties are broken uniformly among $\operatorname{Argmin}_{a} N_{t_k}(s, a)$.

---

## Algorithm Intuition

- **Why component-wise bounds?** $L_1$ confidence balls require a union bound over $2^S$ subsets, introducing a $\sqrt{S}$ factor even when transitions are sparse. Element-wise Bernstein + Bernoulli Laplace bounds adapt automatically to the true support.
- **Why NOSS?** Unobserved transitions have small upper bounds $\mathcal{O}(\ell_n / n)$, yet summing $|\mathcal{S} \setminus \widetilde{\mathcal{S}}|$ such terms can skew early planning. `NOSS` includes only enough high-value states to keep the omitted mass controlled by $\kappa$, preventing spurious exploration to unlikely states.
- **Why lazy support updates?** Recomputing `NOSS` every $L = 5$ EVI iterations dramatically reduces planning overhead while speeding up convergence.

---

## Class Reference

```python
from statrl.settings.markovdecisionprocess.discrete_nostructure.agents.UCRL3Lazy import UCRL3Lazy

agent = UCRL3Lazy(
    nS=6,
    nA=2,
    delta=0.05,
    eta=1.12,
    gamma=10.0,
    lazy_update_freq=5,
    name="UCRL3",
    seed=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `nS` | `int` | required | Number of states $|\mathcal{S}|$. |
| `nA` | `int` | required | Number of actions $|\mathcal{A}|$. |
| `delta` | `float` | `0.05` | Confidence level $\delta \in (0, 1)$. |
| `eta` | `float` | `1.12` | Peeling parameter $\eta > 1$ in $\ell_n(\delta')$. |
| `gamma` | `float` | `10.0` | NOSS scaling constant $\gamma > 0$. |
| `lazy_update_freq` | `int` | `5` | EVI iterations $L \ge 1$ between NOSS support recomputations. |
| `name` | `str` | `"UCRL3"` | Display label for logging and plots. |
| `seed` | `int \| None` | `None` | Seed for reproducible uniform tie-breaking. |

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `t` | `int` | Current global time step ($t \ge 1$). |
| `episode` | `int` | Current episode counter ($k \ge 1$). |
| `N` | `ndarray (nS, nA)` | Total visit counts $N_t(s, a)$. |
| `N_ep` | `ndarray (nS, nA)` | Frozen visit counts $N_{t_k}(s, a)$ at episode start. |
| `v` | `ndarray (nS, nA)` | Within-episode visit counts $v_k(s, a)$. |
| `P_counts` | `ndarray (nS, nA, nS)` | Empirical transition counts $N_t(s, a, s')$. |
| `mean_rewards` | `ndarray (nS, nA)` | Empirical mean reward $\hat{\mu}_t(s, a)$. |
| `m2_rewards` | `ndarray (nS, nA)` | Sum of squared reward deviations (Welford). |
| `policy` | `ndarray (nS,)` | Active optimistic policy $\pi^+_{t_k}$. |
| `u` | `ndarray (nS,)` | Latest optimistic value vector from EVI-NOSS. |

### Methods

- `reset(inistate: int) -> None`: Resets `self.np_random` via `super().reset(inistate)`, clears all empirical counters, and computes episode 1's initial policy.
- `play(state: int) -> int`: Checks the doubling criterion $v_k(s, \pi^+_{t_k}(s)) \ge \max(1, N_{t_k}(s, \pi^+_{t_k}(s)))$; if triggered, starts a new episode with `EVI-NOSS` and returns $\pi^+_{t_k}(s)$.
- `update(state: int, action: int, reward: float, observation: int) -> None`: Updates visit counts, transition counts, Welford reward statistics, and increments `self.t`.

---

## Theoretical Guarantees

Under Theorem 1 of Bourel et al. (2020), with probability at least $1 - 4\delta$ uniformly over all $T \ge 3$, the regret of `UCRL3` satisfies:

$$\mathfrak{R}(\text{UCRL3}, T) = \mathcal{O}\left( \left( D + \sqrt{\sum_{s,a} \left(D_s^2 L_{s,a} \vee 1\right)} \right) \sqrt{T \log(\sqrt{T}/\delta)} \right)$$

where $D$ is the global MDP diameter, $D_s \le D$ is the **local diameter** of state $s$, and

$$L_{s,a} := \left( \sum_{x \in \mathcal{S}} \sqrt{p(x \mid s, a)(1 - p(x \mid s, a))} \right)^2 \le K_{s,a} - 1$$

is the **local effective support** of $p(\cdot \mid s, a)$.

---

## Computational Complexity

- **Per step**: $\mathcal{O}(1)$ for `play` (when no episode transition occurs) and `update`.
- **Per episode** ($\mathcal{O}(SA \log T)$ episodes total): $\mathcal{O}\big(SA + \sum_{s,a} K_{s,a}\big)$ scalar root solves for transition intervals, plus $\mathcal{O}\big(I \cdot (SA\bar{K} + \frac{1}{L} S^2 A)\big)$ for $I$ iterations of lazy `EVI-NOSS`.
