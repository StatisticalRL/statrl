# IMED: Indexed Minimum Empirical Divergence

## Overview

The **IMED (Indexed Minimum Empirical Divergence)** algorithm is an index-based strategy for stochastic multi-armed bandits. It belongs to the class of **asymptotically optimal algorithms** and is particularly designed for Bernoulli reward distributions (though it can be generalized via the KL function).

The algorithm assigns each arm a statistical index combining:
- empirical performance
- divergence from the current best arm
- logarithmic exploration pressure

---

## Core Principle

For each arm \( a \), define:

\[
I_a(t) = N_a(t)\,\mathrm{KL}(\mu_a(t), \mu^*(t)) + \log N_a(t)
\]

where:

- \( N_a(t) \): number of pulls of arm \( a \)
- \( \mu_a(t) \): empirical mean reward of arm \( a \)
- \( \mu^*(t) = \max_a \mu_a(t) \): best empirical mean
- \( \mathrm{KL}(p,q) \): Kullback–Leibler divergence (typically Bernoulli KL)

The selected action is:

\[
a_t = \arg\min_a I_a(t)
\]

---

## Algorithm Intuition

IMED prioritizes arms that:
- are insufficiently explored (small \( N_a \))
- are statistically close to the best arm (small KL divergence)

This produces a **natural exploration-exploitation balance** without explicit tuning parameters.

---

## Class Reference

### `IMED(nbArms, kullback)`

#### Parameters
- `nbArms` (int): number of arms
- `kullback` (callable): KL divergence function

---

## Attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `nbDraws` | `np.ndarray` | Number of pulls per arm |
| `cumRewards` | `np.ndarray` | Cumulative rewards per arm |
| `means` | `np.ndarray` | Empirical mean reward per arm |
| `maxMeans` | float | Best empirical mean across arms |
| `indexes` | `np.ndarray` | IMED index values |

---

## Methods

### `reset()`
Reinitializes all statistics before a new experiment.

---

### `play(state=0)`
Selects an arm according to:

\[
\arg\min_a I_a(t)
\]

Returns:
- `int`: selected arm index

---

### `update(arm, reward)`
Updates internal statistics.

#### Steps
1. Increment pull count
2. Update cumulative reward
3. Update empirical means
4. Compute best empirical mean
5. Recompute IMED index for each arm

---

## Theoretical Properties

IMED is known to:
- achieve **asymptotic optimality** under standard bandit assumptions
- match lower bounds based on KL divergence
- avoid explicit exploration scheduling

---

## Computational Complexity

Let \( K \) be number of arms:

- Update step: \( O(K) \)
- Memory: \( O(K) \)

---

## Remarks

- The choice of KL divergence is problem-specific (e.g., Bernoulli KL for binary rewards)
- The algorithm is sensitive to numerical stability of KL computations
- Designed for stochastic stationary environments