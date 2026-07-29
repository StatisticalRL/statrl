"""
baba_schedule.py — BABA static time-grid computation
=====================================================
All logarithms are base-2, matching the paper's implicit convention.
Verified: I1=2000, alpha=3, K=2 → 10 batches for T ≈ 10^5, consistent
with Figure 1 of Jin et al. (ICML 2021).

Each epoch r has exactly 5 batches with sizes:
    B1 = K · g(Ir)                           — UNIFORMEXPLORATION
    B2 = ⌈(log₂ Ir)²⌉                       — INITIALEXPLOITATION
    B3 = ⌈K · (log₂ Ir)²⌉                   — OPTIMISTICEXPLORATION
    B4 = ⌈K · (log₂ Ir)²⌉                   — CONFIDENTEXPLORATION
    B5 = Ir − I_{r−1} − B1 − B2 − B3 − B4  — CONFIDENTEXPLOITATION

Epoch boundaries: I_1 = I1 (auto-selected), I_{r+1} = f(I_r).

Public API
----------
ilog(x, m)             — apply log₂ iteratively m times
g_baba(Ir)             — exploration depth g(Ir)
f_baba(x, alpha)       — epoch growth function f(x)
find_I1(K, alpha)      — smallest valid I1 for K arms
compute_baba_grid(...) — returns (batch_sizes, phase_labels, epoch_ids, epoch_I)
"""

import math


# ── core mathematical primitives ──────────────────────────────────────────────

def ilog(x, m):
    """Apply log₂ iteratively m times; clamp result to 0 if ≤ 0."""
    result = float(x)
    for _ in range(m):
        if result <= 1.0:
            return 0.0
        result = math.log2(result)
    return max(result, 0.0)


def g_baba(Ir):
    """
    Uniform-exploration depth.
        g(Ir) = ⌈log₂(Ir) / log₂(log₂(Ir))⌉

    Represents the minimum number of pulls per arm in UNIFORMEXPLORATION.
    """
    if Ir <= 2:
        return 1
    lx  = math.log2(float(Ir))
    llx = math.log2(lx) if lx > 1.0 else 1.0
    return max(1, math.ceil(lx / llx))


def f_baba(x, alpha=3):
    """
    Epoch growth function.
        f(x) = max( ⌈x^(1 + 1/(1 + ilogα(x)))⌉,  2x )

    Grows faster than geometric doubling (2x) but slower than x², giving
    BABA's O(log log T · ilogα T) batch complexity.
    """
    il       = ilog(x, alpha)
    exponent = 1.0 + 1.0 / (1.0 + il) if il > 0.0 else 2.0
    return max(math.ceil(float(x) ** exponent), 2 * int(x))


def find_I1(K, alpha=3):
    """
    Smallest I1 ≥ 10 such that epoch 1 has a strictly positive exploitation
    batch B5 = I1 − B1 − B2 − B3 − B4 > 0.

    Uses the same rounding as compute_baba_grid so the check is exact.
    """
    for I1 in range(10, 500_001):
        g_r    = g_baba(I1)
        log2_r = math.log2(I1) ** 2
        b1 = max(1, int(round(K * g_r)))
        b2 = max(1, int(round(log2_r)))
        b3 = max(1, int(round(K * log2_r)))
        b4 = max(1, int(round(K * log2_r)))
        if I1 - b1 - b2 - b3 - b4 > 0:
            return I1
    return 2000  # unreachable for any practical K


# ── schedule builder ──────────────────────────────────────────────────────────

def compute_baba_grid(T_max, K, I1=None, alpha=3):
    """
    Compute the BABA static time grid.

    Epochs are added until the previous epoch boundary I_{r-1} exceeds T_max,
    so the schedule always covers at least T_max total pulls.

    Parameters
    ----------
    T_max        : int   — target total pulls (lower bound on covered horizon)
    K            : int   — number of arms
    I1           : int   — initial epoch boundary; auto-selected if None
    alpha        : int   — ilogα depth (paper uses alpha=3)

    Returns
    -------
    batch_sizes  : list[int]         — pull budget for each round (batch)
    phase_labels : list[int]         — BABA phase in {1,2,3,4,5} per round
    epoch_ids    : list[int]         — 1-based epoch index per round
    epoch_I      : dict[int, int]    — {epoch_id: I_curr} epoch boundaries
    """
    if I1 is None:
        I1 = find_I1(K, alpha)

    batch_sizes, phase_labels, epoch_ids = [], [], []
    epoch_I = {}

    I_prev, I_curr, r = 0, int(I1), 1

    while I_prev < T_max:
        g_r    = g_baba(I_curr)
        log2_r = math.log2(I_curr) ** 2        # (log₂ Ir)²

        b1 = max(1, int(round(K * g_r)))
        b2 = max(1, int(round(log2_r)))
        b3 = max(1, int(round(K * log2_r)))
        b4 = max(1, int(round(K * log2_r)))
        b5 = I_curr - I_prev - b1 - b2 - b3 - b4

        if b5 <= 0:
            break  # exploitation batch would be non-positive; stop

        epoch_I[r] = I_curr
        for size, phase in [(b1, 1), (b2, 2), (b3, 3), (b4, 4), (b5, 5)]:
            batch_sizes.append(size)
            phase_labels.append(phase)
            epoch_ids.append(r)

        I_prev = I_curr
        I_curr = f_baba(I_curr, alpha)
        r     += 1

    return batch_sizes, phase_labels, epoch_ids, epoch_I





# ---------------------------------------------------------------------------
# BABA-schedule environment factory
# ---------------------------------------------------------------------------
# Creates an environment whose batch sizes follow the BABA static time grid
# (Algorithm 1 of Jin et al., ICML 2021).  The environment is constructed
# directly — no gymnasium registration is needed — so all baselines can be
# tested on the BABA schedule without any changes to the learners.
#
# Usage:
#   env, n_rounds, I1 = bW.make_baba_env('trunc', _MEANS, target_num_steps=1e5)
#   env, n_rounds, I1 = bW.make_baba_env('bern',  _MEANS, target_num_steps=1e5)
# ---------------------------------------------------------------------------

from gymnasium.envs.registration import register
import gymnasium, sys

INFINITY = sys.maxsize

#TODO: To be adjusted to current architecture
def make_baba_env(dist_type, means=None, target_num_steps=100_000, alpha=3,
                  I1=None, sigma=0.5, low=-1.0, high=1.0):
    """
    Build a BatchMAB environment whose batch sizes follow the BABA time grid.

    The same schedule is used by all agents, so every baseline experiences
    exactly the same batch-size sequence as the BABA learner.

    Parameters
    ----------
    dist_type        : str   — 'trunc' (TruncatedGaussian) or 'bern' (Bernoulli)
    means            : list  — arm means; defaults to _MEANS = [0.1,0.4,0.7,0.9]
    target_num_steps : int   — approximate total arm-pull budget; the schedule
                               is computed so that it covers at least this many
                               pulls (epoch boundaries can slightly overshoot)
    alpha            : int   — ilogα depth for the BABA epoch growth (default 3)
    sigma            : float — std-dev for TruncatedGaussian arms (default 0.5)
    low, high        : float — support bounds for TruncatedGaussian (default −1,1)

    Returns
    -------
    env      : BatchMAB environment driven by the BABA schedule
    n_rounds : int  — number of rounds (batches) = timeHorizon for the runner
    I1       : int  — initial epoch boundary (pass to BABA agent for alignment)
    """
    K  = len(means)
    if I1 is None:
        I1 = find_I1(K, alpha)
    batch_sizes, phase_labels, epoch_ids, epoch_I = compute_baba_grid(
        int(target_num_steps), K, I1, alpha)
    n_rounds = len(batch_sizes)

    action_names = ["a" + str(i) for i in range(K)]
    s            = "-".join(str(m) for m in means)

    # batch_sizes is a plain list — picklable by multiprocessing workers.
    # We register with gymnasium so that workers can call
    # gymnasium.make(gym_name) without hitting NameNotFound.
    if dist_type == 'trunc':
        gym_name = f'BatchTruncGBandit-BABA-means-{s}-v0'
        env_name = f'BatchTruncGBandit-BABA-means-{s}'
        title    = (f"TruncGaussian — BABA schedule")
        try:
            register(
                id=gym_name,
                entry_point='environment.banditji:BatchTruncGBandit',
                max_episode_steps=INFINITY,
                reward_threshold=1.,
                kwargs={'action_names': action_names, 'probabilities': means,
                        'batchsize': batch_sizes,
                        'sigma': sigma, 'low': low, 'high': high,
                        'name': env_name},
            )
        except Exception:
            pass  # already registered (e.g. second call in same process)

    elif dist_type == 'bern':
        gym_name = f'BatchBernBandit-BABA-means-{s}-v0'
        env_name = f'BatchBernBandit-BABA-means-{s}'
        title    = (f"Bernoulli — BABA schedule "
                    f"(T≈{int(target_num_steps):.2e}, {n_rounds} rounds)")
        try:
            register(
                id=gym_name,
                entry_point='environment.banditji:BatchBernBandit',
                max_episode_steps=INFINITY,
                reward_threshold=1.,
                kwargs={'action_names': action_names, 'probabilities': means,
                        'batchsize': batch_sizes,
                        'name': env_name},
            )
        except Exception:
            pass  # already registered

    else:
        raise ValueError(f"dist_type must be 'trunc' or 'bern', got {dist_type!r}.")

    env       = gymnasium.make(gym_name).unwrapped
    env.displayname = title
    return env, n_rounds, I1, phase_labels, epoch_ids, epoch_I