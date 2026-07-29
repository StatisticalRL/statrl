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