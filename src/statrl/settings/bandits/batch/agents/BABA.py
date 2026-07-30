"""
BABA — Batched Anytime Bandit Algorithm  (Jin et al., ICML 2021)
=================================================================
Implements Algorithm 1 (five-phase epochs) adapted to the
batchplay / batchupdate interface used by the rest of the codebase.

Each call to batchplay(B) commits an arm-pull list for the entire batch
without mid-batch adaptation.  The five phases are reconstructed from the
schedule information passed at construction time.

Usage
-----
    from learners.baba_schedule import compute_baba_grid, find_I1
    sizes, phases, epochs, epoch_I = compute_baba_grid(T, K, I1, alpha)

    agent = BABA(nbArms=K, bound=B,
                 phase_labels=phases, epoch_ids=epochs, epoch_I=epoch_I,
                 kl_type='bernoulli')   # or 'gaussian', variance=...
"""

import math
import numpy as np
from statrl.settings.bandits.batch.agent import BatchBanditAgent
from statrl.settings.bandits.batch.agents.baba_schedule import g_baba


# ---------------------------------------------------------------------------
# KL utilities
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _kl_bernoulli(p: float, q: float) -> float:
    """KL divergence kl(p, q) for Bernoulli arms. Returns ∞ if q ∈ {0,1}."""
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    q = float(np.clip(q, _EPS, 1.0 - _EPS))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _kl_gaussian(mu: float, mu_prime: float, V: float) -> float:
    """KL divergence kl(µ, µ') for Gaussian with variance V."""
    return (mu - mu_prime) ** 2 / (2.0 * V)


def _kl_plus(mu: float, mu_star: float, kl_fn) -> float:
    """kl+(µ, µ*) = kl(µ, µ*) · 1[µ ≤ µ*]  (Section 2.3)."""
    return kl_fn(mu, mu_star) if mu <= mu_star else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

from statrl.settings.bandits.batch.agents.baba_schedule import compute_baba_grid
class BABA(BatchBanditAgent):
    """
    Parameters
    ----------
    nbArms       : int
    bound        : float  — upper bound of reward support (passed through,
                            used for KL clipping in Bernoulli mode)
    phase_labels : list[int]  — phase 1..5 per round (from compute_baba_grid)
    epoch_ids    : list[int]  — epoch index (1-based) per round
    epoch_I      : dict[int, int]  — {epoch_id: Ir} epoch boundaries
    kl_type      : 'bernoulli' | 'gaussian'  — parametric KL to use
    variance     : float  — reward variance V (Gaussian KL only, default 0.25)
    """

    def __init__(self, nbArms, horizon=100_000, bound=1.0,
                 phase_labels=None, epoch_ids=None, epoch_I=None,
                 kl_type='bernoulli', variance=0.25, **kwargs):
        self.nbArms       = nbArms
        self.bound        = bound
        self._phases      = phase_labels   # list[int], indexed by round
        self._epoch_ids   = epoch_ids      # list[int], indexed by round
        self._epoch_I     = epoch_I        # dict {epoch_id: Ir}
        self._variance    = variance
        self.T_target = horizon

        if (self._phases == None):
            _,  self._phases, self._epoch_ids, self._epoch_I = compute_baba_grid(
                self.T_target,  self.nbArms, None, alpha=3)

        if kl_type == 'bernoulli':
            self._kl = _kl_bernoulli
        else:
            self._kl = lambda p, q: _kl_gaussian(p, q, variance)

        BatchBanditAgent.__init__(self, name="BABA")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_phase(self):
        return self._phases[self._round % len(self._phases)]

    def _current_Ir(self):
        epoch = self._epoch_ids[self._round % len(self._epoch_ids)]
        return self._epoch_I[epoch]

    def _log2_Ir(self, Ir):
        """(log₂ Ir)² as used in the schedule."""
        return max(1, int(math.log2(Ir) ** 2))

    def _eps_r(self, Ir):
        """εr = 1 / ln(ln(Ir)), clamped away from zero."""
        l1 = math.log(Ir)
        l2 = math.log(l1) if l1 > 1.0 else 1.0
        return 1.0 / max(l2, 0.1)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self):
        self._round = 0

        # Cumulative arm statistics (updated via batchupdate)
        self._counts      = np.zeros(self.nbArms)
        self._sum_rewards = np.zeros(self.nbArms)
        self._means       = np.zeros(self.nbArms)

        # Epoch-level state
        self._cr   = 0      # "best arm so far" — used as fill arm and for exploitation
        self._a1_r = 0      # best arm identified in phase 1
        self._F    = False  # flag from phase 3 (True → uncertain about best arm)
        self._s1   = 0      # count of a1_r after phase 2

        # Snapshots saved at end of phase 1 (used in phases 3 and after)
        self._means_p1  = np.zeros(self.nbArms)   # empirical means after phase 1
        self._counts_p1 = np.zeros(self.nbArms)   # arm counts after phase 1

        # Empirical mean of a1_r after phase 2
        self._mean_a1_p2 = 0.0

    # ------------------------------------------------------------------
    # batchplay — decide which arms to pull this round
    # ------------------------------------------------------------------

    def batchplay(self, B):
        phase = self._current_phase()
        Ir    = self._current_Ir()

        if   phase == 1: return self._play1(B, Ir)
        elif phase == 2: return self._play2(B)
        elif phase == 3: return self._play3(B, Ir)
        elif phase == 4: return self._play4(B, Ir)
        else:            return self._play5(B)

    # ---- Phase 1: UNIFORMEXPLORATION ------------------------------------
    def _play1(self, B, Ir):
        """Pull each arm to g(Ir) total pulls; fill remainder with cr."""
        g    = g_baba(Ir)
        arms = []
        for arm in range(self.nbArms):
            need = max(0, g - int(self._counts[arm]))
            arms.extend([arm] * need)
        # Fill up to B with the current best arm (exploitation fallback)
        while len(arms) < B:
            arms.append(self._cr)
        return arms[:B]

    # ---- Phase 2: INITIALEXPLOITATION -----------------------------------
    def _play2(self, B):
        """Pull a1,r for all B pulls so its mean estimate concentrates."""
        return [self._a1_r] * B

    # ---- Phase 3: OPTIMISTICEXPLORATION ---------------------------------
    def _play3(self, B, Ir):
        """
        For each arm i ≠ a1,r compute δi,r and allocate
            min{δi,r, log²(Ir)} − current_count
        additional pulls.  Remainder goes to cr.
        """
        log2_ir = self._log2_Ir(Ir)
        eps_r   = self._eps_r(Ir)
        mu_a1   = self._mean_a1_p2
        budget  = math.log(Ir * log2_ir)   # numerator in δi,r formula

        arms = []
        for arm in range(self.nbArms):
            if arm == self._a1_r:
                continue

            mu_i = self._means_p1[arm]

            # Shift means away from each other (Eq. 7)
            mu_i_shifted  = mu_i  + eps_r
            mu_a1_shifted = mu_a1 - eps_r

            if mu_i_shifted >= mu_a1_shifted:
                # Arms indistinguishable or i looks better → pull maximally
                delta_ir = log2_ir
            else:
                kl_val   = self._kl(mu_i_shifted, mu_a1_shifted)
                delta_ir = int(budget / kl_val) if kl_val > _EPS else log2_ir

            target = min(delta_ir, log2_ir)
            need   = max(0, target - int(self._counts_p1[arm]))
            arms.extend([arm] * need)

        # Fill remainder with cr
        while len(arms) < B:
            arms.append(self._cr)
        return arms[:B]

    # ---- Phase 4: CONFIDENTEXPLORATION ----------------------------------
    def _play4(self, B, Ir):
        """
        F=False → confident a1,r is best; exploit it.
        F=True  → uncertain; re-explore all arms equally.
        """
        if not self._F:
            return [self._a1_r] * B
        else:
            log2_ir = self._log2_Ir(Ir)
            arms = []
            for arm in range(self.nbArms):
                arms.extend([arm] * log2_ir)
            # Pad to B with the current best arm (schedule rounding may differ
            # from int(log2_ir) by 1; padding ensures we always return exactly B)
            while len(arms) < B:
                arms.append(self._cr)
            return arms[:B]

    # ---- Phase 5: CONFIDENTEXPLOITATION ---------------------------------
    def _play5(self, B):
        """Pull the confirmed best arm cr for the entire exploitation batch."""
        return [self._cr] * B

    # ------------------------------------------------------------------
    # batchupdate — update statistics and advance phase-specific state
    # ------------------------------------------------------------------

    def batchupdate(self, batcharm, batchreward):
        # ── update cumulative arm statistics ──────────────────────────────
        for arm, rew in zip(batcharm, batchreward):
            self._counts[arm]      += 1
            self._sum_rewards[arm] += rew
            if self._counts[arm] > 0:
                self._means[arm] = self._sum_rewards[arm] / self._counts[arm]

        phase = self._current_phase()
        Ir    = self._current_Ir()

        # ── per-phase state transitions ───────────────────────────────────
        if phase == 1:
            # Identify the arm with the highest empirical mean after phase 1
            self._a1_r      = int(np.argmax(self._means))
            self._means_p1  = self._means.copy()
            self._counts_p1 = self._counts.copy()

        elif phase == 2:
            # Snapshot mean and count of a1,r for use in phase 3
            self._s1         = int(self._counts[self._a1_r])
            self._mean_a1_p2 = self._means[self._a1_r]

        elif phase == 3:
            # ── evaluate flag F (Eq. 8) ───────────────────────────────────
            log2_ir   = self._log2_Ir(Ir)
            threshold = math.log(Ir * log2_ir)   # numerator log(Ir · log²(Ir))
            mu_a1     = self._means[self._a1_r]

            self._F = False
            for arm in range(self.nbArms):
                if arm == self._a1_r:
                    continue
                si = max(1, min(log2_ir, int(self._counts[arm])))
                kl_plus_val = _kl_plus(self._means[arm], mu_a1, self._kl)
                if kl_plus_val < threshold / si:
                    self._F = True
                    break

        elif phase == 4:
            # Update the "confirmed best arm" cr for exploitation
            if not self._F:
                self._cr = self._a1_r
            else:
                self._cr = int(np.argmax(self._means))

        # phase 5: no state change; cr carries over to next epoch

        self._round += 1