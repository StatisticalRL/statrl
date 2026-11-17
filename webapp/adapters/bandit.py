"""Interactive adapter for stochastic bandit learners."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class BanditAdapter:
    """
    Adapter implementing the generic playground protocol for bandits.

    Bandit environments normally have no meaningful state transition, so the
    history contains action/reward pairs and state is kept as None.
    """

    def initial_state(self, env):
        return None

    def unpack_step(self, result):
        # STATRL bandit environments commonly return a reward directly.
        if isinstance(result, tuple):
            if len(result) == 4:
                return result[0], result[1], bool(result[2]), result[3] or {}
            if len(result) == 2:
                return None, result[0], False, {"raw": result[1]}
        return None, result, False, {}

    def reset(self, learner):
        method = getattr(learner, "reset", None)
        if method is not None:
            method()

    def select_action(self, learner, observation=None):
        method = getattr(learner, "select_arm", None)
        if method is None:
            method = getattr(learner, "select_action", None)
        if method is None:
            raise TypeError(
                f"{type(learner).__name__} exposes neither select_arm() "
                "nor select_action()."
            )
        return method()

    def update(self, learner, transition, replay=False):
        method = getattr(learner, "update", None)
        if method is None:
            raise TypeError(f"{type(learner).__name__} has no update() method.")
        method(transition.action, transition.reward)
