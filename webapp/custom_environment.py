"""Custom-environment support.

The browser never executes Python supplied by the user. It sends an explicit
step outcome (next state, reward, done, optional info), which is then recorded
by the same session logic used by ordinary environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CustomStep:
    next_state: Any
    reward: Any
    done: bool = False
    info: dict[str, Any] | None = None

    def as_tuple(self):
        return self.next_state, self.reward, self.done, self.info or {}


class CustomEnvironment:
    """Metadata-only environment used in custom-outcome mode."""

    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.state = self.metadata.get("initial_state")

    def step(self, action):
        raise RuntimeError(
            "CustomEnvironment never samples a transition. "
            "Supply the observed outcome explicitly."
        )

    def reset(self):
        self.state = self.metadata.get("initial_state")
        return self.state
