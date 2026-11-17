"""Core stateful session logic for the STATRL interactive playground."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Transition:
    """One observed interaction, independent of the learner that produced it."""
    step: int
    state: Any
    action: Any
    reward: Any
    next_state: Any = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class LearnerController:
    """
    Owns the currently selected learner.

    Switching learners never changes the environment or history. A new learner
    is instantiated and replayed on all transitions observed so far.
    """

    def __init__(self, factory: Callable[..., Any], adapter: Any, params=None):
        self.factory = factory
        self.adapter = adapter
        self.params = params or {}
        self.learner = None
        self.attach_count = 0

    def attach(self, history: list[Transition]):
        self.learner = self.factory(**self.params)
        self.attach_count += 1
        self.adapter.reset(self.learner)

        for transition in history:
            self.adapter.update(
                self.learner,
                transition,
                replay=True,
            )

    def select_action(self, observation=None):
        if self.learner is None:
            raise RuntimeError("No learner is attached.")
        return self.adapter.select_action(self.learner, observation)

    def update(self, transition: Transition):
        if self.learner is None:
            return
        self.adapter.update(self.learner, transition, replay=False)


class InteractiveSession:
    """
    Environment-independent interaction session.

    The environment is the source of truth. The history is append-only.
    Learners are consumers of that history and can be replaced at any time.
    """

    def __init__(self, environment, adapter):
        self.environment = environment
        self.adapter = adapter
        self.history: list[Transition] = []
        self.current_state = adapter.initial_state(environment)
        self.learner_controller: LearnerController | None = None

    def attach_learner(self, factory, params=None):
        controller = LearnerController(factory, self.adapter, params)
        controller.attach(self.history)
        self.learner_controller = controller

    def recommended_action(self):
        if self.learner_controller is None:
            raise RuntimeError("No learner selected.")
        return self.learner_controller.select_action(self.current_state)

    def step(self, action, outcome=None):
        """
        Execute one interaction.

        If outcome is None, use the real environment's step(action).
        If outcome is supplied, it is used verbatim. This is the custom
        environment mode and does not call the environment dynamics.
        """
        state = self.current_state

        if outcome is None:
            result = self.environment.step(action)
            next_state, reward, done, info = self.adapter.unpack_step(result)
        else:
            next_state, reward, done, info = self.adapter.unpack_step(outcome)

        transition = Transition(
            step=len(self.history),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
        )
        self.history.append(transition)
        self.current_state = next_state

        if self.learner_controller is not None:
            self.learner_controller.update(transition)

        return transition

    def snapshot(self):
        return {
            "state": self.current_state,
            "history": [
                {
                    "step": t.step,
                    "state": t.state,
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state,
                    "done": t.done,
                    "info": t.info,
                }
                for t in self.history
            ],
            "learner_attached": self.learner_controller is not None,
        }
