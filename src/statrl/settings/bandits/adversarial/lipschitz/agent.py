from typing import Any, Optional


class Agent:
    def select_arm(self, observation: Any) -> Any:
        """
        Returns an action x_t in continuous action space.
        """
        raise NotImplementedError

    def update(self, action: Any, reward: float, observation: Optional[Any] = None) -> None:
        """
        Optional update step after observing reward.
        """
        pass