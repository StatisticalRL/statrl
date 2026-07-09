
#" Environment interface are the same, only agents interface are different"
import settings.bandits.stochastic.anytime.agent as agentA
import settings.bandits.stochastic.knownhorizon.agent as agentK
from typing import Any

class AnytimeToKnownHorizonAgentWrapper(
        agentK.BanditAgent):

    def __init__(self, learner: agentA.BanditAgent) -> None:
        self.learner = learner
        super().__init__(self.learner.name+"-anytime")

    def reset(self, horizon: int) -> None:
        self.learner.reset()

    def select_arm(self) -> int:
        return self.learner.select_arm()

    def update(self, arm: int, reward: float) -> None:
        self.learner.update(arm, reward)


    @property
    def policy(self) -> Any:
        return self.learner.policy  # type: ignore[attr-defined]  # policy is oracle-specific, not on the base agent interface


class KnownHorizonToAnytimeAgentWrapper(
        agentA.BanditAgent):

    def __init__(self, learner: agentK.BanditAgent, horizon: int, name: str) -> None:
        """
        Fixes an internal horizon at initilization.
        """
        self.learner = learner
        self.horizon= horizon
        super().__init__(name)

    def reset(self) -> None:
        self.learner.reset(self.horizon)

    def select_arm(self) -> int:
        return self.learner.select_arm()

    def update(self, arm: int, reward: float) -> None:
        self.learner.update(arm, reward)

    @property
    def policy(self) -> Any:
        return self.learner.policy 

