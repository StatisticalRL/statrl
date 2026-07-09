from statrl.settings.bandits.adversarial.lipschitz.agent import Agent
from statrl.settings.bandits.adversarial.lipschitz.environment import LipschitzAdversarialEnv
import numpy as np

def run_lipschitz_online_learning(env: LipschitzAdversarialEnv, learner: Agent) -> np.ndarray:
    """
    Executes the interaction loop between a learner and the
    adversarial Lipschitz environment.

    This is the canonical sequential learning protocol.

    Returns
    -------
    rewards : nparray
        Cumulative reward over the interaction.
    """

    obs, info = env.reset()

    actions = []
    rewards: list[float] = []
    observations = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # ------------------------------------------------------------
        # 1. Learner selects action
        # ------------------------------------------------------------
        action = learner.select_arm(obs)

        # (optional safety clip if using Gym Box space)
        if hasattr(env.action_space, "clip"):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        # ------------------------------------------------------------
        # 2. Environment evaluates adversarial reward
        # ------------------------------------------------------------
        obs_next, reward, terminated, truncated, info = env.step(action)

        # ------------------------------------------------------------
        # 3. Learner updates internal state
        # ------------------------------------------------------------
        learner.update(action, reward, obs)

        # ------------------------------------------------------------
        # 4. Log trajectory
        # ------------------------------------------------------------
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        obs = obs_next

    return np.cumsum(rewards)
# FIXME If we do not return actions and observations, then the function needs simplification
    #return {
    #    "actions": np.array(actions),
    #    "rewards": np.array(rewards),
    #    "observations": observations,
    #}