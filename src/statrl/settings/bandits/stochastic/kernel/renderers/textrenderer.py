

import sys
import string
import numpy as np

from typing import Optional

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class Textrenderer():

    def __init__(self) -> None:
        self.started = False


    def start(self, env: KernelBanditEnv) -> None:
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.displayname) + "\n")
        self.outfile.write("Actions: "+ str(self._grid_shape(env)) + "\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env: KernelBanditEnv) -> None:
        self.outfile.write("-"*30+"\n")

    def _grid_shape(self, env: KernelBanditEnv) -> tuple[int, ...]:
        return tuple(
            len(np.unique(env.arm_features[:, d]))
            for d in range(env.arm_features.shape[1])
        )

    def _nameActions(self, env: KernelBanditEnv) -> list[str]:
        return [
            self._nameAction(env, arm)
            for arm in range(env.number_arms)
        ]

    def _nameAction(self, env: KernelBanditEnv, arm: int) -> str:

        grid_shape = self._grid_shape(env)

        coordinates = np.unravel_index(
            arm,
            grid_shape,
        )

        letters = string.ascii_uppercase

        return "".join(
            letters[i%26]+("'"*int(i/26))
            for i in coordinates
        )

    def render(self, env: KernelBanditEnv, last: tuple[Optional[int], float]) -> None:
        lastaction, lastreward = last

        if not self.started:
            self.start(env)
            self.started = True

        if lastaction is not None:
            action_name = self._nameAction(env, lastaction)
            self.outfile.write(
                f"({action_name})\tr={lastreward:0.2f}\n"
            )
