from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from statrl.settings.bandits.stochastic.kernel.environment import KernelBanditEnv


class PlotRenderer:
    """Matplotlib renderer for Kernel Bandit environments."""

    def __init__(self, pause: float = 0.1) -> None:
        self.pause = pause
        self.fig = None
        self.ax = None
        self.started = False

    def start(self, env: KernelBanditEnv) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.outfile = None

        self._plot_environment(env)

        plt.ion()
        plt.show(block=False)
        plt.pause(self.pause)

    def stop(self, env: KernelBanditEnv) -> None:
        if self.fig is not None:
            plt.ioff()
            plt.show()
            plt.close(self.fig)

        self.started = False

    def render(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:

        if not self.started:
            self.start(env)
            self.started = True
            return

        self._plot_environment(env, last)

        plt.pause(self.pause)

    def _plot_environment(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float] = (None, 0.0),
    ) -> None:

        self.ax.clear()

        dimension = env.arm_features.shape[1]

        if dimension == 1:
            self._plot_1d(env, last)

        elif dimension == 2:
            self._plot_2d(env, last)

        else:
            self._plot_high_dimensional(env, last)

        self.ax.set_title(env.displayname)
        self.fig.canvas.draw_idle()

    def _plot_1d(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:

        x = env.arm_features[:, 0]
        f = env.means

        self.ax.plot(x, f, label="latent function")

        # Optimal arm
        optimal = env.optimal_arm
        self.ax.scatter(
            x[optimal],
            f[optimal],
            marker="*",
            s=150,
            label="optimal",
        )

        # Last selected arm
        lastaction, lastreward = last

        if lastaction is not None:
            self.ax.scatter(
                x[lastaction],
                f[lastaction],
                s=80,
                label=f"selected (r={lastreward:.2f})",
            )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.legend()

    def _plot_2d(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:

        x = env.arm_features[:, 0]
        y = env.arm_features[:, 1]
        f = env.means

        scatter = self.ax.scatter(
            x,
            y,
            c=f,
            s=40,
        )

        self.fig.colorbar(
            scatter,
            ax=self.ax,
            label="f(x)",
        )

        # Optimal arm
        optimal = env.optimal_arm

        self.ax.scatter(
            x[optimal],
            y[optimal],
            marker="*",
            s=200,
            edgecolors="black",
            label="optimal",
        )

        # Last selected arm
        lastaction, lastreward = last

        if lastaction is not None:
            self.ax.scatter(
                x[lastaction],
                y[lastaction],
                s=100,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                label=f"selected (r={lastreward:.2f})",
            )

        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$")
        self.ax.legend()

    def _plot_high_dimensional(
        self,
        env: KernelBanditEnv,
        last: tuple[Optional[int], float],
    ) -> None:

        lastaction, lastreward = last

        self.ax.axis("off")

        text = (
            f"Kernel Bandit\n\n"
            f"Dimension: {env.arm_features.shape[1]}\n"
            f"Number of arms: {env.number_arms}\n"
            f"Optimal arm: {env.optimal_arm}\n"
            f"Optimal value: {env.optimal_mean:.3f}\n"
        )

        if lastaction is not None:
            text += (
                f"\nLast arm: {lastaction}\n"
                f"Reward: {lastreward:.3f}"
            )

        self.ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=14,
        )