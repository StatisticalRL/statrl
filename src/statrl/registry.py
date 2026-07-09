# This module is the bridge between the YAML preset files (plain text, no code)
# and the actual Python classes they describe. Two things happen here:
#   1. load_yaml_registry turns a YAML file into a plain dict.
#   2. resolve_entrypoint turns a dotted string from that dict (e.g.
#      "statrl.settings...BernoulliBandit") into the real, importable class/function.
# SETTINGS at the bottom then ties each experimental "setting" (anytime, known-horizon,
# Lipschitz) to its env/agent YAML files, so calling code never hardcodes file paths.

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


def resolve_entrypoint(dotted_path: str) -> Callable[..., Any]:
    """Resolve 'pkg.mod.Class' or 'pkg.mod.func' to the object it names.""" 
    module_path, sep, attr = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)  # import the module, e.g. "pkg.mod"
    return getattr(module, attr)  # pull "Class" off it


def load_yaml_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a preset YAML file shaped {preset_id: {entrypoint, label, kwargs}}."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)  # yaml.safe_load turns YAML text into nested dicts/lists
    return data or {}  # an empty YAML file loads as None, so fall back to {}


# Describes one experimental "setting", where to find its env/agent YAML preset files, and which Interaction class runs them.
@dataclass
class SettingSpec:
    label: str                       # human-readable name, e.g. "Stochastic (anytime)"
    env_registry: Path                # path to that setting's environments.yaml
    agent_registry: Path               # path to that setting's agents.yaml
    interaction_cls: str                # dotted path to the Interaction subclass that runs env+agent
    agent_wrapper: str | None = None     # optional dotted path to an agent-adapter (see knownhorizon below)


# Build absolute paths to every setting's YAML files, relative to this file's own location,
# so it works regardless of the current working directory the code is run from.
_HERE = Path(__file__).resolve().parent
_ANYTIME_ENV_YAML = _HERE / "settings/bandits/stochastic/anytime/envs/environments.yaml"
_ANYTIME_AGENT_YAML = _HERE / "settings/bandits/stochastic/anytime/agents/agents.yaml"
_LIPSCHITZ_ENV_YAML = _HERE / "settings/bandits/adversarial/lipschitz/envs/environments.yaml"
_LIPSCHITZ_AGENT_YAML = _HERE / "settings/bandits/adversarial/lipschitz/agents/agents.yaml"

# The single source of truth: every experimental setting the codebase supports, and how to
# find/run it. Code elsewhere looks things up as SETTINGS["stochastic_anytime"], etc.,
# instead of hardcoding file paths or import statements.
SETTINGS: dict[str, SettingSpec] = {
    "stochastic_anytime": SettingSpec(
        label="Stochastic (anytime)",
        env_registry=_ANYTIME_ENV_YAML,
        agent_registry=_ANYTIME_AGENT_YAML,
        interaction_cls="statrl.settings.bandits.stochastic.anytime.interaction.BanditInteraction",
    ),
    "stochastic_knownhorizon": SettingSpec(
        # Known-horizon reuses the same env/agent YAML files as anytime (see
        # _ANYTIME_ENV_YAML / _ANYTIME_AGENT_YAML above) -- the agents themselves are
        # written for the "anytime" API. agent_wrapper below adapts one to the other,
        # so the same preset files serve both settings.
        label="Stochastic (known-horizon)",
        env_registry=_ANYTIME_ENV_YAML,
        agent_registry=_ANYTIME_AGENT_YAML,
        interaction_cls="statrl.settings.bandits.stochastic.knownhorizon.interaction.BanditInteraction",
        agent_wrapper=(
            "statrl.settings.bandits.stochastic.knownhorizon.wrappers."
            "wrapper_anytime_knownhorizon.AnytimeToKnownHorizonAgentWrapper"
        ),
    ),
    "adversarial_lipschitz": SettingSpec(
        label="Adversarial (Lipschitz)",
        env_registry=_LIPSCHITZ_ENV_YAML,
        agent_registry=_LIPSCHITZ_AGENT_YAML,
        interaction_cls="statrl.settings.bandits.adversarial.lipschitz.interaction.LipschitzInteraction",
    ),
}
