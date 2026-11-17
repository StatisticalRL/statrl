"""Discovery and native environment loading for the STATRL playground."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any


class Catalog:
    """Inspect the ``statrl.settings`` package hierarchy.

    A selectable setting is a package containing
    ``envs/environments.yaml``.  The environment registry itself is always
    read through STATRL's canonical ``load`` function, and environments are
    constructed through its canonical ``make`` function.
    """

    def __init__(self, settings_root: Path | None = None):
        # Kept for compatibility with the prototype. Package inspection is
        # used as the source of truth, rather than this filesystem path.
        self.settings_root = Path(settings_root) if settings_root else None
        self._setting_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Settings / navigation tree
    # ------------------------------------------------------------------

    def settings(self) -> list[dict[str, Any]]:
        ids = self._discover_setting_ids()
        return [
            {
                "id": setting_id,
                "label": " > ".join(
                    self._pretty_part(p) for p in setting_id.split(".")
                ),
                "name": self._pretty_part(setting_id.rsplit(".", 1)[-1]),
            }
            for setting_id in ids
        ]

    def settings_tree(self) -> list[dict[str, Any]]:
        """Return the selectable settings as a hierarchical navigation tree."""
        setting_ids = set(self._discover_setting_ids())
        root = {"id": "", "label": "STATRL", "children": []}
        nodes: dict[str, dict[str, Any]] = {"": root}

        for setting_id in sorted(setting_ids):
            parts = setting_id.split(".")
            parent_id = ""
            for i, part in enumerate(parts):
                current_id = ".".join(parts[: i + 1])
                if current_id not in nodes:
                    node = {
                        "id": current_id,
                        "label": self._pretty_part(part),
                        "children": [],
                        "selectable": current_id in setting_ids,
                    }
                    nodes[current_id] = node
                    nodes[parent_id]["children"].append(node)
                parent_id = current_id

        self._sort_tree(root)
        return root["children"]

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------

    def environments(self, setting_id: str) -> list[dict[str, Any]]:
        envs = self._load_environment_catalog(setting_id)
        return [
            {
                "id": str(name),
                "label": str(spec.get("displayname", name))
                if isinstance(spec, dict)
                else str(name),
            }
            for name, spec in envs.items()
        ]

    def make_environment(self, setting_id: str, environment_id: str):
        envs = self._load_environment_catalog(setting_id)
        if environment_id not in envs:
            raise ValueError(
                f"Unknown environment {environment_id!r} "
                f"for setting {setting_id!r}."
            )

        from statrl.experiments.utils import make

        return make(envs[environment_id])

    def _load_environment_catalog(self, setting_id: str) -> dict:
        if setting_id not in self._discover_setting_ids():
            raise ValueError(f"Unknown setting: {setting_id}")

        module = importlib.import_module(f"statrl.settings.{setting_id}")
        package_paths = list(getattr(module, "__path__", []))
        if not package_paths:
            raise RuntimeError(f"Setting is not a package: {setting_id}")

        env_path = Path(package_paths[0]) / "envs" / "environments.yaml"
        if not env_path.is_file():
            raise FileNotFoundError(
                f"Environment catalogue not found: {env_path}"
            )

        from statrl.experiments.utils import load

        return load(str(env_path))

    # ------------------------------------------------------------------
    # Learners
    # ------------------------------------------------------------------

    def learners(self, setting_id: str) -> list[dict[str, Any]]:
        setting = self._package_path(setting_id)
        agents_dir = setting / "agents"
        if not agents_dir.exists():
            return []

        result = []
        for path in sorted(agents_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            result.append(
                {
                    "id": path.stem,
                    "label": path.stem,
                    "module": (
                        f"statrl.settings.{setting_id}.agents.{path.stem}"
                    ),
                    # STATRL agent modules conventionally export a class with
                    # the same name as the module (for example IMED.py).
                    "class": (
                        f"statrl.settings.{setting_id}.agents."
                        f"{path.stem}:{path.stem}"
                    ),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Package inspection
    # ------------------------------------------------------------------

    def _discover_setting_ids(self) -> list[str]:
        if self._setting_ids is not None:
            return self._setting_ids

        import statrl.settings as settings_package

        result: list[str] = []
        prefix = settings_package.__name__ + "."

        # Walk the package tree without importing every submodule.  This is
        # more robust than pkgutil.walk_packages(), which imports packages as
        # it descends and can therefore fail because of an unrelated optional
        # dependency in one setting.
        queue: list[tuple[str, Path]] = []
        for package_path in getattr(settings_package, "__path__", []):
            queue.append((prefix[:-1], Path(package_path)))

        while queue:
            parent_name, parent_path = queue.pop(0)
            try:
                entries = list(pkgutil.iter_modules([str(parent_path)]))
            except OSError:
                continue

            for info in entries:
                if not info.ispkg or info.name.startswith("_"):
                    continue

                module_name = f"{parent_name}.{info.name}"
                child_path = parent_path / info.name

                # A package containing the native environment registry is an
                # actual selectable setting. Other packages are taxonomy
                # nodes and are included automatically when building the tree.
                if (child_path / "envs" / "environments.yaml").is_file():
                    result.append(module_name[len(prefix) :])

                queue.append((module_name, child_path))

        self._setting_ids = sorted(set(result))
        return self._setting_ids

    @staticmethod
    def _package_path(setting_id: str) -> Path:
        module = importlib.import_module(f"statrl.settings.{setting_id}")
        paths = list(getattr(module, "__path__", []))
        if not paths:
            raise ValueError(f"Not a package: {setting_id}")
        return Path(paths[0])

    @classmethod
    def _pretty_part(cls, value: str) -> str:
        return value.replace("_", " ").title()

    @classmethod
    def _sort_tree(cls, node: dict[str, Any]) -> None:
        node["children"].sort(key=lambda x: x["label"].lower())
        for child in node["children"]:
            cls._sort_tree(child)


def import_symbol(spec: str):
    """Import ``module:Symbol`` or ``module.Symbol``."""
    if ":" in spec:
        module_name, symbol = spec.split(":", 1)
    else:
        module_name, symbol = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def constructor_signature(cls):
    try:
        return [
            p.name
            for p in inspect.signature(cls).parameters.values()
            if p.name != "self"
        ]
    except (TypeError, ValueError):
        return []
