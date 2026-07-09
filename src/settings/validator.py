"""
Simple validator for StatisticalRL setting folders.

Checks the structure and minimal API requirements described
in the setting guidelines.
"""

from pathlib import Path
import importlib
import yaml
import importlib.util

from experiments.utils import load


# ----------------------------------------------------------------------
# Expected structure
# ----------------------------------------------------------------------

REQUIRED_FILES = [
    "agent.py",
    "environment.py",
    "interaction.py",
    "_test.py",
]

REQUIRED_DIRS = [
    "agents",
    "envs",
    "renderers",
    "wrappers",
]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def report(ok, message):
    symbol = "✓" if ok else "✗"
    print(f"{symbol} {message}")
    return ok


def load_module(module_path, module_name):
    """
    Import a python module from a file path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None:
        raise ImportError(module_name)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ----------------------------------------------------------------------
# Structure validation
# ----------------------------------------------------------------------

def validate_structure(setting_dir):
    """
    Check required files and folders.
    """

    setting_dir = Path(setting_dir)

    success = True

    for filename in REQUIRED_FILES:
        ok = (setting_dir / filename).is_file()
        success &= report(
            ok,
            f"file {filename}"
        )

    for dirname in REQUIRED_DIRS:
        ok = (setting_dir / dirname).is_dir()
        success &= report(
            ok,
            f"directory {dirname}/"
        )

    return success


# ----------------------------------------------------------------------
# Python API validation
# ----------------------------------------------------------------------

def validate_environment(setting_dir):
    """
    Check Environment class and API.
    """

    path = Path(setting_dir) / "environment.py"

    try:
        module = load_module(path, "environment")
    except Exception as e:
        return report(False, f"cannot import environment.py ({e})")

    ok = hasattr(module, "Environment")
    report(ok, "Environment class")

    if not ok:
        return False

    env = module.Environment

    success = True

    for method in ["step", "reset", "render"]:
        success &= report(
            hasattr(env, method),
            f"Environment.{method}()"
        )

    return success


def validate_agent(setting_dir):
    """
    Check Agent class.
    """

    path = Path(setting_dir) / "agent.py"

    try:
        module = load_module(path, "agent")
    except Exception as e:
        return report(False, f"cannot import agent.py ({e})")

    return report(
        hasattr(module, "ABC"),
        "Agent class"
    )


def validate_interaction(setting_dir):
    """
    Check Interaction class and API.
    """

    path = Path(setting_dir) / "interaction.py"

    try:
        module = load_module(path, "interaction")
    except Exception as e:
        return report(False, f"cannot import interaction.py ({e})")

    ok = hasattr(module, "Interaction")

    report(ok, "Interaction class")

    if not ok:
        return False

    cls = module.Interaction

    success = True

    for method in ["run", "renderrun"]:
        success &= report(
            hasattr(cls, method),
            f"Interaction.{method}()"
        )

    return success


# ----------------------------------------------------------------------
# Environment YAML validation
# ----------------------------------------------------------------------

def validate_environment_yaml(setting_dir):
    """
    Validate envs/environment.yaml files.
    """

    path = Path(setting_dir) / "envs" / "environments.yaml"

    if not path.is_file():
        return report(False, "envs/environments.yaml")

    try:
        data = load(path)
    except Exception as e:
        return report(False, f"invalid yaml ({e})")

    ok = isinstance(data, dict)

    report(ok, "environment yaml format")

    if not ok:
        return False

    success = True

    for name, spec in data.items():

        if "entrypoint" not in spec:
            success &= report(
                False,
                f"{name}: missing entrypoint"
            )
            continue

        if "kwargs" not in spec:
            success &= report(
                False,
                f"{name}: missing kwargs"
            )

        try:
            module_name, class_name = spec["entrypoint"].split(":")
            module_path = Path(spec["_base_dir"]) / f"{module_name}.py"

            spec_module = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec_module)
            spec_module.loader.exec_module(module)

            ok = hasattr(module, class_name)

            success &= report(
                ok,
                f"{name}: entrypoint {class_name}"
            )

        except Exception as e:
            success &= report(
                False,
                f"{name}: cannot load entrypoint ({e})"
            )

    return success


# ----------------------------------------------------------------------
# Main validator
# ----------------------------------------------------------------------

def validate_setting(setting_dir):
    """
    Validate a complete setting folder.
    """

    print("=" * 60)
    print(f"Validating setting: {setting_dir}")
    print("=" * 60)

    results = [
        validate_structure(setting_dir),
        validate_environment(setting_dir),
        validate_agent(setting_dir),
        validate_interaction(setting_dir),
        validate_environment_yaml(setting_dir),
    ]

    print("=" * 60)

    if all(results):
        print("SETTING VALID")
    else:
        print("SETTING INVALID")

    return all(results)


# ----------------------------------------------------------------------
# Command line usage
# ----------------------------------------------------------------------

if __name__ == "__main__":

    #import sys

    # if len(sys.argv) != 2:
    #     print(
    #         "Usage: python validator.py <setting_folder>"
    #     )
    #     exit(1)

    #validate_setting(sys.argv[1])

    validate_setting("bandits/stochastic/knownhorizon")