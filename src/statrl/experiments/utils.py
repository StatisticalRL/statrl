
import importlib
import importlib.util
import os
import pickle
from pathlib import Path
from typing import Any

import yaml


def dump(values: Any, filename: str, tag: str, root_folder: str) -> str:
    """Pickle a value to ``{root_folder}{filename}_{tag}``.

    Parameters
    ----------
    values : object
        Anything picklable; in practice a score time series.
    filename : str
        Filename prefix. ``"aux"`` marks a file
        :func:`clear_auxiliaryfiles` may later delete.
    tag : str
        Suffix identifying the run, built from the environment, agent,
        horizon, and a timestamp.
    root_folder : str
        Output directory. Must end with a separator and already exist.

    Returns
    -------
    str
        Path of the file written.
    """
    filenameM = f"{root_folder}{filename}_{tag}"
    with open(filenameM, 'wb') as file:
        pickle.dump(values, file)
    return filenameM

def clear_auxiliaryfiles(env: Any, root_folder: str) -> None:
    """Delete the intermediate per-replicate dumps of one environment.

    Removes every file in ``root_folder`` whose name starts with
    ``aux_{env.name}``, i.e. the per-replicate score series, once the regret
    statistics have been computed from them. The regret pickles, logfile, and
    figures are kept.

    Parameters
    ----------
    env : object
        Environment whose dumps to remove, identified by its ``name``.
    root_folder : str
        Directory to clean. Must end with a separator.

    Warnings
    --------
    Deletes files unconditionally. Anything of yours in ``root_folder`` named
    ``aux_{env.name}*`` will be removed too.
    """
    for file in os.listdir(root_folder):
        if file.startswith("aux_" + env.name):
            os.remove(root_folder + file)

#def load(filename):
#    with open(filename, 'r') as file:
#        return yaml.safe_load(file)
def load(filename: str) -> dict:
    """Read an ``environments.yaml`` registry.

    Each entry maps an environment name to a spec with an ``entrypoint`` and
    optional ``kwargs``. The spec's directory is recorded under ``_base_dir``
    so :func:`make` can resolve the entrypoint relative to the YAML file
    rather than the working directory.

    Parameters
    ----------
    filename : str or path-like
        Path to the YAML registry.

    Returns
    -------
    dict
        The parsed registry, each spec augmented with ``_base_dir``.

    See Also
    --------
    make : Instantiates one of the returned specs.
    """
    path = Path(filename)

    with path.open("r") as f:
        envs = yaml.safe_load(f)

    for spec in envs.values():
        spec["_base_dir"] = path.parent

    return envs

def make(spec: dict) -> Any:
    """Instantiate an environment from a registry spec.

    Imports the module named by the spec's ``entrypoint`` from the file it
    sits next to and calls the class it names with the spec's ``kwargs``.
    Loading by file path rather than by import name is what lets a setting's
    ``envs/`` directory be a plain folder, with no package or installation.

    Parameters
    ----------
    spec : dict
        A registry entry as returned by :func:`load`, with keys
        ``entrypoint`` (``"module:Class"``), ``_base_dir``, and optionally
        ``kwargs`` and ``displayname``.

    Returns
    -------
    object
        The constructed environment, carrying a ``displayname`` attribute —
        the spec's if given, otherwise its ``name``. That value becomes the
        title of the regret figures.

    Raises
    ------
    ImportError
        If the module named by the entrypoint cannot be loaded.

    See Also
    --------
    load : Reads the registry this spec comes from.
    """
    module_name, class_name = spec["entrypoint"].split(":")
    kwargs = spec.get("kwargs", {})

    module_path = Path(spec["_base_dir"]) / f"{module_name}.py"

    spec_module = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_module is None or spec_module.loader is None:
        raise ImportError(f"cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)

    cls = getattr(module, class_name)
    env = cls(**kwargs)
    env.displayname = spec["displayname"] if "displayname" in spec else env.name
    return env




def _load_local_entrypoint(entrypoint: str, base_dir: str) -> Any:
    """Load an object from a Python file relative to the registry."""
    module_name, object_name = entrypoint.split(":", maxsplit=1)

    module_path = Path(base_dir) / f"{module_name}.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load module {module_name} from {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(
            f"Cannot find {object_name} in module {module_name}"
        ) from exc


def _load_import(entrypoint: str) -> Any:
    """Load an object using its absolute Python import path."""
    module_name, object_name = entrypoint.split(":", maxsplit=1)
    module = importlib.import_module(module_name)

    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(
            f"Cannot find {object_name} in module {module_name}"
        ) from exc

def makealgo(spec: dict, env: Any) -> Any:
    module_name, class_name = spec["entrypoint"].split(":", maxsplit=1)

    module_path = Path(spec["_base_dir"]) / f"{module_name}.py"

    module_spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    cls = getattr(module, class_name)

    def resolve_value(value):
        if isinstance(value, dict):
            if "env" in value:
                return getattr(env, value["env"])

            if "import" in value:
                return _load_import(value["import"])

            if "callable" in value:
                factory_spec = value["callable"]
                factory = getattr(module, factory_spec["name"])
                return factory(**factory_spec.get("kwargs", {}))

        return value

    kwargs = {
        key: resolve_value(value)
        for key, value in spec.get("kwargs", {}).items()
    }

    algo = cls(**kwargs)
    algo.displayname = spec.get("displayname", algo.name)

    return algo