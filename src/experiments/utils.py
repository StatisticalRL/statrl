import os
import pickle
from importlib.metadata import entry_points
from typing import Any


def dump(values: Any, filename: str, tag: str, root_folder: str) -> str:
    filenameM = f"{root_folder}{filename}_{tag}"
    with open(filenameM, 'wb') as file:
        pickle.dump(values, file)
    return filenameM

def clear_auxiliaryfiles(env: Any, root_folder: str) -> None:
    for file in os.listdir(root_folder):
        if file.startswith("aux_" + env.name):
            os.remove(root_folder + file)

import yaml
from pathlib import Path

#def load(filename):
#    with open(filename, 'r') as file:
#        return yaml.safe_load(file)
def load(filename):
    filename = Path(filename)

    with filename.open("r") as f:
        envs = yaml.safe_load(f)

    for spec in envs.values():
        spec["_base_dir"] = filename.parent

    return envs

import importlib.util
def make(spec):
    module_name, class_name = spec["entrypoint"].split(":")
    kwargs = spec.get("kwargs", {})

    module_path = Path(spec["_base_dir"]) / f"{module_name}.py"

    spec_module = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)

    cls = getattr(module, class_name)
    return cls(**kwargs)
