from typing import Dict, NamedTuple
from pathlib import Path
import os
import yaml


class Dirs(NamedTuple):
    ckpt_dir: str
    plot_dir: str
    log_dir: str


def get_config(path: str) -> Dict:
    """
    Read configuration parameter form file
    :param path: Path to the yaml configuration file
    :return: Dict with parameters
    """
    with open(path, "r") as stream:
        return yaml.load(stream, yaml.FullLoader)


def setup_directories(
    root: str, folder_path: str, config_file_name: str, save: bool, resume: bool
) -> Dirs:
    # Create root dir if not existent
    if not os.path.isdir(root):
        Path(root).mkdir(parents=True, exist_ok=True)

    # Copy config file to root directory, except config file in root dir was read in
    if not os.path.samefile(folder_path, root):
        from_path = os.path.join(folder_path, config_file_name)
        to_path = os.path.join(root, config_file_name)
        print("Copy config file from", folder_path, "to", root)
        os.system(f"cp {from_path} {to_path}")
    # Create subfolders
    if save:
        ckpt_dir = os.path.join(root, "checkpoints")
        plot_dir = os.path.join(root, "plots")
        log_dir = os.path.join(root, "log")
        for directory in [ckpt_dir, plot_dir, log_dir]:
            Path(directory).mkdir(parents=True, exist_ok=resume)
    else:
        ckpt_dir, plot_dir, log_dir = None, None, None

    return Dirs(ckpt_dir=ckpt_dir, plot_dir=plot_dir, log_dir=log_dir)
