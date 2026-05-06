from main import load_yml_config
import yaml
import pandas as pd
from pathlib import Path
from typing import List


YAML_PATH = "config.yml"
OUTPUT_DIR = "L2_emot_state_estimation/output_data"


def load_yml_config(path: str | Path = YAML_PATH) -> dict:
    """Parse a YAML file and return the Python object it represents."""
    path = Path(path).expanduser()
    dict = {}
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dict["feather_path"] = yml_data["feather_file_path"]
    dict["POW_COLUMNS"] = yml_data["POW_COLUMNS"]

    return dict


class features:
    def __init__(self, pow: List[float]):
        self.pow_columns = load_yml_config()["POW_COLUMNS"]
        self.pow = pow
        self.load()
