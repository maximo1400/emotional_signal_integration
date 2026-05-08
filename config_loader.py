from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yml"


@lru_cache(maxsize=1)
def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_config(keys: list[str], path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _load_config(path)
    return {key: config[key] for key in keys}
