import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yml"


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@lru_cache(maxsize=1)
def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    
    parser = argparse.ArgumentParser(description="Configuration overrides")
    
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=_str2bool, default=value)
        elif isinstance(value, (int, float, str)):
            parser.add_argument(f"--{key}", type=type(value), default=value)
            
    args, _ = parser.parse_known_args()
    
    for key in config:
        if hasattr(args, key):
            config[key] = getattr(args, key)
            
    return config


def get_config(keys: list[str], path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _load_config(path)
    return {key: config[key] for key in keys}
