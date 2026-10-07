"""Load an experiment config = configs/base.yaml + configs/<data>.yaml + command-line overrides.

    python scripts/train.py --data levir                     # defaults
    python scripts/train.py --data hiucd --epochs 30 --seed 1 --augment false
    python scripts/train.py --data=hiucd --seed=1            # '=' form (used by W&B sweeps)

Every override is parsed as YAML, so 30 -> int, 3e-4 -> float, false -> bool.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repository root
CONFIG_DIR = ROOT / "configs"


def _parse_overrides(argv):
    """['--a', '1', '--b=2'] -> {'a': 1, 'b': 2}"""
    out, i = {}, 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected argument {tok!r} (expected --key value or --key=value)")
        if "=" in tok:
            key, val = tok[2:].split("=", 1)
            i += 1
        else:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for {tok}")
            key, val = tok[2:], argv[i + 1]
            i += 2
        out[key.replace("-", "_")] = _to_value(val)
    return out


def _to_value(text):
    val = yaml.safe_load(text)
    if isinstance(val, str):  # YAML reads "3e-4" (no dot) as a string
        try:
            return float(val)
        except ValueError:
            pass
    return val


def _read_yaml(path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_config(argv=None):
    overrides = _parse_overrides(sys.argv[1:] if argv is None else argv)
    data = overrides.pop("data", None)
    if data is None:
        raise ValueError("Missing --data (e.g. --data hiucd or --data levir)")

    cfg = _read_yaml(CONFIG_DIR / "base.yaml")
    cfg.update(_read_yaml(CONFIG_DIR / f"{data}.yaml"))

    unknown = set(overrides) - set(cfg)
    if unknown:  # catches typos like --epoch instead of --epochs
        raise ValueError(f"Unknown config key(s): {sorted(unknown)}. Known: {sorted(cfg)}")
    cfg.update(overrides)
    cfg["data"] = data
    return cfg
