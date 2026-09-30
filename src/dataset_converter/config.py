"""Configuration loading utilities."""
import yaml
from typing import Dict


def load_config(path: str) -> Dict:
	"""Load configuration from YAML file."""
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg if cfg is not None else {}

