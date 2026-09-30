"""Dataset Converter - Main entry point."""
import argparse
import os
import sys

# Make the project root importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from src.dataset_converter.config import load_config
from src.dataset_converter.converters.cluster import ClusterDatasetConverter
from src.dataset_converter.converters.base import BaseConverter

# The converter is chosen by which dataset-specific section the config carries.
CONVERTERS = {
	"cluster_dataset": ClusterDatasetConverter,
}


def main() -> None:
	parser = argparse.ArgumentParser(description="Dataset Converter")
	parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
	args = parser.parse_args()
	
	cfg = load_config(args.config)
	sections = [key for key in CONVERTERS if cfg.get(key) is not None]
	if len(sections) != 1:
		raise ValueError(
			f"{args.config} must contain exactly one of {sorted(CONVERTERS)} sections, "
			f"found {sections or 'none'}"
		)
	converter: BaseConverter = CONVERTERS[sections[0]](cfg)
	converter.run()


if __name__ == "__main__":
	main()
