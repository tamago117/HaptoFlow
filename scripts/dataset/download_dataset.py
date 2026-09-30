"""Download the Cluster Haptic Texture Dataset from the Hugging Face Hub."""
import argparse
import os
import sys

# Make the project root importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from src.utils.data.hf_dataset import (
	ALL_SUBSETS,
	DEFAULT_LOCAL_DIR,
	HF_DATASET_REPO_ID,
	build_allow_patterns,
	download_dataset,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download the Cluster Haptic Texture Dataset from the Hugging Face Hub."
	)
	parser.add_argument(
		"--subset",
		type=str,
		default="all",
		help=f"Comma-separated subset names ({', '.join(ALL_SUBSETS)}) or 'all' (default: all)",
	)
	parser.add_argument(
		"--textures",
		type=str,
		default=None,
		help="Comma-separated texture ids to restrict to (default: every texture)",
	)
	parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID, help="Hub dataset repo id")
	parser.add_argument(
		"--local-dir",
		type=str,
		default=DEFAULT_LOCAL_DIR,
		help=f"Destination directory (default: {DEFAULT_LOCAL_DIR})",
	)
	parser.add_argument("--revision", type=str, default=None, help="Branch, tag or commit to pin")
	parser.add_argument("--list-subsets", action="store_true", help="Print the available subsets and exit")
	parser.add_argument("--dry-run", action="store_true", help="Print the glob patterns and exit")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.list_subsets:
		print("all")
		for name in ALL_SUBSETS:
			print(name)
		return

	patterns = build_allow_patterns(args.subset, args.textures)
	if args.dry_run:
		print("allow_patterns:", patterns if patterns is not None else "<entire repo>")
		return

	path = download_dataset(
		subset=args.subset,
		textures=args.textures,
		repo_id=args.repo_id,
		local_dir=args.local_dir,
		revision=args.revision,
	)
	print(f"Dataset available at: {path}")


if __name__ == "__main__":
	main()
