"""File operation utilities."""
import glob
import os
import shutil
from typing import List, Optional


def find_input_files(input_dir: str, pattern: Optional[str], limit: Optional[int]) -> List[str]:
	"""Find input files matching the pattern."""
	if pattern is None:
		pattern = "*"
	
	recursive = "**" in pattern
	if recursive:
		paths = sorted(glob.glob(os.path.join(input_dir, pattern), recursive=True))
	else:
		paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
	
	paths = [p for p in paths if os.path.isfile(p)]
	
	if limit is not None:
		paths = paths[: int(limit)]
	return paths


def ensure_dir(path: str) -> None:
	"""Ensure directory exists."""
	os.makedirs(path, exist_ok=True)


def confirm_delete(output_dir: str) -> bool:
	"""Ask user if they want to delete existing output directory."""
	if not os.path.exists(output_dir):
		return True
	
	if os.path.isdir(output_dir) and os.listdir(output_dir):
		response = input(
			f"Output directory '{output_dir}' already exists and contains files.\n"
			"Delete it and continue? (y/n): "
		).strip().lower()
		if response in ("y", "yes"):
			shutil.rmtree(output_dir)
			return True
		return False
	return True

