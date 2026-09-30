"""Export low-res per-label texture thumbnails for the Gradio app."""
import argparse
import os
import sys

# Make the project root importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from src.utils.data.label_images import export_label_images


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="datasets/texture_dataset", help="Dataset root (texture_list.xlsx + images/)")
	parser.add_argument("--output", default="assets/texture_label_images", help="Output directory")
	parser.add_argument("--size", type=int, default=160, help="Max side in pixels")
	args = parser.parse_args()
	n = export_label_images(args.input, args.output, size=args.size)
	print(f"Exported {n} label images to {args.output}")


if __name__ == "__main__":
	main()
