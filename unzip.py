import gzip
import shutil
from pathlib import Path


def decompress_gz(input_file: Path, output_file: Path):
    """Decompress a .gz file to the specified output path and remove the original .gz file."""
    with gzip.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    input_file.unlink()  # Remove the original .gz file
    print(f"Decompressed and removed: {input_file} -> {output_file}")


def main(input_dir: str, output_dir: str):
    """
    Recursively find and decompress all .gz files under input_dir into output_dir.

    Args:
        input_dir: Directory containing .gz files.
        output_dir: Directory where decompressed files will be saved.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for gz_file in input_path.rglob("*.gz"):
        # Determine relative path to preserve directory structure
        rel_path = gz_file.relative_to(input_path).with_suffix("")
        dest_file = output_path / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        decompress_gz(gz_file, dest_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decompress all .gz files in a directory recursively and remove originals."
    )
    parser.add_argument("input_dir", help="Path to directory containing .gz files")
    parser.add_argument(
        "output_dir", help="Path to directory to save decompressed files"
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
