#!/usr/bin/env python3
import argparse
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm


def collect_laz_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.laz"))
    if not files:
        raise FileNotFoundError(f"No .laz files found in: {folder}")
    return files


def convert_laz_folder_to_xyz(
    input_dir: Path,
    output_dir: Path,
    xyz_fmt: str = "%.6f",
    delimiter: str = " ",
) -> None:
    laz_files = collect_laz_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for laz_file in tqdm(laz_files, desc="Converting LAZ -> XYZ"):
        las = laspy.read(str(laz_file))
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        out_path = output_dir / f"{laz_file.stem}.xyz"
        np.savetxt(out_path, xyz, fmt=xyz_fmt, delimiter=delimiter)

    print(f"Converted {len(laz_files)} files to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert all .laz files in folder to .xyz.")
    parser.add_argument("input_dir", type=Path, help="Folder with .laz files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for .xyz files (default: <input_dir>/xyz).",
    )
    parser.add_argument(
        "--xyz-fmt",
        type=str,
        default="%.6f",
        help="Floating point format for xyz export (default: %%.6f).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=" ",
        help="Delimiter for xyz export (default: space).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a directory: {input_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "xyz"
    )

    convert_laz_folder_to_xyz(
        input_dir=input_dir,
        output_dir=output_dir,
        xyz_fmt=args.xyz_fmt,
        delimiter=args.delimiter,
    )


if __name__ == "__main__":
    main()
