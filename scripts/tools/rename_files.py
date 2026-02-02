#!/usr/bin/env python3
"""Rename files by replacing 'teaser' with 'dec2024_teaser'.

Usage:
    python scripts/rename_files.py /path/to/folder
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPLACEMENT = ("teaser", "dec2024_teaser")


def rename_teaser_files(root: Path) -> list[tuple[Path, Path]]:
    """Rename files under ``root`` whose names contain ``REPLACEMENT[0]``.

    Args:
        root: Directory to walk.

    Returns:
        A list of tuples mapping original paths to new paths for renamed files.
    """
    assert root.exists(), f"Provided path '{root}' does not exist."
    assert root.is_dir(), f"Provided path '{root}' is not a directory."

    renamed_paths: list[tuple[Path, Path]] = []
    old, new = REPLACEMENT
    for path in sorted(root.rglob("*")):
        if not path.is_file() or old not in path.name:
            continue

        updated_name = path.name.replace(old, new)
        updated_path = path.with_name(updated_name)

        if updated_path.exists() and updated_path != path:
            assert False, (
                "Refusing to overwrite existing path: "
                f"'{updated_path}'. Rename files manually to proceed."
            )

        path.rename(updated_path)
        renamed_paths.append((path, updated_path))

    return renamed_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively rename files so that 'teaser' in the filename is "
            "replaced with 'dec2024_teaser'."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to scan for files that need renaming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renamed = rename_teaser_files(args.folder.resolve())

    if not renamed:
        print("No files required renaming.")
    else:
        print("Renamed the following files:")
        for original, updated in renamed:
            print(f"  {original} -> {updated}")


if __name__ == "__main__":
    main()
