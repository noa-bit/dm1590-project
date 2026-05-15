#!/usr/bin/env python3
"""Flatten drum sample files into data/raw with source-aware filenames."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "drums"
DEST_DIR = ROOT / "data" / "raw"


def safe_name(value: str) -> str:
    """Return a filesystem-friendly filename component."""
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "unknown"


def iter_source_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*") if path.is_file())


def destination_for(source_file: Path, dest_dir: Path) -> Path:
    source_ref = safe_name(source_file.parent.name)
    stem = safe_name(source_file.stem)
    suffix = source_file.suffix.lower()
    candidate = dest_dir / f"{source_ref}__{stem}{suffix}"

    counter = 2
    while candidate.exists():
        candidate = dest_dir / f"{source_ref}__{stem}_{counter}{suffix}"
        counter += 1

    return candidate


def organize(dry_run: bool) -> int:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SOURCE_DIR}")

    files = iter_source_files(SOURCE_DIR)
    if not files:
        print(f"No files found in {SOURCE_DIR}")
        return 0

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    for source_file in files:
        destination = destination_for(source_file, DEST_DIR)
        relative_source = source_file.relative_to(ROOT)
        relative_destination = destination.relative_to(ROOT)

        if dry_run:
            print(f"Would move {relative_source} -> {relative_destination}")
            continue

        shutil.move(str(source_file), str(destination))
        moved += 1

    if dry_run:
        print(f"Dry run complete. {len(files)} files would be moved.")
    else:
        print(f"Moved {moved} files to {DEST_DIR.relative_to(ROOT)}")

    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move files from data/drums into data/raw with source folder references."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned moves without changing files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize(dry_run=args.dry_run)
