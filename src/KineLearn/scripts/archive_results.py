#!/usr/bin/env python3
"""
Archive KineLearn result directories while pruning bulky cached memmaps.
"""

from __future__ import annotations

import argparse
import errno
import os
from dataclasses import dataclass
from pathlib import Path
import shutil


OMIT_SUFFIXES = ("_features.fp32", "_labels.u8")


@dataclass(frozen=True)
class ArchivePlan:
    source: Path
    destination: Path
    moved_files: list[tuple[Path, Path, int]]
    omitted_files: list[tuple[Path, int]]
    directories: list[Path]

    @property
    def moved_bytes(self) -> int:
        return sum(size for _, _, size in self.moved_files)

    @property
    def omitted_bytes(self) -> int:
        return sum(size for _, size in self.omitted_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move a KineLearn results subtree to long-term storage while omitting "
            "cached memmap feature/label arrays."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Source directory to archive, such as results/, "
            "results/<behavior>/<timestamp>/, or another nested results subtree."
        ),
    )
    parser.add_argument(
        "destination",
        help=(
            "Destination directory corresponding to the source root. For example, "
            "'kinelearn-archive-results results /archive/results'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved and omitted without changing any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per moved or omitted file.",
    )
    return parser.parse_args()


def should_omit(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in OMIT_SUFFIXES)


def format_bytes(n_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def ensure_safe_roots(source: Path, destination: Path) -> tuple[Path, Path]:
    source = source.resolve()
    destination = destination.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source}")
    if source == destination:
        raise ValueError("Source and destination must be different directories.")
    if source in destination.parents:
        raise ValueError(
            "Destination cannot be inside the source directory being archived."
        )
    if destination in source.parents:
        raise ValueError(
            "Destination cannot be an ancestor of the source directory being archived."
        )
    return source, destination


def build_archive_plan(source: Path, destination: Path) -> ArchivePlan:
    source, destination = ensure_safe_roots(source, destination)

    moved_files: list[tuple[Path, Path, int]] = []
    omitted_files: list[tuple[Path, int]] = []
    directories = sorted(
        (path for path in source.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        rel_path = path.relative_to(source)
        if should_omit(path):
            omitted_files.append((path, size))
            continue
        moved_files.append((path, destination / rel_path, size))

    validate_destination_plan(destination, moved_files)
    return ArchivePlan(
        source=source,
        destination=destination,
        moved_files=moved_files,
        omitted_files=omitted_files,
        directories=directories,
    )


def validate_destination_plan(
    destination: Path, moved_files: list[tuple[Path, Path, int]]
) -> None:
    collisions: list[Path] = []
    invalid_parents: list[Path] = []

    if destination.exists() and destination.is_file():
        raise FileExistsError(
            f"Destination path exists as a file, not a directory: {destination}"
        )

    for _, dest_path, _ in moved_files:
        if dest_path.exists():
            collisions.append(dest_path)
            continue

        parent = dest_path.parent
        while True:
            if parent.exists():
                if not parent.is_dir():
                    invalid_parents.append(parent)
                break
            if parent == parent.parent:
                break
            parent = parent.parent

    if invalid_parents:
        preview = ", ".join(str(path) for path in invalid_parents[:5])
        raise FileExistsError(
            "Cannot create destination directories because one or more path "
            f"components already exist as files: {preview}"
        )

    if collisions:
        preview = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(
            "Refusing to overwrite existing archive files. "
            f"Collisions: {preview}"
        )


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(src), str(dst))


def remove_empty_directories(root: Path, directories: list[Path]) -> None:
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
    try:
        root.rmdir()
    except OSError:
        pass


def print_plan(plan: ArchivePlan, *, verbose: bool, dry_run: bool) -> None:
    action = "Would move" if dry_run else "Moved"
    omit_action = "Would omit" if dry_run else "Omitted"

    print(f"Source: {plan.source}")
    print(f"Destination: {plan.destination}")
    print(f"{action} files: {len(plan.moved_files)}")
    print(f"{omit_action} memmaps: {len(plan.omitted_files)}")
    print(f"{action} bytes: {plan.moved_bytes} ({format_bytes(plan.moved_bytes)})")
    print(
        f"{omit_action} bytes: {plan.omitted_bytes} "
        f"({format_bytes(plan.omitted_bytes)})"
    )

    if not verbose:
        return

    for src, dst, size in plan.moved_files:
        print(f"MOVE {src} -> {dst} [{format_bytes(size)}]")
    for path, size in plan.omitted_files:
        print(f"OMIT {path} [{format_bytes(size)}]")


def execute_archive(plan: ArchivePlan, *, verbose: bool) -> None:
    plan.destination.mkdir(parents=True, exist_ok=True)

    for src, dst, size in plan.moved_files:
        move_file(src, dst)
        if verbose:
            print(f"MOVE {src} -> {dst} [{format_bytes(size)}]")

    for path, size in plan.omitted_files:
        path.unlink()
        if verbose:
            print(f"OMIT {path} [{format_bytes(size)}]")

    remove_empty_directories(plan.source, plan.directories)


def main() -> None:
    args = parse_args()
    plan = build_archive_plan(Path(args.source), Path(args.destination))

    if args.dry_run:
        print_plan(plan, verbose=args.verbose, dry_run=True)
        return

    execute_archive(plan, verbose=args.verbose)
    print_plan(plan, verbose=False, dry_run=False)


if __name__ == "__main__":
    main()
