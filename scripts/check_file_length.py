#!/usr/bin/env python3
"""Fail if any source file exceeds the project's line limit.

Usage: python scripts/check_file_length.py [--max N] [PATH ...]
With no paths, checks the tracked source trees listed in ROOTS.
"""

import argparse
import pathlib
import sys

ROOTS = ["crates", "python", "tests/python", "scripts"]
EXTENSIONS = {".rs", ".py", ".pyi"}
DEFAULT_MAX = 300


def source_files(paths):
    for raw in paths:
        path = pathlib.Path(raw)
        if path.is_file():
            if path.suffix in EXTENSIONS:
                yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix in EXTENSIONS and "target" not in child.parts:
                    yield child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX)
    parser.add_argument("paths", nargs="*", default=ROOTS)
    args = parser.parse_args()

    too_long = []
    for path in source_files(args.paths):
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        if lines > args.max:
            too_long.append((path, lines))

    for path, lines in too_long:
        print(f"{path}: {lines} lines (limit {args.max})")
    return 1 if too_long else 0


if __name__ == "__main__":
    sys.exit(main())
