"""Run every example in the SrvDB gallery, one interpreter each.

Run (from the repository root, with the gallery requirements installed)::

    python examples/gallery/run_all.py            # full sizes
    python examples/gallery/run_all.py --fast     # small sizes, for CI
    python examples/gallery/run_all.py --only 02 08

Charts go to ``examples/gallery/output/`` (gitignored) unless ``--out``
is given. Exits non-zero if any example fails, printing its output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

GALLERY = Path(__file__).resolve().parent


def examples() -> list[Path]:
    """The script of every ``NN_topic`` directory, in order."""
    found = []
    for folder in sorted(GALLERY.glob("[0-9][0-9]_*")):
        scripts = [p for p in folder.glob("*.py") if not p.name.startswith("_")]
        if len(scripts) != 1:
            raise SystemExit(
                f"{folder.name}: expected one script, found {len(scripts)}"
            )
        found.append(scripts[0])
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="small sizes")
    parser.add_argument("--out", type=Path, default=GALLERY / "output")
    parser.add_argument(
        "--only", nargs="*", default=None, help="example numbers to run, e.g. 02 08"
    )
    args = parser.parse_args()
    failures = []
    for script in examples():
        number = script.parent.name[:2]
        if args.only and number not in args.only:
            continue
        cmd = [sys.executable, str(script), "--out", str(args.out)]
        if args.fast:
            cmd.append("--fast")
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        seconds = time.perf_counter() - start
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"{script.parent.name:<22} {status:<18} {seconds:7.1f} s", flush=True)
        if proc.returncode != 0:
            failures.append(script.parent.name)
            print(proc.stdout[-2000:] + proc.stderr[-4000:])
    summary = "failed: " + ", ".join(failures) if failures else "all examples passed"
    print(f"\n{summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
