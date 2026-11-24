"""Run every ```python example in the Python API reference pages.

Usage (from the repository root, after ``maturin develop``)::

    python website/scripts/check_examples.py [PAGE_DIR]

Each example runs in a fresh temporary directory with a separate
interpreter. When an example contains comment lines starting with ``# ``
right after a ``print(...)`` line, those comments are the expected output and
are compared with what the example printed. Exits non-zero if any example
fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/content/docs/reference/python"
)
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def expected_output(code: str) -> list[str] | None:
    """Comment lines that follow print() calls, or None if there are none."""
    lines, expected, after_print = code.splitlines(), [], False
    for line in lines:
        if line.startswith("print("):
            after_print = True
        elif after_print and line.startswith("# "):
            expected.append(line[2:])
            continue
        else:
            after_print = False
    return expected or None


def runnable(code: str) -> bool:
    """Signature blocks are not examples; examples import something."""
    return "import " in code


def run(page: Path, index: int, code: str) -> str | None:
    """Run one example; return an error description or None."""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if proc.returncode != 0:
        return f"exit {proc.returncode}\n{proc.stderr.strip()}"
    want = expected_output(code)
    got = proc.stdout.splitlines()
    if want is not None and got != want:
        return f"output mismatch\n  expected: {want}\n  got:      {got}"
    return None


def main() -> int:
    page_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    failures = total = 0
    for page in sorted(page_dir.glob("*.md")):
        blocks = [b for b in BLOCK.findall(page.read_text()) if runnable(b)]
        for i, code in enumerate(blocks, 1):
            total += 1
            error = run(page, i, code)
            if error is not None:
                failures += 1
                print(f"FAIL {page.name} example {i}: {error}")
    print(f"{total - failures}/{total} examples passed")
    return 1 if failures or total == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
