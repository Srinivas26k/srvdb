"""Run every ```python example in the documentation pages.

Usage (from the repository root, after ``maturin develop``)::

    python website/scripts/check_examples.py [PAGE_DIR ...]

Without arguments it checks the Python API reference, the Concepts pages,
Get started, the Guides and the Examples gallery (``.md`` and ``.mdx``
pages). Mark a block that cannot run offline with
``<!-- check_examples: skip (reason) -->`` on the line before it (in
``.mdx`` pages, which reject HTML comments, use
``{/* check_examples: skip (reason) */}``).

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

DOCS = Path(__file__).resolve().parent.parent / "src/content/docs"
DEFAULT_DIRS = [
    DOCS / "reference/python",
    DOCS / "concepts",
    DOCS / "start",
    DOCS / "guides",
    DOCS / "examples",
]
# A block directly preceded by ``<!-- check_examples: skip (reason) -->`` is
# not run: it needs software or data the checker cannot assume (a model
# download, a local LLM server). Say why in the marker. MDX pages use the
# ``{/* ... */}`` comment form instead.
BLOCK = re.compile(
    r"^((?:<!--|\{/\*) check_examples: skip[^\n]*(?:-->|\*/\})\n)?"
    r"```python\n(.*?)^```",
    re.S | re.M,
)


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
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS
    pages = sorted(p for d in dirs for p in (*d.glob("*.md"), *d.glob("*.mdx")))
    failures = total = 0
    for page in pages:
        blocks = [
            code
            for skip, code in BLOCK.findall(page.read_text())
            if not skip and runnable(code)
        ]
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
