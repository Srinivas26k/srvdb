"""Smoke test: every gallery example runs in fast mode.

Skipped when the gallery requirements (examples/gallery/requirements.txt)
are not installed, so the core test suite needs none of them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

GALLERY = Path(__file__).resolve().parents[2] / "examples" / "gallery"

for module in ("sklearn", "matplotlib", "numpy", "pandas"):
    pytest.importorskip(module, reason="install examples/gallery/requirements.txt")


def test_gallery_runs_in_fast_mode(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(GALLERY / "run_all.py"), "--fast", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    charts = sorted(p.name for p in tmp_path.glob("*.png"))
    # Every example except 01 (no chart) saves exactly one chart.
    assert len(charts) == 11, charts
