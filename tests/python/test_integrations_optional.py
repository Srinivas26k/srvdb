"""The framework adapters stay optional: no import cost, clear errors."""

import subprocess
import sys

import pytest


def run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_importing_srvdb_does_not_import_frameworks():
    res = run(
        "import sys, srvdb, srvdb.integrations; "
        "assert 'langchain_core' not in sys.modules; "
        "assert 'llama_index' not in sys.modules"
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    ("module", "blocked", "extra"),
    [
        ("langchain", "langchain_core", "srvdb[langchain]"),
        ("llamaindex", "llama_index", "srvdb[llamaindex]"),
    ],
)
def test_missing_extra_raises_import_error_naming_the_extra(module, blocked, extra):
    # Setting a module to None in sys.modules makes importing it fail.
    res = run(
        f"import sys; sys.modules[{blocked!r}] = None; "
        f"import srvdb.integrations.{module}"
    )
    assert res.returncode != 0
    assert "ImportError" in res.stderr
    assert extra in res.stderr
