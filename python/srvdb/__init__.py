"""SrvDB: embedded vector database for local and offline AI applications.

>>> import srvdb
>>> db = srvdb.Database("./vectors", dimension=384)
"""

import warnings

from srvdb._low_level import Database, __version__

__all__ = ["Database", "__version__"]


def __getattr__(name):
    # Deprecated in 0.3, to be removed in 0.4
    if name == "SrvDBPython":
        warnings.warn(
            "srvdb.SrvDBPython is deprecated; use srvdb.Database instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return Database
    raise AttributeError(f"module 'srvdb' has no attribute {name!r}")
