"""SrvDB: embedded vector database for local and offline AI applications.

>>> import srvdb
>>> db = srvdb.Database("./vectors", dimension=384)
"""

from srvdb._low_level import Database, __version__

__all__ = ["Database", "__version__"]
