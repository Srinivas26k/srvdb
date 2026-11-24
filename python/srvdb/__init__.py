"""SrvDB: embedded vector database for local and offline AI applications.

>>> import srvdb
>>> db = srvdb.open("./vectors", dimension=384)

``srvdb.open`` returns a :class:`srvdb.Database`. It shadows the builtin
``open`` only inside the ``srvdb`` namespace; call it as ``srvdb.open``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srvdb._low_level import Database, __version__

if TYPE_CHECKING:
    from srvdb._low_level import Mode

__all__ = ["Database", "__version__", "open"]


def open(path: str, dimension: int = 1536, mode: Mode = "flat") -> Database:  # noqa: A001
    """Open the database at ``path``, creating it if it does not exist.

    Same as ``Database(path, dimension=dimension, mode=mode)``. For HNSW or
    quantized databases with custom parameters use ``Database.new_hnsw``,
    ``Database.new_scalar_quantized`` or ``Database.new_product_quantized``.

    Args:
        path: Directory holding the database files. Used as given.
        dimension: Vector dimension; must match an existing database.
        mode: Index mode: "flat", "hnsw", "sq8"/"scalar", "pq"/"product",
            "ivf" or "auto".

    Returns:
        The open ``Database``.

    Raises:
        ValueError: Invalid arguments (unknown mode, bad dimension, or a
            dimension that does not match the stored database).
        BlockingIOError: The database is already open elsewhere.
        OSError: The files cannot be created or read.
        RuntimeError: Stored data is damaged or an internal fault occurred.
    """
    return Database(path, dimension=dimension, mode=mode)
