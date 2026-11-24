"""SrvDB: embedded vector database for local and offline AI applications.

>>> import srvdb
>>> with srvdb.create("./vectors", dimension=384) as db:  # doctest: +SKIP
...     db.add(ids, vectors, metadatas)
>>> db = srvdb.open("./vectors")                            # doctest: +SKIP
>>> srvdb.delete("./vectors")                               # doctest: +SKIP

``srvdb.create`` and ``srvdb.open`` return a database object (``db``).
``srvdb.open`` and ``srvdb.delete`` shadow builtins only inside the
``srvdb`` namespace; call them as ``srvdb.open`` and ``srvdb.delete``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from srvdb import _low_level
from srvdb._low_level import Database, __version__

if TYPE_CHECKING:
    from srvdb._low_level import Index

__all__ = [
    "Database",
    "__version__",
    "create",
    "delete",
    "drop_links",
    "open",
    "rebuild_checksums",
]


def create(
    path: str,
    dimension: int,
    index: Index = "hnsw",
    *,
    exist_ok: bool = False,
    **options: Any,
) -> Database:
    """Create a new database at ``path``; its settings are fixed here.

    The index and options are recorded on disk, so the database is
    reopened later with ``srvdb.open(path)`` alone.

    Args:
        path: Directory for the database files, used as given. It must not
            exist or must be empty; missing parents are created.
        dimension: Vector dimension, 1 to 4096.
        index: "hnsw" (default), "flat", "ivf", "sq8", "pq", "rabitq" or
            "auto".
        exist_ok: If True and a database already exists at ``path`` with
            the same index, dimension and options, open it instead of
            raising (like ``os.makedirs(exist_ok=True)``).
        **options: Index options; each index accepts only its own:
            hnsw: ``m`` (16), ``ef_construction`` (200), ``ef_search`` (50).
            ivf: ``nlist`` (100), ``nprobe`` (10), ``m``,
            ``ef_construction``, ``ef_search``.
            sq8, pq: ``training``, sample vectors the quantizer is trained
            on (otherwise the first batch added).
            rabitq: ``bits`` (1, 2 or 4; default 4), ``rerank``, ``seed``.

    Returns:
        The open database.

    Raises:
        ValueError: Unknown index, an option that does not apply to the
            index (the message lists the valid ones), a value out of
            range, or with ``exist_ok`` an existing database whose settings
            differ.
        FileExistsError: ``path`` already holds a database or other files.
        BlockingIOError: The database is open elsewhere.
        OSError: The files cannot be created.
        RuntimeError: An internal fault occurred.
    """
    return _low_level.create(path, dimension, index, exist_ok, options)


def open(path: str) -> Database:  # noqa: A001
    """Open the existing database at ``path`` with its recorded settings.

    Args:
        path: Directory of a database made by ``srvdb.create``.

    Returns:
        The open database. Use it as a context manager to save and close
        it on exit: ``with srvdb.open(path) as db: ...``.

    Raises:
        FileNotFoundError: No database exists at ``path``.
        ValueError: A quantized database from SrvDB 0.3 without recorded
            settings (see the upgrading guide).
        BlockingIOError: The database is already open elsewhere.
        OSError: The files cannot be read.
        RuntimeError: Stored data is damaged (checksum mismatch) or an
            internal fault occurred.
    """
    return _low_level.open(path)


def delete(path: str) -> None:
    """Delete the database at ``path`` from disk, directory included.

    Only a closed SrvDB database is removed: the directory must hold
    nothing but SrvDB files, at least one of them with a SrvDB header.
    Deleted data cannot be recovered.

    Raises:
        FileNotFoundError: ``path`` is not a directory.
        ValueError: ``path`` holds other files or no SrvDB database;
            nothing was deleted.
        BlockingIOError: The database is open (close it first).
        OSError: A file could not be removed.
    """
    _low_level.delete(path)


def drop_links(path: str) -> int:
    """Delete every link between records of the closed database at ``path``.

    Use it when ``srvdb.open`` raises RuntimeError saying the links were
    written before an older SrvDB (0.3 or earlier, which ignores links)
    added, deleted or compacted records: those links may point at the
    wrong records. Vectors and metadata are not changed; link again
    afterwards. It never runs on its own.

    Returns:
        Number of links removed.

    Raises:
        FileNotFoundError: No database at ``path``.
        BlockingIOError: The database is open (close it first).
        OSError: A file could not be read or written.
    """
    return _low_level.drop_links(path)


def rebuild_checksums(path: str) -> list[str]:
    """Recompute missing or damaged checksum files (``*.crc``) of the
    closed database at ``path`` so it opens again.

    Use it when ``srvdb.open`` fails because a ``.crc`` file is missing or
    damaged, for example after copying only ``vectors.bin``. It never runs
    on its own. A ``.crc`` file that is intact but disagrees with the data
    means the data is damaged: nothing is rewritten and RuntimeError is
    raised. A rebuilt checksum cannot detect damage already in the data;
    if the data may be damaged, restore a backup instead.

    Returns:
        Names of the checksum files that were rewritten (empty if all were
        intact).

    Raises:
        FileNotFoundError: ``path`` is not a directory.
        BlockingIOError: The database is open (close it first).
        OSError: A file could not be read or written.
        RuntimeError: The data is damaged or truncated; restore a backup.
    """
    return _low_level.rebuild_checksums(path)
