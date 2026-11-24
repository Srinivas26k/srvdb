"""Type stubs for the compiled `srvdb._low_level` extension module.

Keep in sync with crates/srvdb-py/src.

Errors: invalid arguments (wrong dimension, NaN or infinity, k < 1, negative
counts, empty or duplicate ids, metadata that is not a JSON object) and
calls that do not fit the current mode raise ValueError; operating-system I/O
failures raise OSError, and opening a database that is already open elsewhere raises
BlockingIOError (an OSError); anything unexpected raises RuntimeError.

Threads: a Database is safe to share between threads. Every method releases
the GIL; searches run in parallel, and writes (add, upsert, delete,
persist, compact) wait for each other and for running searches instead of
failing.
"""

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal

__version__: str

Mode = Literal[
    "flat",
    "hnsw",
    "sq8",
    "scalar",
    "pq",
    "product",
    "ivf",
    "auto",
    "rabitq1",
    "rabitq2",
    "rabitq4",
]
Index = Literal["flat", "hnsw", "ivf", "sq8", "pq", "rabitq", "auto"]
SwitchableMode = Literal["flat", "hnsw", "sq8", "scalar", "ivf", "auto"]

def create(
    path: str,
    dimension: int,
    index: Index = "hnsw",
    exist_ok: bool = False,
    options: Mapping[str, Any] | None = None,
) -> Database:
    """Create a new database (see ``srvdb.create``). Raises FileExistsError
    if ``path`` is not new or empty, ValueError for bad options or, with
    ``exist_ok``, an existing database with other settings."""

def open(path: str) -> Database:  # noqa: A001
    """Open an existing database with its recorded settings (see
    ``srvdb.open``). Raises FileNotFoundError if there is none."""

def delete(path: str) -> None:
    """Delete a closed SrvDB database directory (see ``srvdb.delete``).
    Raises ValueError for a directory with other files, BlockingIOError
    while the database is open."""

def rebuild_checksums(path: str) -> list[str]:
    """Recompute missing or damaged checksum files of a closed database
    (see ``srvdb.rebuild_checksums``). Returns the rewritten file names.
    Raises RuntimeError if the data itself is damaged."""

class Database:
    """An open SrvDB database; safe to share between threads.

    Writes are visible to search when the call returns and durable once
    persist() returns. The file lock is released when the object is
    garbage-collected.
    """

    def __init__(self, path: str, dimension: int = 1536, mode: Mode = "flat") -> None:
        """Open the database at ``path``, creating it if needed.

        ``dimension`` is 128 to 4096 and must match a stored database.
        ``mode`` is recorded on disk: the default "flat" reopens in the
        recorded mode, another mode raises ``ValueError``.

        Raises:
            ValueError: bad dimension, unknown mode or dimension mismatch.
            BlockingIOError: the database is already open elsewhere.
            OSError: the files cannot be created or read.
            RuntimeError: stored data is damaged or an internal fault occurred.
        """
    @staticmethod
    def new_hnsw(
        path: str,
        dimension: int = 1536,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> Database:
        """Open or create an HNSW database with explicit graph parameters."""
    @staticmethod
    def new_scalar_quantized(
        path: str, dimension: int, training_vectors: Sequence[Sequence[float]]
    ) -> Database:
        """Open or create an SQ8 database trained on ``training_vectors``."""
    @staticmethod
    def new_product_quantized(
        path: str, dimension: int, training_vectors: Sequence[Sequence[float]]
    ) -> Database:
        """Open or create a PQ database trained on ``training_vectors``."""
    @staticmethod
    def new_rabitq(
        path: str,
        dimension: int = 1536,
        bits: Literal[1, 2, 4] = 4,
        rerank: int | None = None,
        seed: int | None = None,
    ) -> Database:
        """Full vectors on disk, searched through 1/2/4-bit RaBitQ codes plus
        exact rerank of k * rerank candidates (default 40/20/10 for 1/2/4
        bits; 0 disables rerank). No training. Raises ValueError for other
        bit widths."""
    def set_rabitq_rerank(self, rerank: int) -> None:
        """Rerank factor for later searches; ValueError outside RaBitQ mode."""
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[str],
    ) -> int:
        """Add new vectors; ValueError if an id exists. Returns the count added."""
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[str],
    ) -> int:
        """Add vectors, replacing existing ids. Returns the count written."""
    def search(
        self,
        query: Sequence[float],
        k: int,
        *,
        where: Mapping[str, Any] | None = None,
        expand: tuple[str, int] | tuple[str, int, int] | None = None,
    ) -> list[tuple[str, float]] | list[tuple[str, float, int]]:
        """Up to ``k`` ``(id, cosine score)`` pairs, best first.

        ``expand=(link_type, hops[, max_results])`` follows outgoing links
        of ``link_type`` from the hits for 1 to 3 hops (at most
        ``max_results`` added, default 100, at most 1000) and returns
        ``(id, score, hop)`` tuples instead; hits have hop 0. Linked
        records inherit the score of the record they were reached from.
        Cannot be combined with ``where``.

        ``where`` keeps only vectors whose metadata matches, e.g.
        ``{"genre": "jazz", "year": {"$gte": 1950}}``. Operators: ``$eq``,
        ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``,
        ``$and``, ``$or``, ``$not``; ``"a.b"`` names a nested field.
        ValueError for an unknown operator or a wrong operand type.
        """
    def search_batch(
        self,
        queries: Sequence[Sequence[float]],
        k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """One search() result list per query, in query order; ``where``
        is evaluated once for the batch."""
    def link(self, source: str, link_type: str, target: str) -> bool:
        """Add the link ``source -link_type-> target`` (durable after
        save()). False if it existed. ``link_type`` matches
        ``[a-z][a-z0-9_]{0,63}``; at most 1024 outgoing links per record."""
    def unlink(self, source: str, link_type: str, target: str) -> bool:
        """Remove a link; False if it did not exist."""
    def neighbors(self, id: str, type: str | None = None) -> list[tuple[str, str]]:
        """Outgoing ``(link_type, target id)`` links of ``id``, sorted.
        Deleting a record removes its links; upsert keeps them."""
    def get(self, id: str) -> str | None:
        """Stored metadata JSON (with ``"__id__"``), or None for an unknown id."""
    def count(self) -> int:
        """Number of searchable vectors; deleted ones are not counted."""
    def save(self) -> None:
        """Make every earlier write durable (crash-safe commit)."""
    def persist(self) -> None:
        """Alias of save()."""
    def close(self) -> None:
        """Release the database and its lock without saving; idempotent.
        Other methods then raise ValueError."""
    def __enter__(self) -> Database: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> bool:
        """Save and close on a clean exit; close without saving on an
        exception."""
    def delete(self, ids: Sequence[str]) -> int:
        """Delete by id, ignoring unknown ids. Returns the count deleted."""
    def get_all_ids(self) -> list[str]:
        """Every live id, in no particular order."""
    def compact(self) -> int:
        """Remove deleted vectors from disk; string ids are unchanged."""
    def set_mode(self, mode: SwitchableMode) -> None:
        """Switch the index mode; recorded by the next ``persist()``."""
    def set_ef_search(self, ef_search: int) -> None:
        """Set HNSW's search candidate list size (at least 1)."""
    def configure_ivf(self, nlist: int = 100, nprobe: int = 10) -> None:
        """Set IVF partitions and partitions searched; then train_ivf()."""
    def train_ivf(self) -> None:
        """Train IVF on the stored vectors; ValueError unless in "ivf" mode."""
    def info(self) -> str:
        """One-line summary: dimension, mode and vector count."""
