"""Type stubs for the compiled `srvdb._low_level` extension module.

Keep in sync with crates/srvdb-py/src.

Errors: invalid arguments (wrong dimension, NaN or infinity, k < 1, empty or
duplicate ids, metadata that is not a JSON object) and calls that do not fit
the current mode raise ValueError; operating-system I/O failures raise
OSError, and opening a database that is already open elsewhere raises
BlockingIOError (an OSError); anything unexpected raises RuntimeError.

Threads: a Database is safe to share between threads. Every method releases
the GIL; searches run in parallel, and writes (add, upsert, delete,
persist, compact) wait for each other and for running searches instead of
failing.
"""

from collections.abc import Sequence
from typing import Literal

__version__: str

Mode = Literal["flat", "hnsw", "sq8", "scalar", "pq", "product", "ivf", "auto"]
SwitchableMode = Literal["flat", "hnsw", "sq8", "scalar", "ivf", "auto"]

class Database:
    """An open SrvDB database; safe to share between threads.

    Writes are visible to search when the call returns and durable once
    persist() returns. The file lock is released when the object is
    garbage-collected.
    """

    def __init__(
        self, path: str, dimension: int = 1536, mode: Mode = "flat"
    ) -> None:
        """Open the database at ``path``, creating it if needed.

        ``dimension`` is 128 to 4096 and must match a stored database.
        ``mode`` is not stored; pass it on every open.

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
    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Up to ``k`` ``(id, cosine score)`` pairs, best first."""
    def search_batch(
        self, queries: Sequence[Sequence[float]], k: int
    ) -> list[list[tuple[str, float]]]:
        """One search() result list per query, in query order."""
    def get(self, id: str) -> str | None:
        """Stored metadata JSON (with ``"__id__"``); ValueError if unknown."""
    def count(self) -> int:
        """Number of searchable vectors; deleted ones are not counted."""
    def persist(self) -> None:
        """Make every earlier write durable (crash-safe commit)."""
    def delete(self, ids: Sequence[str]) -> int:
        """Delete by id, ignoring unknown ids. Returns the count deleted."""
    def get_all_ids(self) -> list[str]:
        """Every live id, in no particular order."""
    def compact(self) -> int:
        """Remove deleted vectors from disk; string ids are unchanged."""
    def set_mode(self, mode: SwitchableMode) -> None:
        """Switch the index mode (not stored across opens)."""
    def set_ef_search(self, ef_search: int) -> None:
        """Set HNSW's search candidate list size (at least 1)."""
    def configure_ivf(self, nlist: int = 100, nprobe: int = 10) -> None:
        """Set IVF partitions and partitions searched; then train_ivf()."""
    def train_ivf(self) -> None:
        """Train IVF on the stored vectors; ValueError unless in "ivf" mode."""
    def info(self) -> str:
        """One-line summary: dimension, mode and vector count."""
