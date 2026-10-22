"""Type stubs for the compiled `srvdb._low_level` extension module.

Keep in sync with crates/srvdb-py/src.

Errors: invalid arguments (wrong dimension, NaN or infinity, k < 1, empty or
duplicate ids, metadata that is not a JSON object) and calls that do not fit
the current mode raise ValueError; operating-system I/O failures raise
OSError; anything unexpected raises RuntimeError.
"""

from collections.abc import Sequence
from typing import Literal

__version__: str

Mode = Literal["flat", "hnsw", "sq8", "scalar", "pq", "product", "ivf", "auto"]
SwitchableMode = Literal["flat", "hnsw", "sq8", "scalar", "ivf", "auto"]

class Database:
    def __init__(
        self, path: str, dimension: int = 1536, mode: Mode = "flat"
    ) -> None: ...
    @staticmethod
    def new_hnsw(
        path: str,
        dimension: int = 1536,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> Database: ...
    @staticmethod
    def new_scalar_quantized(
        path: str, dimension: int, training_vectors: Sequence[Sequence[float]]
    ) -> Database: ...
    @staticmethod
    def new_product_quantized(
        path: str, dimension: int, training_vectors: Sequence[Sequence[float]]
    ) -> Database: ...
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[str],
    ) -> int: ...
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[str],
    ) -> int: ...
    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]: ...
    def search_batch(
        self, queries: Sequence[Sequence[float]], k: int
    ) -> list[list[tuple[str, float]]]: ...
    def get(self, id: str) -> str | None: ...
    def count(self) -> int:
        """Number of searchable vectors; deleted ones are not counted."""
    def persist(self) -> None: ...
    def delete(self, ids: Sequence[str]) -> int: ...
    def get_all_ids(self) -> list[str]: ...
    def set_mode(self, mode: SwitchableMode) -> None: ...
    def configure_ivf(self, nlist: int = 100, nprobe: int = 10) -> None: ...
    def train_ivf(self) -> None: ...
    def info(self) -> str: ...
