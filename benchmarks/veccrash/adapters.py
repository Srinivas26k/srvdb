"""VecCrash adapters: each drives one library through its documented
persistence API. ``open_writer(path, dim)`` creates the store; ``add``
adds one batch; ``durable`` is the library's durability point (a batch is
acked only after it returns); ``read_all(path, dim)`` reopens the store
and returns ``{id: vector}``, raising if it does not open. Imports are
local so a process loads only the library under test.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np

Store = dict[int, np.ndarray]


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --- SrvDB ---------------------------------------------------------------
class SrvDB:
    """srvdb.create / add / save; documented durable after save() returns."""

    index = "flat"

    def open_writer(self, path: str, dim: int) -> Any:
        import srvdb

        return srvdb.create(path, dim, self.index)

    def add(self, db: Any, ids: list[int], vecs: np.ndarray) -> None:
        metas = [json.dumps({"k": i}) for i in ids]
        db.add([str(i) for i in ids], vecs.tolist(), metas)

    def durable(self, db: Any) -> None:
        db.save()

    def read_all(self, path: str, dim: int) -> Store:
        # SrvDB 0.4 has no public vector read-back call, so values are
        # checked by exact self-search: each stored vector must return its
        # own id at cosine similarity ~1, and its metadata must match.
        import srvdb

        db = srvdb.open(path)
        try:
            ids = db.get_all_ids()
            if len(ids) != db.count():
                raise RuntimeError("count() disagrees with get_all_ids()")
            out: Store = {}
            for sid in ids:
                meta = db.get(sid)
                ok = meta is not None and json.loads(meta).get("k") == int(sid)
                # Empty array = "present, value checked by self_search";
                # NaN row = metadata does not match the id (corruption).
                out[int(sid)] = (
                    np.empty(0, np.float32) if ok else np.full(dim, np.nan, np.float32)
                )
            return out
        finally:
            db.close()

    def self_search(self, path: str, queries: dict[int, np.ndarray]) -> dict[int, bool]:
        """For each id, True if searching its expected vector returns it at ~1."""
        import srvdb

        db = srvdb.open(path)
        try:
            keys = list(queries)
            res: dict[int, bool] = {}
            for start in range(0, len(keys), 512):
                chunk = keys[start : start + 512]
                hits = db.search_batch([queries[k].tolist() for k in chunk], 1)
                for k, hit in zip(chunk, hits):
                    res[k] = bool(hit) and hit[0][0] == str(k) and hit[0][1] > 0.99999
            return res
        finally:
            db.close()


class SrvDBHnsw(SrvDB):
    """Same as SrvDB, default HNSW index (graph file saved too)."""

    index = "hnsw"


# --- Chroma --------------------------------------------------------------
class Chroma:
    """chromadb.PersistentClient; data is persisted as add() returns."""

    def open_writer(self, path: str, dim: int) -> Any:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        return client.create_collection("veccrash")

    def add(self, col: Any, ids: list[int], vecs: np.ndarray) -> None:
        col.add(ids=[str(i) for i in ids], embeddings=vecs)

    def durable(self, col: Any) -> None:
        return None  # add() is the durability point; no separate call

    def read_all(self, path: str, dim: int) -> Store:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        col = client.get_collection("veccrash")
        got = col.get(include=["embeddings"])
        embs = got["embeddings"]
        return {int(i): np.asarray(e, np.float32) for i, e in zip(got["ids"], embs)}


# --- LanceDB -------------------------------------------------------------
class LanceDB:
    """lancedb.connect + table.add; each add commits a new table version."""

    def open_writer(self, path: str, dim: int) -> Any:
        import lancedb

        return {"db": lancedb.connect(path), "table": None}

    def add(self, h: Any, ids: list[int], vecs: np.ndarray) -> None:
        rows = [{"id": i, "vector": v} for i, v in zip(ids, vecs)]
        if h["table"] is None:
            h["table"] = h["db"].create_table("veccrash", rows)
        else:
            h["table"].add(rows)

    def durable(self, h: Any) -> None:
        return None  # add() commits a version before returning

    def read_all(self, path: str, dim: int) -> Store:
        import lancedb

        tbl = lancedb.connect(path).open_table("veccrash").to_arrow()
        ids = tbl.column("id").to_pylist()
        vecs = np.asarray(tbl.column("vector").to_pylist(), np.float32)
        return {int(i): v for i, v in zip(ids, vecs)}


# --- FAISS ---------------------------------------------------------------
class Faiss:
    """IndexIDMap(IndexFlatL2) + faiss.write_index(index, path)."""

    file = "index.faiss"

    def open_writer(self, path: str, dim: int) -> Any:
        import faiss

        os.makedirs(path, exist_ok=True)
        return {"index": faiss.IndexIDMap(faiss.IndexFlatL2(dim)), "path": path}

    def add(self, h: Any, ids: list[int], vecs: np.ndarray) -> None:
        h["index"].add_with_ids(vecs, np.asarray(ids, np.int64))

    def durable(self, h: Any) -> None:
        import faiss

        faiss.write_index(h["index"], os.path.join(h["path"], self.file))

    def read_all(self, path: str, dim: int) -> Store:
        import faiss

        idx = faiss.read_index(os.path.join(path, self.file))
        ids = faiss.vector_to_array(idx.id_map)
        vecs = idx.index.reconstruct_n(0, idx.ntotal)
        return {int(i): v for i, v in zip(ids, vecs)}


class FaissAtomic(Faiss):
    """FAISS with the application-level temp + fsync + rename pattern."""

    def durable(self, h: Any) -> None:
        import faiss

        final = os.path.join(h["path"], self.file)
        tmp = final + ".tmp"
        faiss.write_index(h["index"], tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, final)
        _fsync_dir(h["path"])


# --- hnswlib -------------------------------------------------------------
class Hnswlib:
    """hnswlib.Index.add_items + save_index(path)."""

    def open_writer(self, path: str, dim: int) -> Any:
        import hnswlib

        os.makedirs(path, exist_ok=True)
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=200_000, ef_construction=100, M=16)
        return {"index": idx, "file": os.path.join(path, "index.bin")}

    def add(self, h: Any, ids: list[int], vecs: np.ndarray) -> None:
        h["index"].add_items(vecs, np.asarray(ids, np.int64))

    def durable(self, h: Any) -> None:
        h["index"].save_index(h["file"])

    def read_all(self, path: str, dim: int) -> Store:
        import hnswlib

        idx = hnswlib.Index(space="l2", dim=dim)
        idx.load_index(os.path.join(path, "index.bin"), max_elements=200_000)
        ids = idx.get_ids_list()
        vecs = np.asarray(idx.get_items(ids), np.float32) if ids else []
        return {int(i): v for i, v in zip(ids, vecs)}


# --- usearch -------------------------------------------------------------
class USearch:
    """usearch.index.Index.add + Index.save(path)."""

    def open_writer(self, path: str, dim: int) -> Any:
        from usearch.index import Index

        os.makedirs(path, exist_ok=True)
        idx = Index(ndim=dim, metric="l2sq", dtype="f32")
        return {"index": idx, "file": os.path.join(path, "index.usearch")}

    def add(self, h: Any, ids: list[int], vecs: np.ndarray) -> None:
        h["index"].add(np.asarray(ids, np.uint64), vecs)

    def durable(self, h: Any) -> None:
        h["index"].save(h["file"])

    def read_all(self, path: str, dim: int) -> Store:
        from usearch.index import Index

        idx = Index.restore(os.path.join(path, "index.usearch"))
        if idx is None:
            raise RuntimeError("Index.restore returned None")
        keys = np.asarray(idx.keys, np.uint64)
        vecs = idx.get(keys, dtype=np.float32) if len(keys) else []
        return {int(k): np.asarray(v, np.float32) for k, v in zip(keys, vecs)}


# --- sqlite-vec ----------------------------------------------------------
class SqliteVec:
    """SQLite (WAL) + sqlite-vec vec0 table; durable after commit()."""

    def _connect(self, path: str) -> Any:
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(os.path.join(path, "vec.db"))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def open_writer(self, path: str, dim: int) -> Any:
        os.makedirs(path, exist_ok=True)
        conn = self._connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")
        conn.commit()
        return conn

    def add(self, conn: Any, ids: list[int], vecs: np.ndarray) -> None:
        rows = [(i, v.tobytes()) for i, v in zip(ids, vecs)]
        conn.executemany("INSERT INTO v(rowid, embedding) VALUES (?, ?)", rows)

    def durable(self, conn: Any) -> None:
        conn.commit()

    def read_all(self, path: str, dim: int) -> Store:
        conn = self._connect(path)
        try:
            cur = conn.execute("SELECT rowid, embedding FROM v")
            return {int(r): np.frombuffer(b, np.float32).copy() for r, b in cur}
        finally:
            conn.close()


ADAPTERS: dict[str, Callable[[], Any]] = {
    "srvdb": SrvDB,
    "srvdb-hnsw": SrvDBHnsw,
    "chroma": Chroma,
    "lancedb": LanceDB,
    "faiss": Faiss,
    "faiss-atomic": FaissAtomic,
    "hnswlib": Hnswlib,
    "usearch": USearch,
    "sqlite-vec": SqliteVec,
}
