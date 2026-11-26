"""Code run in fresh processes by measure.py: import only, and create an
empty store plus add 1,000 seeded vectors of dimension 128. Each snippet
prints its peak RSS (ru_maxrss, KiB on Linux) as PEAK_RSS_KIB=<n>."""

from __future__ import annotations

TAIL = "\nimport resource\nprint('PEAK_RSS_KIB=%d' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"  # noqa: E501
DATA = (
    "import numpy as np\n"
    "x = np.random.default_rng(42).standard_normal((1000, 128), dtype=np.float32)\n"
    "ids = list(range(1, 1001))\n"
)

IMPORTS = {
    "srvdb": "import srvdb" + TAIL,
    "chroma": "import chromadb" + TAIL,
    "lancedb": "import lancedb" + TAIL,
    "faiss": "import faiss" + TAIL,
    "hnswlib": "import hnswlib" + TAIL,
    "usearch": "from usearch.index import Index" + TAIL,
    "sqlite-vec": "import sqlite3, sqlite_vec" + TAIL,
}

WORKLOADS = {
    "srvdb": DATA
    + (
        "import srvdb\n"
        "db = srvdb.create('db', 128)\n"
        "db.add([str(i) for i in ids], x.tolist(), ['{}'] * 1000)\n"
        "db.save()\n"
    ),
    "chroma": DATA
    + (
        "import chromadb\n"
        "c = chromadb.PersistentClient(path='db').create_collection('veccrash')\n"
        "c.add(ids=[str(i) for i in ids], embeddings=x)\n"
    ),
    "lancedb": DATA
    + (
        "import lancedb\n"
        "lancedb.connect('db').create_table('t', [{'id': i, 'vector': v} for i, v in zip(ids, x)])\n"  # noqa: E501
    ),
    "faiss": DATA
    + (
        "import faiss\n"
        "ix = faiss.IndexIDMap(faiss.IndexHNSWFlat(128, 16))\n"
        "ix.add_with_ids(x, np.asarray(ids, np.int64))\n"
        "faiss.write_index(ix, 'db.faiss')\n"
    ),
    "hnswlib": DATA
    + (
        "import hnswlib\n"
        "ix = hnswlib.Index(space='l2', dim=128)\n"
        "ix.init_index(max_elements=1000, M=16)\n"
        "ix.add_items(x, ids)\n"
        "ix.save_index('db.bin')\n"
    ),
    "usearch": DATA
    + (
        "from usearch.index import Index\n"
        "ix = Index(ndim=128, metric='l2sq')\n"
        "ix.add(np.asarray(ids, np.uint64), x)\n"
        "ix.save('db.usearch')\n"
    ),
    "sqlite-vec": DATA
    + (
        "import sqlite3, sqlite_vec\n"
        "c = sqlite3.connect('db.sqlite')\n"
        "c.enable_load_extension(True); sqlite_vec.load(c)\n"
        "c.execute('CREATE VIRTUAL TABLE v USING vec0(embedding float[128])')\n"
        "c.executemany('INSERT INTO v(rowid, embedding) VALUES (?, ?)', "
        "[(i, r.tobytes()) for i, r in zip(ids, x)])\n"
        "c.commit()\n"
    ),
}
for _k in WORKLOADS:
    WORKLOADS[_k] += TAIL
