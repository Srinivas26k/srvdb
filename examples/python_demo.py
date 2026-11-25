"""SrvDB Python example: create, add, search, save, reopen and delete.

Run after ``maturin develop`` (or ``pip install srvdb``)::

    python examples/python_demo.py
"""

import json
import random

import srvdb

DIM = 384


def vec():
    return [random.random() - 0.5 for _ in range(DIM)]


def main():
    print(f"SrvDB {srvdb.__version__}")

    # A new database; its settings are fixed here and recorded on disk.
    # exist_ok=True reopens it if an earlier run left one with the same settings.
    with srvdb.create("./python_demo_db", DIM, "hnsw", exist_ok=True, m=16) as db:
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [vec() for _ in ids]
        metadatas = [json.dumps({"title": f"Document {i}"}) for i in range(1, 4)]
        db.upsert(ids, embeddings, metadatas)
        print(f"{db.count()} vectors in {db}")
        for doc_id, score in db.search(embeddings[0], k=3):
            print(f"  {doc_id:6s} {score:.4f} {db.get(doc_id)}")
        db.delete(["doc2"])
    # The with block saved and closed the database.

    db = srvdb.open("./python_demo_db")  # settings are read from disk
    print(f"Reopened: {db.count()} vectors, ids {sorted(db.get_all_ids())}")
    db.close()

    # SQ8 stores one byte per value; train it on a sample you choose.
    sample = [vec() for _ in range(200)]
    with srvdb.create("./python_sq8_db", DIM, "sq8", exist_ok=True, training=sample) as sq:
        sq.upsert(["sq1"], [sample[0]], ['{"info": "quantized"}'])
        print(f"SQ8 database: {sq}")

    srvdb.delete("./python_demo_db")
    srvdb.delete("./python_sq8_db")
    print("Deleted both databases.")


if __name__ == "__main__":
    main()
