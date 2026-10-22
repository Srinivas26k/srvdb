"""Concurrency contract (ADR 0005) through the Python API."""

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import srvdb

DIM = 128


def vectors(n, seed=3):
    rng = random.Random(seed)
    return [[rng.random() - 0.5 for _ in range(DIM)] for _ in range(n)]


def test_second_open_of_the_same_path_is_refused(tmp_path):
    first = srvdb.Database(str(tmp_path), dimension=DIM)
    with pytest.raises(BlockingIOError, match="already open"):
        srvdb.Database(str(tmp_path), dimension=DIM)
    with pytest.raises(OSError):  # BlockingIOError is an OSError
        srvdb.Database(str(tmp_path), dimension=DIM)
    del first
    assert srvdb.Database(str(tmp_path), dimension=DIM).count() == 0


def test_searches_from_many_threads(tmp_path):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    vs = vectors(300)
    db.add([f"d{i}" for i in range(300)], vs, ["{}"] * 300)

    def top(i):
        return db.search(vs[i], 1)[0][0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(top, range(300)))
    assert found == [f"d{i}" for i in range(300)]


def test_add_releases_the_gil(tmp_path):
    db = srvdb.Database.new_hnsw(str(tmp_path), dimension=DIM)
    vs = vectors(1500)
    ids = [f"d{i}" for i in range(1500)]
    span = {}

    def insert():
        span["start"] = time.perf_counter()
        db.add(ids, vs, ["{}"] * 1500)
        span["end"] = time.perf_counter()

    writer = threading.Thread(target=insert)
    ticks = []
    writer.start()
    while writer.is_alive():
        ticks.append(time.perf_counter())  # runs only while the GIL is free
    writer.join()

    assert db.count() == 1500
    insert_time = span["end"] - span["start"]
    during = [t for t in ticks if span["start"] <= t <= span["end"]]
    covered = during[-1] - during[0] if len(during) > 1 else 0.0
    # Holding the GIL would freeze this thread for the whole insert
    assert covered > insert_time / 2, (
        f"main thread was blocked: it ran for {covered:.3f}s "
        f"of a {insert_time:.3f}s insert"
    )


def test_mixed_reads_and_writes_from_many_threads(tmp_path):
    """Writers and readers share one Database without errors or deadlock."""
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    base = vectors(200, seed=11)
    db.add([f"base{i}" for i in range(200)], base, ["{}"] * 200)
    errors = []

    def writer(w):
        try:
            mine = vectors(60, seed=100 + w)
            for i in range(60):
                db.add([f"w{w}-{i}"], [mine[i]], ['{"v": 1}'])
            for i in range(0, 60, 2):
                db.upsert([f"w{w}-{i}"], [mine[i]], ['{"v": 2}'])
            db.delete([f"w{w}-{i}" for i in range(1, 60, 3)])
        except Exception as e:  # collected and asserted below
            errors.append(repr(e))

    def reader():
        try:
            for i in range(150):
                assert db.search(base[i], 1)[0][0] == f"base{i}"
                db.count()
                db.get(f"base{i}")
        except Exception as e:  # collected and asserted below
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "threads deadlocked"
    assert errors == []

    deleted_per_writer = len(range(1, 60, 3))
    assert db.count() == 200 + 4 * (60 - deleted_per_writer)
    assert json.loads(db.get("w0-0"))["v"] == 2
    assert "w0-1" not in db.get_all_ids()
