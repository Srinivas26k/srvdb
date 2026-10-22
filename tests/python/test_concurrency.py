"""Concurrency contract (ADR 0005) through the Python API."""

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
