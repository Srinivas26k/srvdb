"""Memory budget (ADR 0018): create(index="auto", memory_mb=...), report,
step-down, refusal and reopen."""

import random

import pytest

import srvdb

DIM = 32


def vectors(n, seed):
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]


def test_budget_reports_steps_down_refuses_and_reopens(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM, index="auto", memory_mb=1)
    report = db.memory_report()
    assert report["mode"] == "flat"
    assert report["exact"] is True
    assert report["budget_bytes"] == 1 << 20
    flat_capacity = report["capacity"]

    data = vectors(flat_capacity + 1, 1)
    db.add([f"v{i}" for i in range(len(data))], data, ["{}"] * len(data))
    report = db.memory_report()
    assert (report["mode"], report["bits"]) == ("rabitq", 4)
    assert report["total_bytes"] <= report["limit_bytes"]
    assert db.search(data[5], k=1)[0][0] == "v5"

    big = vectors(80_000, 2)
    with pytest.raises(ValueError, match="memory_mb"):
        db.add([f"w{i}" for i in range(len(big))], big, ["{}"] * len(big))
    assert db.memory_report()["count"] == flat_capacity + 1
    db.save()
    db.close()

    with srvdb.open(path) as db:
        report = db.memory_report()
        assert (report["mode"], report["bits"]) == ("rabitq", 4)
        assert report["budget_bytes"] == 1 << 20


def test_memory_mb_is_validated_in_the_core(tmp_path):
    with pytest.raises(ValueError, match="memory_mb"):
        srvdb.create(str(tmp_path / "a"), DIM, index="auto", memory_mb=0)
    with pytest.raises(ValueError, match="memory_mb"):
        srvdb.create(str(tmp_path / "b"), DIM, index="hnsw", memory_mb=50)


def test_report_without_budget(tmp_path):
    with srvdb.create(str(tmp_path / "db"), DIM, index="flat") as db:
        report = db.memory_report()
        assert report["budget_bytes"] is None
        assert report["capacity"] is None
        assert report["vector_bytes"] == 0
