"""Helpers shared by the framework adapters. Framework-free."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from srvdb import Database

#: Key under which the adapters store the chunk text in SrvDB metadata.
TEXT_KEY = "text"
#: Key under which the adapters store the user metadata in SrvDB metadata.
META_KEY = "metadata"


#: Most values SrvDB accepts in one ``$in`` / ``$nin`` list
#: (``srvdb::Filter::MAX_LIST_LEN`` in the core).
MAX_IN_VALUES = 10_000


def has_dotted_key(where: dict[str, Any]) -> bool:
    """Whether any field name in ``where`` contains a dot, at any level.

    SrvDB reads ``"a.b"`` as the field ``b`` inside ``a``; the adapters
    store user metadata flat, so such a key must be matched in Python.
    """
    for key, value in where.items():
        if key in ("$and", "$or") and isinstance(value, list):
            if any(isinstance(v, dict) and has_dotted_key(v) for v in value):
                return True
        elif key == "$not" and isinstance(value, dict):
            if has_dotted_key(value):
                return True
        elif "." in key:
            return True
    return False


def check_id_count(name: str, ids: Sequence[str]) -> None:
    """Raise ValueError if ``ids`` exceeds the SrvDB ``$in`` limit."""
    if len(ids) > MAX_IN_VALUES:
        raise ValueError(
            f"{name} has {len(ids)} ids; SrvDB filters accept at most "
            f"{MAX_IN_VALUES}. Split the query or filter on a metadata field."
        )


def new_ids(n: int) -> list[str]:
    """Return ``n`` fresh random string ids."""
    return [str(uuid.uuid4()) for _ in range(n)]


def load(db: Database, id_: str) -> dict[str, Any] | None:
    """Return the stored record for ``id_``, or None if the id is unknown."""
    try:
        raw = db.get(id_)
    except ValueError:  # SrvDB reports an unknown id as ValueError
        return None
    if raw is None:
        return None
    record = json.loads(raw)
    return record if isinstance(record, dict) else {}


def write(
    db: Database,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    records: Iterable[dict[str, Any]],
) -> None:
    """Upsert vectors with their records, then save so they are durable."""
    if len(ids) != len(embeddings):
        raise ValueError(
            f"got {len(ids)} ids but {len(embeddings)} embeddings; "
            "pass one embedding per id"
        )
    if not ids:
        return
    metas = [json.dumps(r, ensure_ascii=False) for r in records]
    db.upsert(list(ids), [[float(x) for x in e] for e in embeddings], metas)
    db.save()


def remove(db: Database, ids: Sequence[str]) -> int:
    """Delete ``ids`` (unknown ones are ignored) and save the delete."""
    if not ids:
        return 0
    removed = db.delete(list(ids))
    db.save()
    return removed


def nested(where: dict[str, Any]) -> dict[str, Any]:
    """Point the field names of a ``where`` filter at the user metadata.

    The adapters store user metadata under ``META_KEY``, so ``{"a": 1}``
    becomes ``{"metadata.a": 1}``; ``$and``, ``$or`` and ``$not`` are
    rewritten recursively.
    """
    out: dict[str, Any] = {}
    for key, value in where.items():
        if key in ("$and", "$or") and isinstance(value, list):
            out[key] = [nested(v) if isinstance(v, dict) else v for v in value]
        elif key == "$not" and isinstance(value, dict):
            out[key] = nested(value)
        elif key.startswith("$"):
            out[key] = value  # the core rejects it with a clear message
        else:
            out[f"{META_KEY}.{key}"] = value
    return out


def search(
    db: Database,
    query: Sequence[float],
    k: int,
    keep: Callable[[str, dict[str, Any]], bool] | None = None,
    where: dict[str, Any] | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Return up to ``k`` ``(id, cosine score, record)`` hits, best first.

    ``where`` is a native SrvDB filter over the stored record and is
    evaluated by the database, which returns ``k`` hits whenever ``k``
    records match. ``keep(id, record)`` is a Python predicate for filters
    SrvDB cannot express; the search then over-fetches, doubling the
    candidate count until ``k`` kept hits are found or every vector was seen.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    total = db.count()
    if total == 0:
        return []
    vector = [float(x) for x in query]
    fetch = min(total, k if keep is None else k * 4)
    while True:
        hits = []
        for id_, score in db.search(vector, fetch, where=where):
            record = load(db, id_)
            if record is None or (keep is not None and not keep(id_, record)):
                continue
            hits.append((id_, float(score), record))
            if len(hits) == k:
                return hits
        if keep is None or fetch >= total:
            return hits
        fetch = min(total, fetch * 2)
