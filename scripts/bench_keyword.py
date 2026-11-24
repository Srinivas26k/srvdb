"""Compare SrvDB keyword search (BM25, ADR 0015) with SQLite FTS5.

Runs on a BEIR-format dataset that you download yourself (the script makes
no network requests), for example SciFact from
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip::

    python scripts/bench_keyword.py --data ./scifact --repeat 3

The directory must hold ``corpus.jsonl``, ``queries.jsonl`` and
``qrels/test.tsv``. Reports indexing time, median and p95 query latency and
nDCG@10 for each engine. Publish numbers only with the hardware, OS,
dataset, commit hash and this command (AGENTS.md section 11), measured on a
quiet machine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

import srvdb

K = 10
DIM = 128  # SrvDB needs a vector per record; keyword search ignores it


def load(data: Path):
    corpus = {}
    with open(data / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = f"{doc.get('title', '')} {doc['text']}"
    queries = {}
    with open(data / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(data / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    return corpus, {q: queries[q] for q in qrels if q in queries}, qrels


def ndcg(ranked: list[str], rels: dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:K]))
    ideal = sorted(rels.values(), reverse=True)[:K]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def run_srvdb(corpus, queries, workdir):
    start = time.perf_counter()
    db = srvdb.create(str(Path(workdir) / "srvdb"), DIM, "flat")
    ids = list(corpus)
    zero = [1.0] + [0.0] * (DIM - 1)
    for i in range(0, len(ids), 1000):
        chunk = ids[i : i + 1000]
        db.add(chunk, [zero] * len(chunk), [json.dumps({"text": corpus[d]}) for d in chunk])
    db.enable_keyword_search("text")
    db.save()
    build = time.perf_counter() - start

    def search(text):
        try:
            return [d for d, _ in db.search(None, K, text=text)]
        except ValueError:  # a query with no letters or digits
            return []

    return build, search, db.close


def run_fts5(corpus, queries, workdir):
    start = time.perf_counter()
    con = sqlite3.connect(Path(workdir) / "fts.db")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, body)")
    con.executemany("INSERT INTO docs VALUES (?, ?)", corpus.items())
    con.commit()
    build = time.perf_counter() - start

    def search(text):
        # OR of quoted words, the usual way to run free text through FTS5
        words = [w for w in re.split(r"\W+", text) if w]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = con.execute(
            "SELECT id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (match, K),
        )
        return [r[0] for r in rows]

    return build, search, con.close


def measure(name, runner, corpus, queries, qrels, repeat):
    with tempfile.TemporaryDirectory() as workdir:
        build, search, close = runner(corpus, queries, workdir)
        times, scores = [], []
        for _ in range(repeat):
            for qid, text in queries.items():
                t0 = time.perf_counter()
                ranked = search(text)
                times.append(time.perf_counter() - t0)
                scores.append(ndcg(ranked, qrels[qid]))
        close()
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    print(
        f"{name:8s} build {build:7.2f} s   median {statistics.median(times) * 1e3:7.3f} ms"
        f"   p95 {p95 * 1e3:7.3f} ms   nDCG@{K} {statistics.mean(scores):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="BEIR dataset directory")
    parser.add_argument("--repeat", type=int, default=3, help="passes over the queries")
    args = parser.parse_args()
    corpus, queries, qrels = load(args.data)
    print(f"{len(corpus)} documents, {len(queries)} queries, srvdb {srvdb.__version__}, "
          f"sqlite {sqlite3.sqlite_version}")
    measure("srvdb", run_srvdb, corpus, queries, qrels, args.repeat)
    measure("fts5", run_fts5, corpus, queries, qrels, args.repeat)


if __name__ == "__main__":
    main()
