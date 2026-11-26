"""Plain-language findings for REPORT.md, with every number read from results.json."""

from __future__ import annotations

DB, NY, GL, SY = (
    "dbpedia-openai-100k",
    "nytimes-256-angular",
    "glove-100-angular",
    "synthetic-768",
)


def findings(res: dict) -> str:
    d = res["datasets"]

    def r(ds: str, mode: str) -> str:
        return f"{d[ds]['modes'][mode]['recall_at_10']:.4f}"

    def b(ds: str, mode: str) -> str:
        m = d[ds]["modes"][mode]
        return f"{m['memory_report']['vector_bytes'] / m['n']:,.0f} B"

    def disk(ds: str, mode: str) -> str:
        return f"{d[ds]['modes'][mode]['disk_bytes_per_vector_total']:,.0f} B"

    def c(ds: str, index: str, field: str = "recall_at_10") -> str:
        row = next(x for x in d[ds].get("competitors", []) if x["index"] == index)
        v = row[field]
        return f"{v:.4f}" if field == "recall_at_10" else f"{v:,.2f} B"

    m1 = d[DB]["modes"]["rabitq1"]
    ratio = (
        d[DB]["modes"]["flat"]["memory_report"]["vector_bytes"]
        / m1["memory_report"]["vector_bytes"]
    )
    crash = res.get("crash_test") or {}
    lines = [
        "## Key findings",
        "",
        "Recall is recall@10 over the query set; bytes are search code bytes/vector.",
        "",
        f"- **High-dimensional LLM embeddings compress best.** On DBpedia OpenAI "
        f"ada-002 (1536-d, 100k), RaBitQ 1-bit scans {b(DB, 'rabitq1')} per vector "
        f"instead of {b(DB, 'flat')} ({ratio:.1f}x smaller) at recall@10 "
        f"{r(DB, 'rabitq1')} with default rerank; 2-bit {r(DB, 'rabitq2')}, "
        f"4-bit {r(DB, 'rabitq4')}.",
        f"- **Rerank is what keeps recall.** Without rerank RaBitQ 1-bit gives "
        f"{r(DB, 'rabitq1-norerank')} (DBpedia), {r(NY, 'rabitq1-norerank')} "
        f"(NYTimes), {r(GL, 'rabitq1-norerank')} (GloVe). Rerank reads the float32 "
        f"vectors RaBitQ keeps on disk, so its disk total per vector exceeds flat's "
        f"({disk(DB, 'rabitq1')} vs {disk(DB, 'flat')} on DBpedia). The saving is in "
        "what the scan reads and keeps hot, not in disk use.",
        f"- **Low dimensions hurt 1-bit codes.** GloVe-100 RaBitQ 1-bit: "
        f"{r(GL, 'rabitq1')}; use 2-bit ({r(GL, 'rabitq2')}, {b(GL, 'rabitq2')}) or "
        f"4-bit ({r(GL, 'rabitq4')}, {b(GL, 'rabitq4')}). NYTimes-256 1-bit: "
        f"{r(NY, 'rabitq1')}.",
        f"- **PQ is the weak spot (issue #95).** SrvDB PQ: {r(DB, 'pq')} (DBpedia), "
        f"{r(NY, 'pq')} (NYTimes), {r(GL, 'pq')} (GloVe), {r(SY, 'pq')} (synthetic). "
        "Prefer RaBitQ until #95 is fixed.",
        f"- **SQ8 is a safe 4x:** {r(DB, 'sq8')}, {r(NY, 'sq8')}, {r(GL, 'sq8')}, "
        f"{r(SY, 'sq8')} (DBpedia, NYTimes, GloVe, synthetic).",
        "- **Against FAISS and turbovec** (same data, queries, ground truth):",
        f"  - SrvDB RaBitQ with default rerank vs FAISS `IndexRaBitQ` + float32 "
        f"refine at the same factor, 1-bit: DBpedia {r(DB, 'rabitq1')} vs "
        f"{c(DB, 'IndexRaBitQ 1-bit + refine')}; NYTimes {r(NY, 'rabitq1')} vs "
        f"{c(NY, 'IndexRaBitQ 1-bit + refine')}; GloVe {r(GL, 'rabitq1')} vs "
        f"{c(GL, 'IndexRaBitQ 1-bit + refine')} (FAISS ahead on GloVe). Codes: "
        f"DBpedia 1-bit {b(DB, 'rabitq1')} vs "
        f"{c(DB, 'IndexRaBitQ 1-bit', 'code_bytes_per_vector')}.",
        f"  - Without rerank FAISS is higher: DBpedia 1-bit "
        f"{c(DB, 'IndexRaBitQ 1-bit')} vs SrvDB {r(DB, 'rabitq1-norerank')}; SrvDB "
        "leans more on its rerank step.",
        f"  - turbovec (TurboQuant) 4-bit on DBpedia: "
        f"{c(DB, 'TurboQuantIndex 4-bit')} at "
        f"{c(DB, 'TurboQuantIndex 4-bit', 'code_bytes_per_vector')}, no rerank; "
        f"SrvDB RaBitQ 4-bit {r(DB, 'rabitq4-norerank')} without rerank, "
        f"{r(DB, 'rabitq4')} with. turbovec cannot run GloVe-100 (dimension must "
        "be a multiple of 8).",
        f"  - Sign bits + float32 rerank (pgvector-style) is strong on DBpedia: "
        f"{c(DB, 'IndexBinaryFlat sign bits + rerank')}, but falls off at lower "
        f"dimensions: NYTimes {c(NY, 'IndexBinaryFlat sign bits + rerank')}, GloVe "
        f"{c(GL, 'IndexBinaryFlat sign bits + rerank')}.",
        f"  - FAISS PQ (M = dim/8) is also low: DBpedia "
        f"{c(DB, 'IndexPQ M=192 x 8-bit')}, GloVe {c(GL, 'IndexPQ M=10 x 8-bit')}; "
        "PQ at these code sizes is weak in general, not only in SrvDB.",
        f"- **NYTimes flat recall is {r(NY, 'flat')}, not 1.0.** The exact float32 "
        "scan disagrees with the ann-benchmarks ground truth on a few neighbours; "
        "NYTimes has many near-tied distances. Read compressed modes against this "
        "ceiling.",
    ]
    if crash.get("passed"):
        lines.append(
            f"- **Crash safety reproduced locally:** {crash['kills_total']:,} kill -9s "
            f"across {', '.join(crash['per_mode'])}; every committed vector verified "
            "after reopen, 0 committed records lost."
        )
    peak = max(m["peak_rss_mib"] for ds in d.values() for m in ds["modes"].values())
    lines.append(
        f"- **Peak RSS** stays at or under {peak:,.1f} MiB per mode, including the "
        "benchmark's own copy of the vectors."
    )
    return "\n".join(lines)
