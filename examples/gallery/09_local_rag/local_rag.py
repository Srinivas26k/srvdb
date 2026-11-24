"""Local RAG retrieval over the SrvDB documentation, fully offline.

Splits the documentation pages in this repository
(``website/src/content/docs/{start,concepts,guides}``) into chunks at
``##`` headings, embeds them, and stores each chunk with its page and
heading as metadata. A small labelled question set (the page that answers
each question) gives hit@k for vector, keyword and hybrid retrieval; the
best chunk for each question is printed, which is the context you would
hand to a local LLM. The chart plots hit@k for k = 1 to 5.

Embeddings default to TF-IDF + TruncatedSVD (no download). Pass
``--model DIR`` to use a sentence-transformers model you have already
downloaded to ``DIR``; the script sets ``HF_HUB_OFFLINE=1`` so nothing is
fetched.

Run::

    python examples/gallery/09_local_rag/local_rag.py [--fast] [--model DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

DOCS = _common.GALLERY.parent.parent / "website/src/content/docs"
KS = (1, 2, 3, 4, 5)
QUESTIONS = [
    ("what is durable after save returns and what is lost in a crash", "durability"),
    (
        "how do I remove deleted vectors from disk to reclaim space",
        "delete-upsert-compact",
    ),
    ("can several threads search the same database at the same time", "concurrency"),
    (
        "restrict results to records whose metadata year is at least 2020",
        "filtered-search",
    ),
    ("combine BM25 keyword ranking with vector similarity", "keyword-search"),
    ("connect records with typed edges and follow them from search hits", "links"),
    ("keep the database under a fixed number of megabytes of RAM", "memory-budget"),
    ("which python exception is raised for a wrong vector dimension", "errors"),
    ("open a database written by srvdb 0.3 after upgrading", "upgrading"),
    ("use srvdb as a vector store in langchain", "frameworks"),
    ("how do graph layers and ef_search work in hnsw", "hnsw"),
    ("store one byte per dimension instead of a float", "sq8"),
    ("split vectors into subvectors with codebooks", "pq"),
    ("partition vectors into clusters and probe only a few", "ivf"),
    ("install the python package with pip", "installation"),
]


def chunks() -> list[tuple[str, str, str]]:
    """(page, heading, text) for every ## section of the docs pages."""
    out = []
    for folder in ("start", "concepts", "guides"):
        for page in sorted((DOCS / folder).glob("*.md")):
            body = page.read_text(encoding="utf-8").split("---", 2)[-1]
            for part in re.split(r"^## ", body, flags=re.M):
                title, _, text = part.partition("\n")
                if len(text.split()) >= 15:
                    out.append((page.stem, title.strip(), text.strip()))
    return out


def embedder(texts: list[str], model_dir: str | None):
    if model_dir:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_dir, local_files_only=True)
        return lambda batch: model.encode(batch, normalize_embeddings=True)
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(texts)
    svd = TruncatedSVD(128, random_state=0).fit(tfidf.transform(texts))
    return lambda batch: _common.normalize(svd.transform(tfidf.transform(batch)))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None)
    known, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("09 local RAG")
    data = chunks()
    texts = [f"{h}\n{t}" for _, h, t in data]
    embed = embedder(texts, known.model)
    vectors = np.asarray(embed(texts), dtype=np.float32)
    questions = QUESTIONS[:5] if args.fast else QUESTIONS
    print(f"embedding: {known.model or 'TF-IDF + TruncatedSVD(128)'}")
    print(
        f"{len(data)} chunks from {len({p for p, _, _ in data})} pages, "
        f"{len(questions)} labelled questions\n"
    )

    hit = {m: np.zeros(len(KS)) for m in ("vector", "keyword", "hybrid")}
    with _common.scratch_dir() as tmp:
        path = str(tmp / "docs-rag")
        with srvdb.create(path, dimension=vectors.shape[1], index="flat") as db:
            db.add(
                [str(i) for i in range(len(data))],
                vectors.tolist(),
                [_common.meta(page=p, heading=h, text=t) for p, h, t in data],
            )
            db.enable_keyword_search("text")
        with srvdb.open(path) as db:
            for q, page in questions:
                vec = embed([q])[0].tolist()
                calls = {
                    "vector": (vec, None),
                    "keyword": (None, q),
                    "hybrid": (vec, q),
                }
                for mode, (v, text) in calls.items():
                    pages = [
                        json.loads(db.get(i))["page"]
                        for i, _ in db.search(v, k=max(KS), text=text)
                    ]
                    hit[mode] += [page in pages[:k] for k in KS]
                top = json.loads(db.get(db.search(vec, k=1, text=q)[0][0]))
                print(f"Q: {q}\n   context: {top['page']} / {top['heading']}")

    print()
    for mode, counts in hit.items():
        rates = counts / len(questions)
        print(f"{mode:<8} " + "  ".join(f"hit@{k}={r:.2f}" for k, r in zip(KS, rates)))

    plt = _common.pyplot()
    fig, ax = plt.subplots()
    for (mode, counts), marker in zip(hit.items(), "osD"):
        ax.plot(KS, counts / len(questions), marker=marker, label=mode)
    ax.set_xticks(KS)
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylabel("share of questions whose page is retrieved")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(
        f"Retrieval over the SrvDB docs, {len(questions)} questions (this run)"
    )
    _common.save_chart(fig, args.out, "09_local_rag.png")


if __name__ == "__main__":
    main()
