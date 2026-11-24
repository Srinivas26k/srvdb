"""Hybrid search: vector, keyword (BM25) and both fused, on two query kinds.

Generates a synthetic support knowledge base: each article belongs to one
of eight topics and carries a unique ticket code such as ``KB-40213``.
Embeddings are TF-IDF reduced with TruncatedSVD (offline, no download),
which captures topics but blurs rare codes. Two query sets are measured:

- topic queries (a few topic words): precision@5, an article is relevant
  when it has the query's topic;
- code queries (a ticket code plus one topic word): hit@5, whether the
  article with that code is in the top 5.

Each query set runs vector-only, keyword-only (``text=`` with no vector)
and hybrid (vector plus ``text=``, fused with Reciprocal Rank Fusion). The
chart is a grouped bar chart of the six measured numbers.

Run::

    python examples/gallery/04_hybrid_search/hybrid_search.py [--fast]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

K = 5
DIM = 64
TOPICS = {
    "billing": "invoice refund payment charge card receipt subscription price",
    "login": "password login account reset email locked username verify",
    "shipping": "delivery package courier tracking address parcel delayed warehouse",
    "printer": "printer paper jam toner cartridge print queue driver",
    "network": "wifi router network connection signal ethernet dns latency",
    "mobile": "phone app battery screen update android iphone notification",
    "storage": "disk backup storage folder sync files quota restore",
    "security": "virus malware phishing firewall alert scan suspicious breach",
}


def corpus(n: int, rng: np.random.Generator):
    topics = list(TOPICS)
    docs, labels, codes = [], [], []
    filler = "please help customer issue problem question support today".split()
    for i in range(n):
        topic = topics[i % len(topics)]
        words = list(rng.choice(TOPICS[topic].split(), size=6))
        words += list(rng.choice(filler, size=4))
        code = f"KB-{10000 + i}"
        docs.append(f"{code} " + " ".join(rng.permutation(words)))
        labels.append(topic)
        codes.append(code)
    return docs, labels, codes


def run(db, embed, queries, mode: str) -> list[list[str]]:
    out = []
    for q in queries:
        vec = embed(q) if mode != "keyword" else None
        text = q if mode != "vector" else None
        out.append([hit[0] for hit in db.search(vec, k=K, text=text)])
    return out


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("04 hybrid search")
    rng = np.random.default_rng(4)
    n, nq = (400, 40) if args.fast else (4000, 200)
    docs, labels, codes = corpus(n, rng)
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z0-9-]+").fit(docs)
    svd = TruncatedSVD(DIM, random_state=0).fit(tfidf.transform(docs))

    def embed(text: str) -> list[float]:
        return svd.transform(tfidf.transform([text]))[0].tolist()

    vectors = _common.normalize(svd.transform(tfidf.transform(docs)))
    picks = rng.choice(n, size=nq, replace=False)
    topic_q = [" ".join(rng.choice(TOPICS[labels[i]].split(), 3)) for i in picks]
    code_q = [f"{codes[i]} {TOPICS[labels[i]].split()[0]}" for i in picks]
    print(f"corpus: {n} synthetic articles, {len(TOPICS)} topics; {nq} queries each\n")

    results = {}
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "kb"), dimension=DIM, index="flat") as db:
            db.add(
                [str(i) for i in range(n)],
                vectors.tolist(),
                [_common.meta(text=d, topic=t) for d, t in zip(docs, labels)],
            )
            db.enable_keyword_search("text")
            for mode in ("vector", "keyword", "hybrid"):
                hits = run(db, embed, topic_q, mode)
                prec = np.mean(
                    [
                        sum(labels[int(h)] == labels[i] for h in row) / K
                        for row, i in zip(hits, picks)
                    ]
                )
                hits = run(db, embed, code_q, mode)
                hit = np.mean([str(i) in row for row, i in zip(hits, picks)])
                results[mode] = (prec, hit)
                print(
                    f"{mode:<8} topic precision@{K}={prec:.3f}  code hit@{K}={hit:.3f}"
                )

    plt = _common.pyplot()
    fig, ax = plt.subplots()
    x = np.arange(2)
    shades = {"vector": "#bbbbbb", "keyword": "#777777", "hybrid": "#111111"}
    for j, (mode, vals) in enumerate(results.items()):
        ax.bar(x + (j - 1) * 0.26, vals, 0.26, label=mode, color=shades[mode])
    ax.set_xticks(x, [f"topic queries\nprecision@{K}", f"code queries\nhit@{K}"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper center")
    ax.set_title(f"Vector vs keyword vs hybrid, {n} articles (this run)")
    _common.save_chart(fig, args.out, "04_hybrid_search.png")


if __name__ == "__main__":
    main()
