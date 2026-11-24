"""LangChain and LlamaIndex: the same search through each adapter.

Stores one synthetic corpus three times: directly with ``db.add``,
through ``srvdb.integrations.langchain.SrvDBVectorStore`` and through
``srvdb.integrations.llamaindex.SrvDBVectorStore``, all with the same
offline TF-IDF + TruncatedSVD embeddings. For a set of queries it checks
that each adapter returns the same top-k ids as the direct search, runs a
filtered query through each, and measures mean latency per query. The
chart compares those latencies (adapter overhead over the direct call).

A framework that is not installed is skipped with a message, so the
script still runs with only the gallery requirements. Install them with
``pip install "srvdb[langchain]" "srvdb[llamaindex]"``.

Run::

    python examples/gallery/11_frameworks/frameworks.py [--fast]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

K = 5
DIM = 128
WORDS = {
    "kitchen": "oven pan knife recipe bake boil spice flour sugar butter",
    "garden": "soil seed water rose tomato prune compost weed shovel sun",
    "travel": "train ticket hotel passport flight luggage map beach city tour",
    "music": "guitar piano chord melody drum song tempo choir violin note",
}


def corpus(n: int, rng: np.random.Generator):
    topics = list(WORDS)
    texts = [" ".join(rng.choice(WORDS[topics[i % 4]].split(), 8)) for i in range(n)]
    return texts, [topics[i % 4] for i in range(n)]


def timed(fn, queries) -> tuple[list[list[str]], float]:
    with _common.Timer() as t:
        out = [fn(q) for q in queries]
    return out, 1000 * t.seconds / len(queries)


def with_langchain(path, texts, topics, embed, qvecs):
    from langchain_core.embeddings import Embeddings

    from srvdb.integrations.langchain import SrvDBVectorStore

    class LsaEmbeddings(Embeddings):
        def embed_documents(self, batch):
            return [embed(t) for t in batch]

        def embed_query(self, text):
            return embed(text)

    store = SrvDBVectorStore(srvdb.create(path, dimension=DIM), LsaEmbeddings())
    store.add_texts(
        texts, [{"topic": t} for t in topics], ids=[str(i) for i in range(len(texts))]
    )
    docs = store.similarity_search("guitar melody", k=2, filter={"topic": "music"})
    print("langchain  filtered:", [(d.id, d.metadata["topic"]) for d in docs])
    results = timed(
        lambda v: [
            d.id for d, _ in store.similarity_search_with_score_by_vector(v, k=K)
        ],
        qvecs,
    )
    store.database.close()
    return results


def with_llamaindex(path, texts, topics, embed, qvecs):
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import (
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    from srvdb.integrations.llamaindex import SrvDBVectorStore

    db = srvdb.create(path, dimension=DIM)
    store = SrvDBVectorStore(db)
    store.add(
        [
            TextNode(id_=str(i), text=t, metadata={"topic": c}, embedding=embed(t))
            for i, (t, c) in enumerate(zip(texts, topics))
        ]
    )
    flt = MetadataFilters(filters=[MetadataFilter(key="topic", value="garden")])
    res = store.query(
        VectorStoreQuery(
            query_embedding=embed("rose compost"), similarity_top_k=2, filters=flt
        )
    )
    print("llamaindex filtered:", [(n.node_id, n.metadata["topic"]) for n in res.nodes])
    results = timed(
        lambda v: (
            store.query(VectorStoreQuery(query_embedding=v, similarity_top_k=K)).ids
        ),
        qvecs,
    )
    db.close()
    return results


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment(
        "11 LangChain and LlamaIndex", extra=("langchain_core", "llama_index.core")
    )
    rng = np.random.default_rng(11)
    n, nq = (400, 30) if args.fast else (5000, 300)
    texts, topics = corpus(n, rng)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(texts)
    svd = TruncatedSVD(DIM, random_state=0).fit(tfidf.transform(texts))

    def embed(text: str) -> list[float]:
        return svd.transform(tfidf.transform([text]))[0].tolist()

    qvecs = [
        embed(" ".join(rng.choice(WORDS[t].split(), 3)))
        for t in rng.choice(list(WORDS), nq)
    ]
    latency = {}
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "direct"), dimension=DIM) as db:
            db.add(
                [str(i) for i in range(n)],
                [embed(t) for t in texts],
                [_common.meta(topic=t) for t in topics],
            )
            direct, latency["direct db.search"] = timed(
                lambda v: [i for i, _ in db.search(v, k=K)], qvecs
            )
        runners = {
            "langchain": ("langchain_core", with_langchain),
            "llamaindex": ("llama_index.core", with_llamaindex),
        }
        for name, (module, run) in runners.items():
            if importlib.util.find_spec(module.split(".")[0]) is None:
                print(f"{name}: not installed, skipped (pip install 'srvdb[{name}]')")
                continue
            ids, ms = run(str(tmp / name), texts, topics, embed, qvecs)
            same = np.mean([set(a) == set(b) for a, b in zip(ids, direct)])
            latency[f"{name} adapter"] = ms
            print(f"{name:<10} same top-{K} ids as direct search: {same:.1%}")

    print(f"\n{n} records, {nq} queries, k={K}")
    for name, ms in latency.items():
        print(f"{name:<20} {ms:.3f} ms/query")
    plt = _common.pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    # Every path is timed the same way: one query per call.
    bars = ax.bar(list(latency), list(latency.values()), color="#333333", width=0.5)
    _common.label_bars(ax, bars)
    ax.set_ylabel("mean latency per query (ms)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Adapter overhead, {n} records, k={K} (this machine)")
    _common.save_chart(fig, args.out, "11_frameworks.png")


if __name__ == "__main__":
    main()
