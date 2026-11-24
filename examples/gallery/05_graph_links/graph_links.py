"""Graph links: follow ``next`` links from search hits to reach the answer.

Documents are often chunked so that a heading chunk names the subject and
the following chunk holds the details, which never repeat the subject's
name. A vector search for the subject then finds the heading but not the
details. This script stores synthetic manuals as chunk pairs, links each
heading to its body with ``db.link(heading, "next", body)``, and measures
how often the body chunk is returned:

- plain ``db.search(query, k)``;
- ``db.search(query, k, expand=("next", 1))``, which adds linked records
  (hop 1) after the hits (hop 0).

The chart compares the two hit rates measured in this run.

Run::

    python examples/gallery/05_graph_links/graph_links.py [--fast]
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

K = 3
DIM = 64
NOUNS = "pump valve motor sensor relay boiler filter fan gearbox compressor".split()
ADJ = "primary backup north south upper lower inner outer main spare".split()
STEPS = (
    "turn the dial clockwise then hold the reset button for ten seconds "
    "check the seal replace the gasket tighten bolts to spec record torque "
    "drain fluid flush line refill wait five minutes and inspect for leaks"
).split()


def chunks(n: int, rng: np.random.Generator):
    heads, bodies = [], []
    for i in range(n):
        name = f"{ADJ[i % 10]} {NOUNS[(i // 10) % 10]} unit{i}"
        heads.append(f"maintenance of the {name}")
        bodies.append(" ".join(rng.choice(STEPS, size=14)))
    return heads, bodies


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("05 graph links")
    rng = np.random.default_rng(5)
    n = 100 if args.fast else 1000
    heads, bodies = chunks(n, rng)
    texts = heads + bodies
    tfidf = TfidfVectorizer().fit(texts)
    svd = TruncatedSVD(DIM, random_state=0).fit(tfidf.transform(texts))
    vectors = _common.normalize(svd.transform(tfidf.transform(texts)))

    def embed(text: str) -> list[float]:
        return svd.transform(tfidf.transform([text]))[0].tolist()

    ids = [f"h{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "manuals"), dimension=DIM, index="flat") as db:
            db.add(ids, vectors.tolist(), [_common.meta(text=t) for t in texts])
            for i in range(n):
                db.link(f"h{i}", "next", f"b{i}")
            print(f"stored {db.count()} chunks with {n} 'next' links")
            print(f"neighbors of h0: {db.neighbors('h0')}\n")

            queries = [f"how do I service the {h.split('of the ')[1]}" for h in heads]
            plain = expanded = 0
            for i, q in enumerate(queries):
                vec = embed(q)
                plain += f"b{i}" in {hit[0] for hit in db.search(vec, k=K)}
                hits = db.search(vec, k=K, expand=("next", 1))
                expanded += f"b{i}" in {hit[0] for hit in hits}
            example = db.search(embed(queries[0]), k=2, expand=("next", 1))
            print(f"query: {queries[0]!r}")
            for id_, score, hop in example:
                print(f"  {id_:<6} hop={hop} score={score:.3f}")

    rates = {"search(k=3)": plain / n, 'search(k=3,\nexpand=("next", 1))': expanded / n}
    print()
    for label, rate in rates.items():
        print(f"body chunk found by {label.replace(chr(10), ' ')}: {rate:.1%}")

    plt = _common.pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(rates), list(rates.values()), color=["#999999", "#111111"], width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of queries whose answer chunk is returned")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Following links, {n} heading/body pairs (this run)")
    _common.save_chart(fig, args.out, "05_graph_links.png")


if __name__ == "__main__":
    main()
