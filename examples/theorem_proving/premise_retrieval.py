"""Premise retrieval for Lean proof steps with SrvDB, fully offline.

Given a proof state (a Lean goal), find lemmas that are likely to help.
Shows vector, keyword and hybrid search, graph expansion over lemma
dependencies, and recall@k on a small labelled set, all computed here.

Run (from the repository root)::

    pip install -r examples/theorem_proving/requirements.txt
    python examples/theorem_proving/premise_retrieval.py

The embedding is TF-IDF over character n-grams reduced with TruncatedSVD:
deterministic, no download, no network. It is a baseline, not a prover;
see README.md for swapping in a real embedding model and for running on
LeanDojo's benchmark split.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import srvdb

HERE = Path(__file__).resolve().parent
DIM = 64
KS = (1, 5, 10)


def read_tsv(path: Path, columns: int) -> list[list[str]]:
    """Rows of a tab-separated file, skipping blank lines and # comments."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) < columns - 1:
            raise ValueError(f"{path.name}: expected {columns} columns: {line[:60]!r}")
        rows.append((cells + [""] * columns)[:columns])
    return rows


def split_list(cell: str) -> list[str]:
    return [x for x in cell.split(",") if x]


def keyword_text(name: str, statement: str) -> str:
    """Words for BM25: the name split on . and _, plus the statement."""
    return f"{re.sub(r'[._]', ' ', name)} {name} {statement}"


class Embedder:
    """TF-IDF (character 2-4 grams) + TruncatedSVD, fitted on the lemmas.

    Swap this class for any model that maps text to a fixed-size vector;
    nothing else in the script depends on how vectors are made.
    """

    def __init__(self, corpus: list[str], dim: int) -> None:
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
        )
        matrix = self.tfidf.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        self.svd.fit(matrix)
        self.dimension = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        vectors = self.svd.transform(self.tfidf.transform(texts))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


def build(db, lemmas: list[list[str]], embed: Embedder) -> int:
    """Insert every lemma, enable keyword search, link dependencies."""
    names = [row[0] for row in lemmas]
    vectors = embed([row[2] for row in lemmas])
    metas = [
        json.dumps(
            {
                "name": name,
                "module": module,
                "namespace": name.split(".")[0] if "." in name else "_root_",
                "statement": statement,
                "text": keyword_text(name, statement),
            }
        )
        for name, module, statement, _ in lemmas
    ]
    db.upsert(names, vectors.tolist(), metas)
    db.enable_keyword_search("text")
    known, links = set(names), 0
    for name, _, _, uses in lemmas:
        for dep in split_list(uses):
            if dep in known:
                db.link(name, "uses", dep)
                links += 1
    return links


def searches(db, embed: Embedder, goal: str, k: int) -> dict[str, list[str]]:
    """Top-k lemma names for one goal from each retrieval method."""
    q = embed([goal])[0].tolist()
    return {
        "vector": [i for i, _ in db.search(q, k)],
        "keyword": [i for i, _ in db.search(None, k, text=goal)],
        "hybrid": [i for i, _ in db.search(q, k, text=goal)],
        "hybrid+uses": [
            i for i, _, _ in db.search(q, k, text=goal, expand=("uses", 1))
        ],
    }


def evaluate(
    db, embed: Embedder, steps: list[list[str]]
) -> dict[str, dict[int, float]]:
    """Mean recall@k: the share of a step's useful lemmas found in the top k."""
    totals: dict[str, dict[int, float]] = {}
    for goal, useful_cell in steps:
        useful = set(split_list(useful_cell))
        for k in KS:
            for method, ids in searches(db, embed, goal, k).items():
                hit = len(useful & set(ids)) / len(useful)
                totals.setdefault(method, {}).setdefault(k, 0.0)
                totals[method][k] += hit
    return {
        m: {k: v / len(steps) for k, v in by_k.items()} for m, by_k in totals.items()
    }


def show(title: str, rows) -> None:
    print(f"\n{title}")
    for row in rows:
        print("  " + "  ".join(str(c) for c in row))


def demo(db, embed: Embedder) -> None:
    goal = "p a b : ℕ hp : p.Prime h : p ∣ a * b ⊢ p ∣ a ∨ p ∣ b"
    q = embed([goal])[0].tolist()
    print(f"\nGoal: {goal}")
    show(
        "Vector search (embedding of the goal):",
        [(i, f"{s:.3f}") for i, s in db.search(q, 5)],
    )
    show(
        "Keyword search for a theorem name (BM25):",
        [(i, f"{s:.3f}") for i, s in db.search(None, 5, text="dvd_mul Prime")],
    )
    show(
        "Hybrid search (vector + BM25, RRF):",
        [(i, f"{s:.4f}") for i, s in db.search(q, 5, text=goal)],
    )
    show(
        "Hybrid hits expanded along 'uses' links (hop 0 = hit):",
        [(i, f"hop {h}") for i, _, h in db.search(q, 3, text=goal, expand=("uses", 2))],
    )
    show(
        "Vector search restricted to the Nat namespace (where=...):",
        [(i, f"{s:.3f}") for i, s in db.search(q, 3, where={"namespace": "Nat"})],
    )
    print(
        "\nDirect dependencies of Nat.Prime.dvd_mul:",
        db.neighbors("Nat.Prime.dvd_mul", type="uses"),
    )


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def cpu_model() -> str:
    """CPU model name on Linux, else what platform reports."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass  # not Linux: fall back to platform.processor()
    return platform.processor() or platform.machine()


def environment() -> list[tuple[str, str]]:
    return [
        ("cpu", f"{cpu_model()} ({platform.machine()})"),
        ("cpus", str(os.cpu_count())),
        ("os", platform.platform()),
        ("python", sys.version.split()[0]),
        ("srvdb", getattr(srvdb, "__version__", "unknown")),
        ("numpy", np.__version__),
        ("scikit-learn", sklearn.__version__),
        ("commit", git_commit()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lemmas", type=Path, default=HERE / "data/lemmas.tsv")
    parser.add_argument("--steps", type=Path, default=HERE / "data/proof_steps.tsv")
    parser.add_argument(
        "--index", default="flat", help="SrvDB index mode (flat is exact)"
    )
    parser.add_argument(
        "--no-demo", action="store_true", help="only print the evaluation"
    )
    args = parser.parse_args()

    lemmas = read_tsv(args.lemmas, 4)
    steps = read_tsv(args.steps, 2)
    known = {row[0] for row in lemmas}
    steps = [s for s in steps if set(split_list(s[1])) & known]
    embed = Embedder([row[2] for row in lemmas], min(DIM, len(lemmas) - 1))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "premises")
        with srvdb.create(
            path, dimension=embed.dimension, index=args.index, exist_ok=True
        ) as db:
            links = build(db, lemmas, embed)
            db.save()
            print(
                f"Indexed {len(lemmas)} lemmas, {links} 'uses' links,"
                f" dimension {embed.dimension}."
            )
            if not args.no_demo:
                demo(db, embed)
            recall = evaluate(db, embed, steps)

    print(f"\nMean recall@k over {len(steps)} labelled proof steps:")
    print("  " + f"{'method':<12}" + "".join(f"{'@' + str(k):>8}" for k in KS))
    for method, by_k in recall.items():
        print("  " + f"{method:<12}" + "".join(f"{by_k[k]:>8.3f}" for k in KS))
    print("  (hybrid+uses returns the k hits plus their direct dependencies, so it")
    print("   looks at more than k lemmas; compare it with care.)")
    show("Environment:", environment())


if __name__ == "__main__":
    main()
