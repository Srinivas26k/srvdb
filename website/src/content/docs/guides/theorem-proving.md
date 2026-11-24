---
title: Premise retrieval for theorem proving
description: Find Lean lemmas that help with a proof step, using vector, keyword and hybrid search plus dependency links, fully offline.
---

A theorem prover that uses retrieval needs to find, for a given proof state,
the library lemmas (premises) likely to help with the next step. The
`examples/theorem_proving/` example in the repository does this for a
sample of 207 Lean mathlib lemmas and measures recall@k on 45 labelled proof
steps. It runs offline: TF-IDF and TruncatedSVD from scikit-learn produce the
embeddings, so there is nothing to download. It ranks lemmas and does not
prove anything.

```bash
pip install -r examples/theorem_proving/requirements.txt
python examples/theorem_proving/premise_retrieval.py
```

## The pattern

Each lemma is one record. Its statement is embedded for vector search,
`text` holds its name and statement for BM25, and a `uses` link points to
each lemma its proof relies on. The vectors below are placeholders. In the
example they come from the embedding of each statement.

```python
import json
import srvdb

lemmas = {
    "Nat.Prime.dvd_mul": "p ∣ m * n ↔ p ∣ m ∨ p ∣ n",
    "Nat.Prime.coprime_iff_not_dvd": "Nat.Coprime p n ↔ ¬p ∣ n",
    "Nat.Coprime.dvd_of_dvd_mul_left": "k ∣ m * n → k ∣ m",
}
vectors = [[float(i == j) for j in range(8)] for i in range(len(lemmas))]
metas = [json.dumps({"namespace": "Nat", "text": f"{n} {s}"}) for n, s in lemmas.items()]

with srvdb.create("./premises", dimension=8, index="flat", exist_ok=True) as db:
    db.upsert(list(lemmas), vectors, metas)
    db.enable_keyword_search("text")
    db.link("Nat.Prime.dvd_mul", "uses", "Nat.Prime.coprime_iff_not_dvd")
    db.link("Nat.Prime.dvd_mul", "uses", "Nat.Coprime.dvd_of_dvd_mul_left")

    goal = vectors[0]  # embedding of the proof state
    print([i for i, _ in db.search(None, 1, text="dvd_mul")])
    # ['Nat.Prime.dvd_mul']
    print([(i, hop) for i, _, hop in db.search(goal, 1, expand=("uses", 1))])
    # [('Nat.Prime.dvd_mul', 0), ('Nat.Coprime.dvd_of_dvd_mul_left', 1), ('Nat.Prime.coprime_iff_not_dvd', 1)]
    print(len(db.search(goal, 3, text="dvd mul", where={"namespace": "Nat"})))
    # 3
```

- `db.search(q, k)` ranks lemmas by vector similarity.
- `db.search(None, k, text=...)` finds lemmas by name or identifier (BM25).
- `db.search(q, k, text=...)` fuses both rankings.
- `expand=("uses", hops)` adds the lemmas the hits depend on, with a hop
  distance.
- `where=` restricts the results by metadata, for example to one namespace.

## Results on the bundled sample

The script computes this recall@k and prints it with the hardware and
versions:

| Method | @1 | @5 | @10 |
| --- | --- | --- | --- |
| vector | 0.711 | 0.922 | 0.944 |
| keyword | 0.511 | 0.789 | 0.833 |
| hybrid | 0.600 | 0.856 | 0.900 |
| hybrid + `uses` expansion | 0.633 | 0.911 | 0.922 |

These numbers come from an AMD Ryzen 5 3550H running Linux 7.0, Python
3.13.12, srvdb 0.3.0, numpy 2.5.5 and scikit-learn 1.9.1, with
`python examples/theorem_proving/premise_retrieval.py --no-demo`. Because the
labelled set is small and written next to the lemmas, these numbers show
that the pipeline works, not how it performs on real proofs. The expanded
row looks at more than k lemmas.

## Further steps

The example's README explains how to:

- **Swap in a real embedding model.** Replace the `Embedder` class with a
  local model, such as a sentence-transformers model or LeanDojo's ReProver
  retriever, and normalize its vectors.
- **Evaluate on LeanDojo Benchmark 4.** Download it yourself, then convert a
  split with `leandojo_to_tsv.py` and pass it with `--lemmas` and `--steps`.
  The example never downloads anything.

The lemma statements come from mathlib4 and Lean 4, both under Apache-2.0.
The `NOTICE` file in the example attributes them.
