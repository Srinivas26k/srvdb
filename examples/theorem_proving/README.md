# Premise retrieval for Lean proof steps

Given a Lean proof state, find the library lemmas (premises) that are most
likely to help with the next step. This is the retrieval half of
retrieval-augmented theorem proving. The example does no proving: it ranks
lemmas, and you measure how good the ranking is.

The example runs fully offline. It needs no model download, no network and
no GPU.

## What it shows

- **Vector search**: embed the goal and find lemmas with similar statements.
- **Keyword search** (BM25) for theorem names and identifiers such as
  `dvd_mul` or `Prime`. Embeddings blur these.
- **Hybrid search**: vector and BM25 rankings fused in one call
  (`db.search(q, k, text=goal)`).
- **Graph expansion**: each lemma is linked to the lemmas its proof uses
  (`db.link(a, "uses", b)`). `expand=("uses", 2)` brings in those
  dependencies.
- **Filtered search**: `where={"namespace": "Nat"}`.
- **recall@k** on 45 labelled proof steps, each paired with the lemmas that
  help with it. The script computes this itself.

## Run

From the repository root:

```bash
pip install -r examples/theorem_proving/requirements.txt   # srvdb, numpy, scikit-learn
python examples/theorem_proving/premise_retrieval.py
```

These packages are only needed for this example. `srvdb` itself does not
depend on them.

Options: `--index hnsw` (default `flat`, which is exact), `--no-demo` (print
only the evaluation), and `--lemmas` / `--steps` to use your own TSV files.
The database is created in a temporary directory and deleted when the script
ends.

## Results

Output of `python examples/theorem_proving/premise_retrieval.py --no-demo`:

```
Indexed 207 lemmas, 121 'uses' links, dimension 64.

Mean recall@k over 45 labelled proof steps:
  method            @1      @5     @10
  vector         0.711   0.922   0.944
  keyword        0.511   0.789   0.833
  hybrid         0.600   0.856   0.900
  hybrid+uses    0.633   0.911   0.922
```

- Hardware: AMD Ryzen 5 3550H (x86_64, 8 logical CPUs)
- OS: Linux 7.0.0-43-generic (glibc 2.43)
- Software: Python 3.13.12, srvdb 0.3.0 (built from source), numpy 2.5.5,
  scikit-learn 1.9.1
- Dataset: `data/lemmas.tsv` (207 lemmas) and `data/proof_steps.tsv`
  (45 steps), both in this directory
- Command: the one above, run from the repository root. The script prints
  the commit hash.

How to read these numbers:

- They are deterministic. TF-IDF and SVD use a fixed seed, and `flat`
  search is exact.
- The set is small, and the goals were written next to the lemma
  statements, so the goals often share notation with the lemma that solves
  them. The numbers check that the pipeline works. They do not predict
  results on real proofs. Use the LeanDojo split below for that.
- `hybrid+uses` returns the k hits plus their direct dependencies, so it
  looks at more than k lemmas.
- On this sample, BM25 over Lean notation is weaker than character n-grams.
  Fusion therefore scores below vector-only search here. Keyword search
  works best for name lookups (see the demo output).

## Swap in a real embedding model

All embedding happens in the `Embedder` class. It takes a list of strings,
returns unit-length float32 vectors and has a `dimension` attribute, which
the script passes to `srvdb.create`. Replace it with any local model. For
example, with `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, corpus, dim):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # local copy
        self.dimension = self.model.get_sentence_embedding_dimension()

    def __call__(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
```

Download the model before you run the script, then set
`HF_HUB_OFFLINE=1` to keep it offline. Use a model trained on code or
mathematics, such as the ReProver retriever from LeanDojo, to get
representative numbers. Normalize the vectors: SrvDB scores are cosine
similarities.

## Run against LeanDojo's benchmark

[LeanDojo](https://leandojo.org) publishes premise-retrieval benchmarks
extracted from mathlib (LeanDojo Benchmark 4, DOI
[10.5281/zenodo.8040109](https://doi.org/10.5281/zenodo.8040109)). Check
its licence before you redistribute anything from it. Neither the script nor
SrvDB downloads it. Follow these steps yourself:

1. Download and extract the benchmark archive from the DOI above into a
   directory of your choice, e.g. `~/data/leandojo_benchmark_4/`. It
   contains `corpus.jsonl` and split directories such as `random/` and
   `novel_premises/`.
2. Convert a split into this example's TSV format (offline):

   ```bash
   python examples/theorem_proving/leandojo_to_tsv.py \
       ~/data/leandojo_benchmark_4/corpus.jsonl \
       ~/data/leandojo_benchmark_4/random/test.json \
       ~/data/leandojo_tsv --max-steps 2000
   ```

3. Run the evaluation on it:

   ```bash
   python examples/theorem_proving/premise_retrieval.py \
       --lemmas ~/data/leandojo_tsv/lemmas.tsv \
       --steps ~/data/leandojo_tsv/proof_steps.tsv --index hnsw --no-demo
   ```

LeanDojo has no lemma-to-lemma dependency edges, so the converted file has
an empty `uses` column, and `hybrid+uses` then equals `hybrid`. The corpus
holds about 150,000 premises. TF-IDF fitting and insertion take minutes and
several GB of memory. Retrieval in the LeanDojo paper also restricts
premises to those accessible from the current file. This example does not
do that, so its numbers are not directly comparable to the paper's.

## Files

- `premise_retrieval.py`: the example
- `leandojo_to_tsv.py`: converts a downloaded LeanDojo split
- `data/lemmas.tsv`: lemma name, module, statement, dependencies
- `data/proof_steps.tsv`: goal and useful lemmas
- `requirements.txt`: example-only dependencies
- `NOTICE`: attribution for the mathlib and Lean statements (Apache-2.0)
