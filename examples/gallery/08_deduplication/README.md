# 08 Deduplication

Plants near-duplicates in the digits images: a 10 % sample is copied with
Gaussian pixel noise. All images go into a flat SrvDB database; each
record's nearest other record comes from `db.search(vector, k=2)`, and a
pair is flagged when its cosine similarity is at least a threshold.

```bash
python examples/gallery/08_deduplication/deduplication.py [--fast]
```

Output: flagged pairs, precision and recall against the planted pairs for
thresholds from 0.90 to 1.00, the threshold with the best F1, and
`output/08_deduplication.png`: precision (solid) and recall (dashed)
against the threshold, with the best-F1 threshold marked.

Distinct handwritten digits of the same class are often very similar, so
a loose threshold flags them as duplicates and precision drops; too strict
a threshold misses the noisy copies. Choose the threshold on a labelled
sample of your own data, as this script does.
